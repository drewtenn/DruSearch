#!/usr/bin/env python3
"""Generate language-specific feature schemas from feature_schema.json.

This is the codegen step that backs the cross-language parity guarantee:
both training and serving code import generated files whose contents are a
deterministic function of feature_schema.json. Drift is caught by
`make check-feature-parity`, which regenerates and `git diff --exit-code`s.

Outputs:
  pipelines/pipelines/features/_generated.py
  services/api-go/internal/features/schema_generated.go
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "libs" / "schema" / "feature_schema.json"
PY_OUT = REPO_ROOT / "pipelines" / "pipelines" / "features" / "_generated.py"
GO_OUT = REPO_ROOT / "services" / "api-go" / "internal" / "features" / "schema_generated.go"

HEADER_NOTE = (
    "Generated from libs/schema/feature_schema.json by libs/schema/codegen.py.\n"
    "DO NOT EDIT BY HAND. Run `make check-feature-parity` after schema changes."
)

VALID_KINDS = {"FLOAT", "INT", "BOOL"}
VALID_SOURCES = {
    "RETRIEVAL",
    "STATIC_PRODUCT",
    "PRODUCT_AGG",
    "INTERACTION",
    "ONLINE_USER",
    "ONLINE_SESSION",
}


def load_schema() -> dict:
    with SCHEMA_PATH.open() as f:
        schema = json.load(f)
    features = schema["features"]
    for i, f in enumerate(features):
        if f["index"] != i:
            raise SystemExit(f"feature_schema.json: feature {f['name']} index {f['index']} != position {i} (append-only, no gaps)")
        if f["kind"] not in VALID_KINDS:
            raise SystemExit(f"feature_schema.json: {f['name']} kind {f['kind']} not in {sorted(VALID_KINDS)}")
        if f["source"] not in VALID_SOURCES:
            raise SystemExit(f"feature_schema.json: {f['name']} source {f['source']} not in {sorted(VALID_SOURCES)}")
    return schema


def render_python(schema: dict) -> str:
    lines = [
        '"""' + HEADER_NOTE.replace("\n", "\n") + '"""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "",
        f'SCHEMA_VERSION = "{schema["version"]}"',
        "",
        "@dataclass(frozen=True)",
        "class GeneratedFeature:",
        "    index: int",
        "    name: str",
        "    kind: str",
        "    source: str",
        "    description: str",
        "",
        "FEATURES: tuple[GeneratedFeature, ...] = (",
    ]
    for f in schema["features"]:
        desc = f["description"].replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f'    GeneratedFeature(index={f["index"]}, name="{f["name"]}", '
            f'kind="{f["kind"]}", source="{f["source"]}", description="{desc}"),'
        )
    lines += [
        ")",
        "",
        "FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)",
        "NUM_FEATURES: int = len(FEATURES)",
        "",
    ]
    # name -> index constants (idiomatic uppercase)
    for f in schema["features"]:
        const = "IDX_" + f["name"].upper()
        lines.append(f'{const} = {f["index"]}')
    lines.append("")
    return "\n".join(lines)


def render_go(schema: dict) -> str:
    lines = [
        "// " + HEADER_NOTE.replace("\n", "\n// "),
        "",
        "package features",
        "",
        f'const SchemaVersion = "{schema["version"]}"',
        "",
        "const (",
    ]
    for f in schema["features"]:
        const = "Gen" + _camel(f["name"])
        lines.append(f"\t{const} = {f['index']}")
    lines += [
        "",
        f"\tGenNumFeatures = {len(schema['features'])}",
        ")",
        "",
        "// GenNames is the ordered feature-name list aligned with the indices above.",
        "var GenNames = [...]string{",
    ]
    for f in schema["features"]:
        lines.append(f'\t"{f["name"]}",')
    lines += [
        "}",
        "",
    ]
    return "\n".join(lines)


def _camel(snake: str) -> str:
    return "".join(part.capitalize() for part in snake.split("_"))


def main(argv: list[str]) -> int:
    check = "--check" in argv
    schema = load_schema()
    py_new = render_python(schema)
    go_new = render_go(schema)

    if check:
        py_old = PY_OUT.read_text() if PY_OUT.exists() else ""
        go_old = GO_OUT.read_text() if GO_OUT.exists() else ""
        drift = []
        if py_old != py_new:
            drift.append(str(PY_OUT.relative_to(REPO_ROOT)))
        if go_old != go_new:
            drift.append(str(GO_OUT.relative_to(REPO_ROOT)))
        if drift:
            print("ERROR: feature schema drift detected. Re-run `make regen-feature-schema` and commit:", file=sys.stderr)
            for d in drift:
                print(f"  {d}", file=sys.stderr)
            return 1
        print("feature schema in sync")
        return 0

    PY_OUT.write_text(py_new)
    GO_OUT.write_text(go_new)
    print(f"wrote {PY_OUT.relative_to(REPO_ROOT)}")
    print(f"wrote {GO_OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
