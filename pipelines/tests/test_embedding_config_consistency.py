import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def test_embedding_defaults_match_index_template_dimension():
    env = _env_example()
    template = json.loads((ROOT / "infra/opensearch/index_template.json").read_text())
    title_vec = template["template"]["mappings"]["properties"]["title_vec"]

    assert title_vec["dimension"] == int(env["EMBEDDER_DIM"])


def test_embedding_model_defaults_match_env_example():
    env = _env_example()
    model = env["EMBEDDER_MODEL"]
    dim = env["EMBEDDER_DIM"]

    assert f'EMBEDDER_MODEL", "{model}"' in (
        ROOT / "pipelines/pipelines/common/config.py"
    ).read_text()
    assert f'EMBEDDER_DIM", "{dim}"' in (
        ROOT / "pipelines/pipelines/common/config.py"
    ).read_text()
    assert f'EMBEDDER_MODEL:-{model}' in (ROOT / "docker-compose.yml").read_text()
    assert f'os.getenv("EMBEDDER_MODEL", "{model}")' in (
        ROOT / "services/embedder-py/app/model.py"
    ).read_text()
