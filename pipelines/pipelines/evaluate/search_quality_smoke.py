from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

BASE = "http://api:8080"

CASES = [
    {"query": "mens nike running shoes", "must_path": "Men", "must_brand": "Nike"},
    {"query": "mens jordan basketball", "must_path": "Basketball", "must_title_token": "jordan"},
    {"query": "womens running shoes", "must_path": "Women"},
    {"query": "boys shoes", "must_path": "Boys"},
    {"query": "girls sandals", "must_path": "Girls"},
]


def fetch(query: str) -> dict:
    url = f"{BASE}/search?{urllib.parse.urlencode({'q': query, 'k': 5})}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read())


def main() -> int:
    failures: list[str] = []
    for case in CASES:
        data = fetch(case["query"])
        results = data.get("results", [])
        if not results:
            failures.append(f"{case['query']}: no results")
            continue
        top = results[0]
        path = top.get("category_path") or []
        if case.get("must_path") and case["must_path"] not in path:
            failures.append(f"{case['query']}: top path {path} missing {case['must_path']}")
        if case.get("must_brand") and top.get("brand") != case["must_brand"]:
            failures.append(f"{case['query']}: top brand {top.get('brand')} != {case['must_brand']}")
        if case.get("must_title_token") and case["must_title_token"].lower() not in (top.get("title") or "").lower():
            failures.append(f"{case['query']}: top title {top.get('title')} missing {case['must_title_token']}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("search quality smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
