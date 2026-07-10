#!/usr/bin/env python3
"""Refresh the README.md dashboard and tech-reference table.

Run by .github/workflows/refresh-readme.yml on every push to main (and weekly,
because GitHub's traffic API only retains 14 days of data). Also runnable
locally: `python scripts/refresh_readme.py` — without a token it skips traffic
and still refreshes the timestamp and tech table.

What it does:
1. Snapshots repo traffic (views/clones per day) into stats/traffic-history.json,
   merging with prior snapshots so popularity accumulates beyond GitHub's
   14-day window. Needs a token with push access to the repo: the workflow
   passes TRAFFIC_TOKEN (a classic PAT with `repo` scope) if configured,
   falling back to GITHUB_TOKEN (which GitHub may reject for traffic endpoints).
2. Rewrites README.md between <!-- STATS:START/END --> with the dashboard table.
3. Rewrites README.md between <!-- TECH:START/END --> from tech-stack.json,
   auto-detecting versions from repo pins (Cargo.toml, pyproject.toml, ...).
4. Updates the '**Last updated:**' line.

Stdlib only — no pip installs in CI.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
STACK_FILE = ROOT / "tech-stack.json"
HISTORY_FILE = ROOT / "stats" / "traffic-history.json"
REPO = os.environ.get("GITHUB_REPOSITORY", "trocca/ai-infra-gpu-engineering-ramp")
TOKEN = os.environ.get("TRAFFIC_TOKEN") or os.environ.get("GITHUB_TOKEN")


def api_get(path: str):
    url = f"https://api.github.com/repos/{REPO}{path}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "refresh-readme"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"  [skip] GET {path}: {exc}", file=sys.stderr)
        return None


def load_history() -> dict:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    return {"views": {}, "clones": {}}


def merge_traffic(history: dict) -> dict:
    """Merge the current 14-day traffic window into the accumulated history."""
    for kind in ("views", "clones"):
        data = api_get(f"/traffic/{kind}")
        if not data:
            continue
        for day in data.get(kind, []):
            date = day["timestamp"][:10]
            # Overwrite: GitHub's numbers for a given day only grow/stabilize.
            history[kind][date] = [day["count"], day["uniques"]]
    return history


def stats_table(history: dict) -> str:
    repo_info = api_get("") or {}
    stars = repo_info.get("stargazers_count", "–")
    forks = repo_info.get("forks_count", "–")
    watchers = repo_info.get("subscribers_count", "–")

    def totals(kind):
        days = history.get(kind, {})
        if not days:
            return None
        count = sum(v[0] for v in days.values())
        uniques = sum(v[1] for v in days.values())
        recent = sorted(days.items())[-14:]
        recent_count = sum(v[0] for _, v in recent)
        return count, uniques, recent_count, len(days)

    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| ⭐ Stars | {stars} |",
        f"| 🍴 Forks | {forks} |",
        f"| 👀 Watchers | {watchers} |",
    ]
    v = totals("views")
    if v:
        lines.append(f"| 📈 Views since tracking began | {v[0]} ({v[1]} daily-unique visitors) |")
        lines.append(f"| 📅 Views, last 14 days | {v[2]} |")
    c = totals("clones")
    if c:
        lines.append(f"| ⬇️ Clones since tracking began | {c[0]} ({c[1]} daily-unique cloners) |")
    if not v and not c:
        lines.append("| 📈 Traffic | _no snapshot yet — needs `TRAFFIC_TOKEN` secret (PAT with repo scope)_ |")
    tracked = totals("views")
    since = min(history["views"]) if history.get("views") else None
    footer = f"\n_Traffic tracked since {since}; refreshed on every push by GitHub Actions._" if since else ""
    return "\n".join(lines) + footer


def detect_version(entry: dict) -> str:
    det = entry.get("detect")
    fallback = entry.get("version", "n/a")
    if not det:
        return fallback
    found = set()
    for f in sorted(ROOT.glob(det["glob"])):
        m = re.search(det["pattern"], f.read_text(encoding="utf-8", errors="replace"))
        if m:
            found.add(m.group(1))
    return " / ".join(sorted(found)) if found else fallback


def tech_table() -> str:
    stack = json.loads(STACK_FILE.read_text(encoding="utf-8"))
    lines = [
        "| Tech | Area | Version | Where it's pinned / verified |",
        "|------|------|---------|------------------------------|",
    ]
    for e in stack["entries"]:
        version = detect_version(e)
        lines.append(f"| **{e['name']}** | {e['area']} | `{version}` | {e['source']} |")
    lines.append(
        "\n_Versions auto-detected from repo pins (`Cargo.toml`, `pyproject.toml`, "
        "`rust-toolchain.toml`) where possible, else from [tech-stack.json](tech-stack.json). "
        "Regenerated on every push._"
    )
    return "\n".join(lines)


def replace_section(text: str, marker: str, content: str) -> str:
    pattern = re.compile(
        rf"(<!-- {marker}:START -->).*?(<!-- {marker}:END -->)", re.DOTALL
    )
    if not pattern.search(text):
        print(f"  [warn] markers <!-- {marker}:START/END --> not found in README", file=sys.stderr)
        return text
    return pattern.sub(rf"\1\n{content}\n\2", text)


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    history = merge_traffic(load_history())
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    text = README.read_text(encoding="utf-8")
    text = re.sub(r"\*\*Last updated:\*\* .*", f"**Last updated:** {now}", text, count=1)
    text = replace_section(text, "STATS", stats_table(history))
    text = replace_section(text, "TECH", tech_table())
    README.write_text(text, encoding="utf-8")
    print(f"README refreshed at {now} "
          f"({len(history['views'])} view-days, {len(history['clones'])} clone-days tracked)")


if __name__ == "__main__":
    main()
