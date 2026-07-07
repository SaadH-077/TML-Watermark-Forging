"""Leaderboard submission for TML 2026 Task 4 (watermark forgery).

Safety first: this NEVER posts unless you pass --yes. Default is a dry-run that validates
the zip + key and prints the pre-submission summary. The server enforces a 60-min cooldown
on success (2 min on error), so each real submit is a real cost.

Usage:
  python -m src.submit submissions/candidate_recon.zip            # dry-run (validate only)
  python -m src.submit submissions/candidate_recon.zip --yes      # actually submit

API key: put `TML_API_KEY=...` in a `.env` file at the project root (gitignored), or export
it in the environment. Never hard-code or log it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://34.63.153.158"          # from data/submission.py ("donot change")
TASK_ID = "22-forging-task"
SUBMIT_URL = f"{BASE_URL}/submit/{TASK_ID}"
LEADERBOARD = f"{BASE_URL}/leaderboard_page"


TEAM = "team_V"
SCORE_KEY = "22_forging_task::" + TEAM


def fetch_score(team: str = TEAM) -> float | None:
    """Read our current best public score from the leaderboard page (it embeds a
    `currentScores["22_forging_task::team_X"] = <float>` JS map)."""
    import re

    import requests

    r = requests.get(LEADERBOARD, timeout=30)
    m = re.search(r'currentScores\["22_forging_task::' + re.escape(team) + r'"\]\s*=\s*([0-9.]+)', r.text)
    return float(m.group(1)) if m else None


def fetch_standings(top: int = 12):
    """Return [(team, score)] for the forging task, best-first."""
    import re

    import requests

    r = requests.get(LEADERBOARD, timeout=30)
    rows = re.findall(r'currentScores\["22_forging_task::(team_[A-Z]+)"\]\s*=\s*([0-9.]+)', r.text)
    rows = sorted(((t, float(s)) for t, s in rows), key=lambda x: -x[1])
    return rows[:top]


def load_key() -> str | None:
    k = os.environ.get("TML_API_KEY")
    if k:
        return k.strip()
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line.startswith("TML_API_KEY") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def validate(zip_path: Path) -> dict:
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    expected = {f"{i}.png" for i in range(1, 201)}
    got = set(names)
    return {
        "ok": got == expected and len(names) == 200 and not any("/" in n for n in names),
        "n": len(names),
        "missing": sorted(expected - got)[:5],
        "unexpected": sorted(got - expected)[:5],
    }


def submit(zip_path: str, yes: bool = False):
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        sys.exit(f"File not found: {zip_path}")

    val = validate(zip_path)
    size_mb = zip_path.stat().st_size / 1e6
    print(f"zip: {zip_path}  ({size_mb:.1f} MB)")
    print(f"format valid: {val['ok']} | files: {val['n']} | "
          f"missing: {val['missing']} | unexpected: {val['unexpected']}")
    if not val["ok"]:
        sys.exit("ABORT: zip does not meet the 200-flat-PNG contract.")

    key = load_key()
    print("API key present:", bool(key))
    if not key:
        sys.exit("ABORT: no TML_API_KEY (put it in .env at project root or export it).")

    if not yes:
        print("\nDRY-RUN ok. Re-run with --yes to actually submit "
              "(this starts the 60-min cooldown).")
        return

    import requests

    print(f"\nPOST {SUBMIT_URL} ...")
    with open(zip_path, "rb") as f:
        files = {"file": (zip_path.name, f, "zip")}
        resp = requests.post(SUBMIT_URL, headers={"X-API-Key": key}, files=files)

    print("HTTP", resp.status_code)
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    print("server response:")
    print(json.dumps(body, indent=2)[:4000])
    print(f"\nleaderboard: {LEADERBOARD}")
    return resp.status_code, body


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    yes = "--yes" in sys.argv
    if not args:
        sys.exit("usage: python -m src.submit <zip> [--yes]")
    submit(args[0], yes=yes)
