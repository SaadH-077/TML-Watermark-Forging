#!/usr/bin/env python3
"""Optional: re-download the FULL course dataset (all 8 schemes' carriers + all 200 targets).

This bundle already ships the subset build_best.py needs (data/extracted/clean_targets and
watermarked_sources/WM_3). Run this only if you want the other schemes' carriers too, e.g. to
re-derive the native forges from scratch with the encoders/ scripts.

    python3 fetch_data.py
"""
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
snapshot_download("SprintML/tml2026_task4", repo_type="dataset", local_dir=str(ROOT / "data"))
z = ROOT / "data" / "Dataset.zip"
if z.exists():
    with zipfile.ZipFile(z) as zf:
        zf.extractall(ROOT / "data")
    print("extracted ->", ROOT / "data" / "extracted")
else:
    print("Dataset.zip not found after download; check the HF repo layout.")
