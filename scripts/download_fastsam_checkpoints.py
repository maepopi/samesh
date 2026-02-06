#!/usr/bin/env python3
"""
Download FastSAM checkpoints into the repo checkpoints/ folder.
Uses Ultralytics assets (FastSAM-x.pt, FastSAM-s.pt).
See: https://github.com/CASIA-LMC-Lab/FastSAM#model-checkpoints
     https://docs.ultralytics.com/models/fast-sam/
"""

import os
import sys
from pathlib import Path

# Repo root (parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

# Ultralytics assets release URLs (v8.4.0)
FASTSAM_URLS = {
    "FastSAM-x.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-x.pt",
    "FastSAM-s.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt",
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file with a simple progress indication."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0)) or None
            data = resp.read()
    except Exception as e:
        print(f"  Failed: {e}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    size_mb = len(data) / (1024 * 1024)
    print(f"  Saved {dest.name} ({size_mb:.1f} MB)")
    return True


def main():
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints directory: {CHECKPOINTS_DIR}")
    for name, url in FASTSAM_URLS.items():
        dest = CHECKPOINTS_DIR / name
        if dest.exists():
            print(f"[skip] {name} (already exists)")
            continue
        print(f"Downloading {name} ...")
        if not download_file(url, dest):
            print(f"  Retrying with requests if available...")
            try:
                import requests
                r = requests.get(url, stream=True, timeout=120)
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  Saved {dest.name} ({dest.stat().st_size / (1024*1024):.1f} MB)")
            except Exception as e:
                print(f"  Failed: {e}")
                sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
