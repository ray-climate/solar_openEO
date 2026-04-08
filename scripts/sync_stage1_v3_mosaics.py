#!/usr/bin/env python
"""Download Stage-1-v3 mosaics from Google Drive with rclone."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction_pipeline import config as cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=cfg.RCLONE_REMOTE)
    parser.add_argument("--folder", default=cfg.DRIVE_FOLDER)
    parser.add_argument("--out-dir", default=str(cfg.MOSAICS_DIR))
    parser.add_argument("--transfers", type=int, default=8)
    parser.add_argument("--checkers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rclone",
        "copy",
        f"{args.remote}:{args.folder}",
        str(out_dir),
        "--transfers",
        str(args.transfers),
        "--checkers",
        str(args.checkers),
        "--include",
        "*.tif",
        "--progress",
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
