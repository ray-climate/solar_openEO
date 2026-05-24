#!/usr/bin/env python
"""Recover the 50-site OpenEO test: pick up existing `rw_*` jobs on CDSE,
download finished results, wait for the rest, then run inference.

Replaces re-submission when the original SLURM run died after CDSE accepted
all jobs but the local Python died before download.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))

# Load the test script so we can reuse select_sites + process_site
spec = importlib.util.spec_from_file_location(
    "rw", REPO / "scripts/19_test_real_world_sites.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

OUTPUT_DIR = REPO / "docs/realworld_unseen_50_2026Q1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sites = mod.select_sites(n_total=50, unseen_only=True, seed=42)
    wanted = {s["chip_id"]: s for s in sites}
    print(f"Target: {len(wanted)} sites.")

    import openeo
    conn = openeo.connect(mod.BACKEND_URL)
    conn.authenticate_oidc()

    # Map title (rw_<chip_id>) -> openeo job object, preferring latest finished
    all_jobs = conn.list_jobs()
    by_chip: dict[str, dict] = {}  # chip_id -> {"status":..., "id":..., "created":...}
    for j in all_jobs:
        title = j.get("title") or ""
        if not title.startswith("rw_"):
            continue
        cid = title[len("rw_"):]
        if cid not in wanted:
            continue
        rec = {"id": j["id"], "status": j.get("status"), "created": j.get("created", "")}
        prev = by_chip.get(cid)
        if prev is None:
            by_chip[cid] = rec
            continue
        # prefer finished; among same status, prefer newer
        prev_score = (prev["status"] == "finished", prev.get("created", ""))
        new_score = (rec["status"] == "finished", rec.get("created", ""))
        if new_score > prev_score:
            by_chip[cid] = rec

    have_local: list[str] = []
    for cid in wanted:
        nc = OUTPUT_DIR / f"{cid}_stack.nc"
        if nc.exists():
            have_local.append(cid)
    print(f"Already on disk: {len(have_local)}")

    matched_cids = [c for c in wanted if c in by_chip]
    missing_cids = [c for c in wanted if c not in by_chip and c not in have_local]
    print(f"Matched to existing CDSE jobs: {len(matched_cids)}")
    print(f"No job + no local file (will need fresh submission): {len(missing_cids)}")
    if missing_cids:
        print(f"  -> {missing_cids[:5]}{'...' if len(missing_cids)>5 else ''}")

    # Track and pull each
    started = time.time()
    n_done = len(have_local); n_failed = 0
    polling = {cid: by_chip[cid] for cid in matched_cids
               if cid not in have_local}

    # Use job.start_job() only if status is "created" (queue them up automatically)
    for cid, rec in list(polling.items()):
        if rec["status"] == "created":
            try:
                conn.job(rec["id"]).start_job()
                print(f"  [start] {cid} -> {rec['id']}")
            except Exception as exc:
                print(f"  [start-fail] {cid}: {exc}")

    print(f"\nPolling {len(polling)} jobs ...")
    while polling:
        for cid in list(polling.keys()):
            job = conn.job(polling[cid]["id"])
            try:
                status = job.status()
            except Exception as exc:
                print(f"  [{(time.time()-started)/60:5.1f} min] POLL_FAIL {cid}: {exc}")
                continue
            if status == "finished":
                nc_path = OUTPUT_DIR / f"{cid}_stack.nc"
                try:
                    tmp = OUTPUT_DIR / f"_tmp_{cid}"
                    tmp.mkdir(parents=True, exist_ok=True)
                    job.get_results().download_files(tmp)
                    ncs = list(tmp.glob("*.nc"))
                    if not ncs:
                        raise FileNotFoundError(f"No netCDF for {cid}")
                    ncs[0].rename(nc_path)
                    for f in tmp.iterdir():
                        f.unlink()
                    tmp.rmdir()
                    size_mb = nc_path.stat().st_size / 1e6
                    n_done += 1
                    print(f"  [{(time.time()-started)/60:5.1f} min] FINISHED {cid} ({size_mb:.1f} MB)   "
                          f"[done={n_done}, pending={len(polling)-1}]")
                except Exception as exc:
                    print(f"  [{(time.time()-started)/60:5.1f} min] DOWNLOAD_FAIL {cid}: {exc}")
                    n_failed += 1
                del polling[cid]
            elif status in ("error", "canceled"):
                n_failed += 1
                print(f"  [{(time.time()-started)/60:5.1f} min] FAILED {cid} -> {status}")
                del polling[cid]
        if polling:
            time.sleep(20)

    # Submit fresh jobs for the missing ones (respecting throttle)
    if missing_cids:
        miss_sites = [wanted[c] for c in missing_cids]
        print(f"\nFresh submission for {len(miss_sites)} missing sites ...")
        mod.download_all_throttled(conn, miss_sites, OUTPUT_DIR, max_concurrent=20)

    print(f"\nDownload phase complete. n_done={n_done}, n_failed={n_failed}")

    # Inference phase
    print(f"\nMosaic + inference + render:")
    ok = 0
    for s in sites:
        nc = OUTPUT_DIR / f"{s['chip_id']}_stack.nc"
        res = mod.process_site(s, nc, OUTPUT_DIR)
        if res.get("status") == "ok":
            ok += 1
    print(f"\n{ok}/{len(sites)} sites processed. PNGs in {OUTPUT_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
