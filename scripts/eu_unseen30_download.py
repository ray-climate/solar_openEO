#!/usr/bin/env python
"""Download 2026 Mar-May stacks for 30 NEW EU-unseen sites (not in the first
88, not in v2 train/val). Login-node only (CDSE S3 unreachable from compute).
Reuses the throttled OpenEO download loop. Stacks -> europe_unseen30/stacks/.
"""
from __future__ import annotations
import csv, importlib.util, sys, time
from pathlib import Path

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))
OUT = REPO / "docs/europe_unseen30_2026MarMay"
NC = OUT / "stacks"
EXISTING = REPO / "docs/europe_unseen_2026MarMay_filtered/filter_summary.csv"
BACKEND = "https://openeo.dataspace.copernicus.eu"


def select_new(n=30):
    EU = importlib.util.spec_from_file_location("eu", REPO/"scripts/test_europe_unseen.py")
    eu = importlib.util.module_from_spec(EU); EU.loader.exec_module(eu)
    existing = {r["chip_id"] for r in csv.DictReader(EXISTING.open())}
    pool = eu.select_sites(249)                       # full eligible pool
    new = [s for s in pool if s["chip_id"] not in existing]
    # keep a mix of large + medium
    lg = [s for s in new if s["tier"] == "large"]
    md = [s for s in new if s["tier"] == "medium"]
    pick = lg[:max(8, n//4)] + md[:n - len(lg[:max(8, n//4)])]
    return pick[:n], eu


def main():
    NC.mkdir(parents=True, exist_ok=True)
    sites, eu = select_new(30)
    print(f"Selected {len(sites)} new sites "
          f"({sum(s['tier']=='large' for s in sites)} large, "
          f"{sum(s['tier']=='medium' for s in sites)} medium)", flush=True)
    # write the manifest so the render step uses the same selection
    with (OUT/"sites.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chip_id","tier","unseen_class","lat","lon","area_m2","half_size_km"])
        w.writeheader()
        for s in sites:
            w.writerow({k: s[k] for k in w.fieldnames})

    import openeo
    conn = openeo.connect(BACKEND); conn.authenticate_oidc()
    print("Authenticated.", flush=True)
    spec = importlib.util.spec_from_file_location("rw", REPO/"scripts/19_test_real_world_sites.py")
    rw = importlib.util.module_from_spec(spec); spec.loader.exec_module(rw)
    rw.TEMPORAL = ["2026-03-01", "2026-05-31"]

    pending = [s for s in sites if not (NC/f"{s['chip_id']}_stack.nc").exists()]
    print(f"{len(sites)-len(pending)} cached, {len(pending)} to download", flush=True)
    in_flight = {}; backoff = 0.0; t0 = time.time(); done = 0; fail = 0
    while pending or in_flight:
        now = time.time()
        if now >= backoff:
            while pending and len(in_flight) < 25:
                s = pending.pop(0)
                try:
                    job = rw._build_job(conn, s); job.start_job()
                except Exception as e:
                    pending.insert(0, s); backoff = now + 60
                    print(f"  SUBMIT_FAIL {s['chip_id']}: {str(e)[:80]}", flush=True); break
                in_flight[s["chip_id"]] = {"site": s, "job": job, "nc": NC/f"{s['chip_id']}_stack.nc"}
                print(f"  [{(time.time()-t0)/60:.1f}m] submit {s['chip_id']} inflight={len(in_flight)} pend={len(pending)}", flush=True)
        for cid, p in list(in_flight.items()):
            try: st = p["job"].status()
            except Exception as e: print(f"  POLL_FAIL {cid}: {e}", flush=True); continue
            if st == "finished":
                try:
                    tmp = NC/f"_tmp_{cid}"; tmp.mkdir(parents=True, exist_ok=True)
                    p["job"].get_results().download_files(tmp)
                    ncs = list(tmp.glob("*.nc"))
                    if not ncs: raise FileNotFoundError("no nc")
                    ncs[0].rename(p["nc"])
                    for x in tmp.iterdir(): x.unlink()
                    tmp.rmdir(); done += 1
                    print(f"  [{(time.time()-t0)/60:.1f}m] FINISHED {cid} ({p['nc'].stat().st_size/1e6:.0f}MB) [done={done}]", flush=True)
                except Exception as e:
                    fail += 1; print(f"  DL_FAIL {cid}: {str(e)[:80]}", flush=True)
                del in_flight[cid]
            elif st in ("error", "canceled"):
                fail += 1; print(f"  FAILED {cid}->{st}", flush=True); del in_flight[cid]
        if in_flight or pending: time.sleep(20)
    print(f"\nDownload done: {done} ok, {fail} failed", flush=True)


if __name__ == "__main__":
    main()
