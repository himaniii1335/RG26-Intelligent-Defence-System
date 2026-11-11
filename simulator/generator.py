"""
Event generator for the RG26 Intelligent Defence System.

Two jobs:

  live  - continuous background traffic, so the dashboard is never empty: real
          NSL-KDD flows replayed against facility zones, plus routine sensor
          readings and the occasional genuine attack.

  demo  - the scripted incident used in the evaluation. A reconnaissance scan
          against the server room is followed, inside the correlation window,
          by an unauthorised badge at the server room door. Either event alone
          is ordinary; together they are the coordinated intrusion the project
          is built to catch.

Run:  python simulator/generator.py --mode demo
      python simulator/generator.py --mode live
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.preprocess import FEATURES, load_nsl_kdd  # noqa: E402

API = "http://127.0.0.1:8000"
TEST_FILE = ROOT / "KDDTest+.txt"

# Which zones carry network traffic, and which asset is the usual target.
NETWORK_ZONES = {
    "server_room": "SRV-01",
    "control_room": "WS-01",
    "perimeter": "GATE-01",
    "utility": "PWR-01",
}

BADGES_OK = ["EMP-1042", "EMP-2217", "EMP-3390"]
BADGE_BAD = "VIS-9981"


def post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=10)
        if r.status_code >= 400:
            print(f"  ! {path} -> {r.status_code} {r.text[:160]}")
            return None
        return r.json()
    except requests.RequestException as exc:
        print(f"  ! backend unreachable: {exc}")
        return None


def load_pool() -> dict[str, list[dict]]:
    """Group real test-set flows by attack family so a scenario can pick one."""
    if not TEST_FILE.exists():
        sys.exit(f"Missing {TEST_FILE}. The generator replays real NSL-KDD flows.")
    df = load_nsl_kdd(str(TEST_FILE))
    pool: dict[str, list[dict]] = {}
    for category, group in df.groupby("category"):
        pool[category] = group[FEATURES].to_dict("records")
    print(f"  flow pool: " + ", ".join(f"{k}={len(v)}" for k, v in pool.items()))
    return pool


def send_flow(pool: dict, category: str, zone: str, quiet: bool = False) -> dict | None:
    flows = pool.get(category)
    if not flows:
        return None
    res = post("/api/events/network", {
        "zone_id": zone,
        "asset_id": NETWORK_ZONES.get(zone),
        "flow": random.choice(flows),
        "true_label": category,
    })
    if res and not quiet:
        p = res["prediction"]
        mark = "THREAT" if p["is_threat"] else "ok    "
        print(f"  [{mark}] {zone:<13} truth={category:<7} "
              f"predicted={p['label']:<16} risk={res['zone']['risk']:.1f}")
    return res


def send_physical(payload: dict, quiet: bool = False) -> dict | None:
    res = post("/api/events/physical", payload)
    if res and not quiet:
        mark = "ANOMALY" if res["anomaly"] else "ok     "
        print(f"  [{mark}] {payload['zone_id']:<13} {payload['type']:<14} "
              f"{res['explanation']}")
    return res


def announce_combined(res: dict | None) -> None:
    if res and res.get("combined_alert"):
        a = res["combined_alert"]
        print("\n" + "!" * 70)
        print(f"  CORRELATED ALERT  risk={a['risk']}  zone={a['zone_id']}")
        print(f"  {a['title']}")
        print(f"  {a['detail']}")
        print("!" * 70 + "\n")


# ----------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------

def scenario_demo(pool: dict) -> None:
    print("\n--- Stage 1: normal operations -------------------------------")
    for _ in range(6):
        send_flow(pool, "Normal", random.choice(list(NETWORK_ZONES)))
        time.sleep(0.4)
    send_physical({"zone_id": "perimeter", "asset_id": "GATE-01",
                   "type": "door_access", "badge_id": random.choice(BADGES_OK),
                   "authorised": True})
    send_physical({"zone_id": "server_room", "asset_id": "SEN-02",
                   "type": "temperature", "value": 22.4})
    time.sleep(1.5)

    print("\n--- Stage 2: reconnaissance against the server room ----------")
    for _ in range(5):
        send_flow(pool, "Probe", "server_room")
        time.sleep(0.6)
    time.sleep(1.5)

    print("\n--- Stage 3: unauthorised badge at the server room door ------")
    res = send_physical({"zone_id": "server_room", "asset_id": "DOOR-02",
                         "type": "door_access", "badge_id": BADGE_BAD,
                         "authorised": False})
    announce_combined(res)
    time.sleep(1.5)

    print("\n--- Stage 4: privilege escalation attempt --------------------")
    for category in ["R2L", "U2R", "U2R"]:
        res = send_flow(pool, category, "server_room")
        announce_combined(res)
        time.sleep(0.8)

    res = send_physical({"zone_id": "server_room", "asset_id": "CAM-03",
                         "type": "camera", "tampered": True})
    announce_combined(res)

    print("\n--- Stage 5: final facility state ----------------------------")
    try:
        snap = requests.get(f"{API}/api/twin", timeout=10).json()
    except requests.RequestException as exc:
        print(f"  ! could not read twin state: {exc}")
        return
    print(f"  facility risk: {snap['facility_risk']}")
    for z in snap["zones"]:
        flag = " <-- CORRELATED" if z["correlated"] else ""
        print(f"    {z['name']:<28} {z['status']:<9} risk={z['risk']:>5.1f}"
              f"  cyber={z['cyber_score']:.2f} physical={z['physical_score']:.2f}{flag}")


def scenario_live(pool: dict, interval: float) -> None:
    print("  live mode: Ctrl+C to stop\n")
    weights = [("Normal", 0.72), ("DoS", 0.10), ("Probe", 0.10),
               ("R2L", 0.05), ("U2R", 0.03)]
    categories = [c for c, _ in weights]
    probs = [w for _, w in weights]
    tick = 0
    while True:
        tick += 1
        send_flow(pool, random.choices(categories, probs)[0],
                  random.choice(list(NETWORK_ZONES)))
        if tick % 5 == 0:
            send_physical({"zone_id": "server_room", "asset_id": "SEN-02",
                           "type": "temperature",
                           "value": round(random.uniform(20, 29), 1)})
        if tick % 11 == 0:
            res = send_physical({
                "zone_id": random.choice(["server_room", "armoury", "control_room"]),
                "asset_id": "DOOR-02", "type": "door_access",
                "badge_id": random.choice(BADGES_OK + [BADGE_BAD]),
                "authorised": random.random() > 0.3})
            announce_combined(res)
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="RG26 event generator")
    ap.add_argument("--mode", choices=["demo", "live"], default="demo")
    ap.add_argument("--interval", type=float, default=1.2,
                    help="seconds between events in live mode")
    ap.add_argument("--reset", action="store_true",
                    help="clear the twin and alert history before starting")
    args = ap.parse_args()

    try:
        health = requests.get(f"{API}/api/health", timeout=5).json()
    except requests.RequestException:
        sys.exit(f"Backend is not running. Start it first:\n"
                 f"  uvicorn backend.main:app --port 8000")
    print(f"  backend ok, {health['zones']} zones")

    if args.reset:
        post("/api/reset", {})
        print("  state reset")

    pool = load_pool()
    if args.mode == "demo":
        scenario_demo(pool)
    else:
        try:
            scenario_live(pool, args.interval)
        except KeyboardInterrupt:
            print("\n  stopped")


if __name__ == "__main__":
    main()
