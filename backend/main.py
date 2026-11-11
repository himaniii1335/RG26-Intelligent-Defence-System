"""
RG26 Intelligent Defence System - backend API.

One process ties the whole system together:

    network flow  -> hybrid detector -> twin zone -> risk -> alert
    sensor event  -> physical rules  -> twin zone -> risk -> alert

and raises a separate, higher-severity alert when the same zone shows both
inside the correlation window.

Run:  uvicorn backend.main:app --reload --port 8000
Then: http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import physical                     # noqa: E402
from backend.db import Database                   # noqa: E402
from backend.twin import (CRITICAL_THRESHOLD, ELEVATED_THRESHOLD,  # noqa: E402
                          FacilityTwin)

app = FastAPI(title="RG26 Intelligent Defence System", version="0.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

twin = FacilityTwin()
db = Database()
detector = None          # loaded lazily so the API still starts without a model
model_error: str | None = None

# Short-term counters that are not worth a database round trip.
badge_failures: dict[str, int] = defaultdict(int)
last_combined_alert: dict[str, float] = {}
COMBINED_ALERT_COOLDOWN = 20  # seconds, stops one incident spamming the panel


# ----------------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------------

class NetworkEvent(BaseModel):
    zone_id: str
    asset_id: str | None = None
    flow: dict = Field(..., description="Raw NSL-KDD flow features")
    true_label: str | None = Field(
        None, description="Ground truth, when replaying a labelled dataset"
    )


class PhysicalEvent(BaseModel):
    zone_id: str
    asset_id: str | None = None
    type: str
    value: float | None = None
    badge_id: str | None = None
    authorised: bool | None = True
    forced: bool | None = False
    tampered: bool | None = False
    detected: bool | None = False
    ts: float | None = None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def get_detector():
    """Load the model on first use and remember any failure."""
    global detector, model_error
    if detector is None and model_error is None:
        try:
            from ml.detector import HybridDetector
            detector = HybridDetector()
        except Exception as exc:
            model_error = str(exc)
    if detector is None:
        raise HTTPException(503, f"Detection model unavailable: {model_error}")
    return detector


def severity_for(risk: float) -> str:
    if risk >= CRITICAL_THRESHOLD:
        return "critical"
    if risk >= ELEVATED_THRESHOLD:
        return "warning"
    return "info"


def maybe_combined_alert(zone_id: str) -> dict | None:
    """Raise the cross-domain alert this project exists to produce."""
    state = twin.zones[zone_id].risk()
    if not state["correlated"]:
        return None

    now = time.time()
    if now - last_combined_alert.get(zone_id, 0) < COMBINED_ALERT_COOLDOWN:
        return None
    last_combined_alert[zone_id] = now

    cyber = [e for e in state["recent_events"] if e["domain"] == "cyber"]
    phys = [e for e in state["recent_events"] if e["domain"] == "physical"]
    return db.log_alert(
        zone_id=zone_id,
        severity="critical",
        kind="combined",
        title=f"Correlated physical and cyber activity in {state['name']}",
        detail=(
            f"{cyber[0]['label'] if cyber else 'network threat'} and "
            f"{phys[0]['label'] if phys else 'physical anomaly'} observed in the "
            f"same zone within the correlation window."
        ),
        risk=state["risk"],
        evidence={
            "cyber_events": cyber,
            "physical_events": phys,
            "cyber_score": state["cyber_score"],
            "physical_score": state["physical_score"],
            "reason": (
                "Either signal alone would be routine. Together, in one zone and "
                "one time window, they match the pattern of a coordinated intrusion, "
                "so the zone is escalated rather than scored as two separate incidents."
            ),
        },
    )


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": detector is not None,
        "model_error": model_error,
        "zones": len(twin.zones),
    }


@app.post("/api/events/network")
def ingest_network(event: NetworkEvent) -> dict:
    if event.zone_id not in twin.zones:
        raise HTTPException(404, f"Unknown zone '{event.zone_id}'")

    result = get_detector().predict(event.flow)
    db.log_event("cyber", event.zone_id, result["label"], result["is_threat"],
                 {"flow": event.flow, "prediction": result,
                  "true_label": event.true_label}, event.asset_id)

    alert = None
    if result["is_threat"]:
        twin.record_cyber(event.zone_id, result["label"], result["confidence"],
                          f"{result['label']} on {event.asset_id or event.zone_id}")
        state = twin.zones[event.zone_id].risk()
        alert = db.log_alert(
            zone_id=event.zone_id,
            severity=severity_for(state["risk"]),
            kind="cyber",
            title=f"{result['label']} detected in {state['name']}",
            detail=(
                f"Flagged by {result['source']} "
                f"({'anomaly strength' if result['anomaly_flag'] and result['rf_label'] == 'Normal' else 'confidence'}"
                f" {result['confidence'] * 100:.0f}%)."
            ),
            risk=state["risk"],
            evidence={
                "detector": result["source"],
                "rf_probabilities": result["rf_probabilities"],
                "anomaly_score": result["anomaly_score"],
                "top_features": result["top_features"],
                "asset_id": event.asset_id,
                "true_label": event.true_label,
            },
        )

    combined = maybe_combined_alert(event.zone_id)
    return {
        "prediction": result,
        "zone": twin.zones[event.zone_id].risk(),
        "alert": alert,
        "combined_alert": combined,
    }


@app.post("/api/events/physical")
def ingest_physical(event: PhysicalEvent) -> dict:
    if event.zone_id not in twin.zones:
        raise HTTPException(404, f"Unknown zone '{event.zone_id}'")

    payload = event.model_dump()
    key = f"{event.zone_id}:{event.badge_id or 'any'}"
    anomaly, explanation = physical.evaluate(payload, badge_failures[key])

    if event.type == "badge_failure":
        badge_failures[key] += 1
    elif event.type == "door_access" and event.authorised:
        badge_failures[key] = 0

    db.log_event("physical", event.zone_id, anomaly or event.type, anomaly is not None,
                 payload, event.asset_id)

    alert = None
    if anomaly:
        twin.record_physical(event.zone_id, anomaly, explanation)
        state = twin.zones[event.zone_id].risk()
        alert = db.log_alert(
            zone_id=event.zone_id,
            severity=severity_for(state["risk"]),
            kind="physical",
            title=f"{anomaly.replace('_', ' ').title()} in {state['name']}",
            detail=explanation,
            risk=state["risk"],
            evidence={"rule": anomaly, "event": payload, "asset_id": event.asset_id},
        )

    combined = maybe_combined_alert(event.zone_id)
    return {
        "anomaly": anomaly,
        "explanation": explanation,
        "zone": twin.zones[event.zone_id].risk(),
        "alert": alert,
        "combined_alert": combined,
    }


@app.get("/api/twin")
def get_twin() -> dict:
    snap = twin.snapshot()
    snap["stats"] = db.stats()
    return snap


@app.get("/api/alerts")
def get_alerts(limit: int = 50) -> dict:
    return {"alerts": db.recent_alerts(limit)}


@app.get("/api/model/metrics")
def model_metrics() -> dict:
    path = ROOT / "ml" / "artifacts" / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "Model has not been trained yet.")
    return json.loads(path.read_text())


@app.post("/api/reset")
def reset() -> dict:
    twin.reset()
    db.clear()
    badge_failures.clear()
    last_combined_alert.clear()
    return {"status": "reset"}


@app.get("/")
def dashboard() -> FileResponse:
    """Analyst view: zones, alerts and the evidence behind each decision."""
    index = ROOT / "dashboard" / "index.html"
    if not index.exists():
        raise HTTPException(404, "Dashboard not found.")
    return FileResponse(index)


@app.get("/3d")
def facility_3d() -> FileResponse:
    """Facility view: the same twin state rendered as the building layout."""
    page = ROOT / "dashboard" / "facility3d.html"
    if not page.exists():
        raise HTTPException(404, "3D view not found.")
    return FileResponse(page)


# Three.js ships with the repository rather than from a CDN, so the demo does
# not depend on internet access in the evaluation room.
_vendor = ROOT / "dashboard" / "vendor"
if _vendor.exists():
    app.mount("/vendor", StaticFiles(directory=str(_vendor)), name="vendor")
