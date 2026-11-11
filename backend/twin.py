"""
Digital twin of the facility, plus the physical-cyber risk model.

This module is the part of the project that the literature review says is
missing elsewhere: most digital-twin security work scores network threats and
physical anomalies in separate pipelines. Here both land on the same zone
object, inside the same time window, and a zone that sees both at once is
scored higher than the sum of the two taken separately.

The module is deliberately free of FastAPI and database code so that the
scoring logic can be reasoned about, and tested, on its own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# Tunable parameters. Every one of these is a defensible choice, not a magic
# number, and they are all in one place so they can be justified in review.
# ----------------------------------------------------------------------------

WINDOW_SECONDS = 120        # how long an event keeps influencing its zone
CORRELATION_BONUS = 35      # extra risk when physical and cyber overlap
ELEVATED_THRESHOLD = 30     # risk at which a zone turns amber
CRITICAL_THRESHOLD = 70     # risk at which a zone turns red

# Severity per detected attack family. R2L and U2R rank highest because they
# represent an intruder already holding or escalating credentials, which is far
# worse inside a defence facility than a volumetric flood.
CYBER_SEVERITY = {
    "DoS": 0.60,
    "Probe": 0.40,
    "R2L": 0.85,
    "U2R": 0.95,
    "Unknown/Zero-day": 0.70,
}

# Severity per physical anomaly type.
PHYSICAL_SEVERITY = {
    "unauthorised_access": 0.85,
    "after_hours_access": 0.55,
    "camera_tamper": 0.80,
    "forced_door": 0.90,
    "temperature_critical": 0.50,
    "power_anomaly": 0.45,
    "repeated_badge_failure": 0.65,
}


@dataclass
class Asset:
    asset_id: str
    name: str
    kind: str  # "server" | "sensor" | "door" | "camera" | "utility"


@dataclass
class ZoneEvent:
    """One scored event retained inside a zone's sliding window."""
    at: float
    domain: str          # "cyber" or "physical"
    label: str
    severity: float
    detail: str


@dataclass
class Zone:
    zone_id: str
    name: str
    criticality: float          # 0-1, how much this zone matters
    assets: list[Asset] = field(default_factory=list)
    events: list[ZoneEvent] = field(default_factory=list)

    # -- window maintenance --------------------------------------------------

    def prune(self, now: float | None = None) -> None:
        now = now or time.time()
        self.events = [e for e in self.events if now - e.at <= WINDOW_SECONDS]

    def add(self, domain: str, label: str, severity: float, detail: str,
            at: float | None = None) -> None:
        self.events.append(
            ZoneEvent(at=at or time.time(), domain=domain, label=label,
                      severity=severity, detail=detail)
        )
        self.prune()

    # -- scoring -------------------------------------------------------------

    def _domain_score(self, domain: str, now: float) -> float:
        """Recency-weighted strongest signal, softened by repetition.

        Taking the maximum keeps one severe event from being diluted by noise;
        the small additive term means repeated events still push the score up.
        """
        relevant = [e for e in self.events if e.domain == domain]
        if not relevant:
            return 0.0
        weighted = [
            e.severity * (1.0 - (now - e.at) / WINDOW_SECONDS) for e in relevant
        ]
        weighted = [max(0.0, w) for w in weighted]
        if not weighted:
            return 0.0
        strongest = max(weighted)
        extra = sum(sorted(weighted, reverse=True)[1:]) * 0.15
        return min(1.0, strongest + extra)

    def risk(self, now: float | None = None) -> dict:
        now = now or time.time()
        self.prune(now)

        cyber = self._domain_score("cyber", now)
        physical = self._domain_score("physical", now)

        base = (cyber + physical) / 2 * 100
        correlated = cyber > 0 and physical > 0
        if correlated:
            base += CORRELATION_BONUS * min(cyber, physical)

        score = round(min(100.0, base * (0.6 + 0.4 * self.criticality)), 1)

        if score >= CRITICAL_THRESHOLD:
            status = "critical"
        elif score >= ELEVATED_THRESHOLD:
            status = "elevated"
        else:
            status = "normal"

        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "criticality": self.criticality,
            "cyber_score": round(cyber, 3),
            "physical_score": round(physical, 3),
            "correlated": correlated,
            "risk": score,
            "status": status,
            "asset_count": len(self.assets),
            "recent_events": [
                {"domain": e.domain, "label": e.label, "detail": e.detail,
                 "age": round(now - e.at, 1)}
                for e in sorted(self.events, key=lambda e: e.at, reverse=True)[:5]
            ],
        }


class FacilityTwin:
    """The live state of the facility: zones, assets and their current risk."""

    def __init__(self) -> None:
        self.zones: dict[str, Zone] = {}
        self._build_default_facility()

    def _build_default_facility(self) -> None:
        layout = [
            ("perimeter", "Perimeter & Main Gate", 0.5, [
                ("GATE-01", "Main gate controller", "door"),
                ("CAM-01", "Gate camera", "camera"),
                ("SEN-01", "Fence motion sensor", "sensor"),
            ]),
            ("server_room", "Server Room", 0.95, [
                ("SRV-01", "Application server", "server"),
                ("SRV-02", "Database server", "server"),
                ("DOOR-02", "Server room door", "door"),
                ("SEN-02", "Rack temperature sensor", "sensor"),
            ]),
            ("control_room", "Command & Control Room", 1.0, [
                ("WS-01", "Operator workstation", "server"),
                ("DOOR-03", "Control room door", "door"),
                ("CAM-03", "Control room camera", "camera"),
            ]),
            ("armoury", "Armoury & Secure Storage", 0.9, [
                ("DOOR-04", "Armoury vault door", "door"),
                ("SEN-04", "Vault vibration sensor", "sensor"),
            ]),
            ("utility", "Power & Utility Block", 0.6, [
                ("PWR-01", "Main power feed", "utility"),
                ("SEN-05", "Generator telemetry", "sensor"),
            ]),
        ]
        for zone_id, name, criticality, assets in layout:
            self.zones[zone_id] = Zone(
                zone_id=zone_id,
                name=name,
                criticality=criticality,
                assets=[Asset(a, n, k) for a, n, k in assets],
            )

    # -- ingest --------------------------------------------------------------

    def record_cyber(self, zone_id: str, label: str, confidence: float,
                     detail: str) -> float:
        """Register a detected network threat against a zone."""
        zone = self.zones[zone_id]
        severity = CYBER_SEVERITY.get(label, 0.5) * max(0.3, min(1.0, confidence))
        zone.add("cyber", label, severity, detail)
        return severity

    def record_physical(self, zone_id: str, anomaly: str, detail: str) -> float:
        """Register a physical anomaly against a zone."""
        zone = self.zones[zone_id]
        severity = PHYSICAL_SEVERITY.get(anomaly, 0.5)
        zone.add("physical", anomaly, severity, detail)
        return severity

    # -- read ----------------------------------------------------------------

    def snapshot(self) -> dict:
        now = time.time()
        zones = [z.risk(now) for z in self.zones.values()]
        return {
            "generated_at": now,
            "zones": zones,
            "facility_risk": round(max([z["risk"] for z in zones], default=0.0), 1),
            "zones_critical": sum(1 for z in zones if z["status"] == "critical"),
            "zones_elevated": sum(1 for z in zones if z["status"] == "elevated"),
            "correlated_zones": [z["zone_id"] for z in zones if z["correlated"]],
            "window_seconds": WINDOW_SECONDS,
        }

    def reset(self) -> None:
        for zone in self.zones.values():
            zone.events.clear()
