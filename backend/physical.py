"""
Physical-side anomaly rules.

The cyber side is learned from data; the physical side is rule-based, and
that is a deliberate choice rather than a shortcut. A facility has no labelled
history of break-ins to learn from, the rules here encode published access
control policy, and a guard has to be able to read the reason an alarm fired.
Each rule returns the anomaly name used by the risk model plus a plain-English
explanation that goes straight onto the dashboard.
"""

from __future__ import annotations

from datetime import datetime

# Normal working hours for the facility (24h clock, local time).
WORK_START_HOUR = 7
WORK_END_HOUR = 20

# Per-zone environmental limits.
TEMPERATURE_LIMIT_C = {"server_room": 27.0, "utility": 40.0}
POWER_NOMINAL_KW = {"utility": 120.0}
POWER_TOLERANCE = 0.25  # 25% deviation from nominal is an anomaly

BADGE_FAILURE_THRESHOLD = 3  # failed reads inside one window


def _is_after_hours(ts: float | None = None) -> bool:
    hour = datetime.fromtimestamp(ts).hour if ts else datetime.now().hour
    return hour < WORK_START_HOUR or hour >= WORK_END_HOUR


def evaluate(event: dict, badge_failures: int = 0) -> tuple[str | None, str]:
    """Classify one physical sensor event.

    Returns (anomaly_name, explanation). anomaly_name is None when the event
    is normal behaviour, in which case it is still logged but scores nothing.
    """
    kind = event.get("type")
    zone = event.get("zone_id", "unknown")

    if kind == "door_access":
        badge = event.get("badge_id", "UNKNOWN")
        if not event.get("authorised", True):
            return (
                "unauthorised_access",
                f"Badge {badge} is not authorised for {zone}.",
            )
        if event.get("forced"):
            return ("forced_door", f"Door in {zone} opened without a badge read.")
        if _is_after_hours(event.get("ts")):
            return (
                "after_hours_access",
                f"Badge {badge} entered {zone} outside working hours "
                f"({WORK_START_HOUR}:00-{WORK_END_HOUR}:00).",
            )
        return (None, f"Authorised entry by {badge}.")

    if kind == "badge_failure":
        if badge_failures + 1 >= BADGE_FAILURE_THRESHOLD:
            return (
                "repeated_badge_failure",
                f"{badge_failures + 1} failed badge reads at {zone} in quick succession.",
            )
        return (None, "Single failed badge read.")

    if kind == "camera":
        if event.get("tampered"):
            return ("camera_tamper", f"Camera feed in {zone} lost or obstructed.")
        return (None, "Camera healthy.")

    if kind == "temperature":
        value = float(event.get("value", 0))
        limit = TEMPERATURE_LIMIT_C.get(zone, 35.0)
        if value > limit:
            return (
                "temperature_critical",
                f"Temperature {value:.1f}C exceeds the {limit:.0f}C limit for {zone}.",
            )
        return (None, f"Temperature {value:.1f}C is within limits.")

    if kind == "power":
        value = float(event.get("value", 0))
        nominal = POWER_NOMINAL_KW.get(zone, 100.0)
        deviation = abs(value - nominal) / nominal
        if deviation > POWER_TOLERANCE:
            return (
                "power_anomaly",
                f"Power draw {value:.1f}kW deviates {deviation * 100:.0f}% "
                f"from the {nominal:.0f}kW nominal load.",
            )
        return (None, f"Power draw {value:.1f}kW is nominal.")

    if kind == "motion":
        if _is_after_hours(event.get("ts")) and event.get("detected"):
            return (
                "after_hours_access",
                f"Motion detected in {zone} outside working hours.",
            )
        return (None, "Motion within expected pattern.")

    return (None, f"Unhandled event type '{kind}'.")
