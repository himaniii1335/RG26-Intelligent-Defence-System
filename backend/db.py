"""
SQLite persistence for events and alerts.

SQLite is used instead of PostgreSQL for this phase on purpose: it needs no
server, so the whole system runs from one command during evaluation. The
access layer below is the only place that touches SQL, so moving to
PostgreSQL later is a change to this file alone.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "ids.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    domain      TEXT    NOT NULL,          -- 'cyber' | 'physical'
    zone_id     TEXT    NOT NULL,
    asset_id    TEXT,
    label       TEXT    NOT NULL,
    is_threat   INTEGER NOT NULL DEFAULT 0,
    payload     TEXT    NOT NULL           -- original event, JSON
);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    zone_id     TEXT    NOT NULL,
    severity    TEXT    NOT NULL,          -- 'info' | 'warning' | 'critical'
    kind        TEXT    NOT NULL,          -- 'cyber' | 'physical' | 'combined'
    title       TEXT    NOT NULL,
    detail      TEXT    NOT NULL,
    risk        REAL    NOT NULL,
    evidence    TEXT    NOT NULL           -- explanation payload, JSON
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);
"""


class Database:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = str(path)
        # The API is single-process but FastAPI serves requests on a thread
        # pool, so the connection must not be pinned to its creating thread.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- writes --------------------------------------------------------------

    def log_event(self, domain: str, zone_id: str, label: str, is_threat: bool,
                  payload: dict, asset_id: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO events (ts, domain, zone_id, asset_id, label, is_threat, payload)"
            " VALUES (?,?,?,?,?,?,?)",
            (time.time(), domain, zone_id, asset_id, label, int(is_threat),
             json.dumps(payload, default=str)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def log_alert(self, zone_id: str, severity: str, kind: str, title: str,
                  detail: str, risk: float, evidence: dict) -> dict:
        ts = time.time()
        cur = self.conn.execute(
            "INSERT INTO alerts (ts, zone_id, severity, kind, title, detail, risk, evidence)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (ts, zone_id, severity, kind, title, detail, risk,
             json.dumps(evidence, default=str)),
        )
        self.conn.commit()
        return {
            "id": int(cur.lastrowid), "ts": ts, "zone_id": zone_id,
            "severity": severity, "kind": kind, "title": title,
            "detail": detail, "risk": risk, "evidence": evidence,
        }

    # -- reads ---------------------------------------------------------------

    def recent_alerts(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["evidence"] = json.loads(d["evidence"])
            out.append(d)
        return out

    def stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS total,"
            " SUM(CASE WHEN domain='cyber' THEN 1 ELSE 0 END) AS cyber,"
            " SUM(CASE WHEN domain='physical' THEN 1 ELSE 0 END) AS physical,"
            " SUM(is_threat) AS threats FROM events"
        ).fetchone()
        alerts = self.conn.execute(
            "SELECT COUNT(*) AS n,"
            " SUM(CASE WHEN kind='combined' THEN 1 ELSE 0 END) AS combined FROM alerts"
        ).fetchone()
        return {
            "events_total": row["total"] or 0,
            "events_cyber": row["cyber"] or 0,
            "events_physical": row["physical"] or 0,
            "threats_detected": row["threats"] or 0,
            "alerts_total": alerts["n"] or 0,
            "alerts_combined": alerts["combined"] or 0,
        }

    def clear(self) -> None:
        self.conn.executescript("DELETE FROM events; DELETE FROM alerts;")
        self.conn.commit()
