import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Any

DEFAULT_SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
  key TEXT PRIMARY KEY,
  display_name TEXT,
  front_components TEXT,
  label_components TEXT,
  meta_json TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS inspections (
  barcode TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  device_type TEXT,
  front_status TEXT NOT NULL,
  front_components_json TEXT,
  label_status TEXT NOT NULL,
  label_components_json TEXT,
  barcode_status TEXT NOT NULL,
  overall TEXT NOT NULL,
  fail_phase TEXT,
  fail_reasons_json TEXT,
  inspected_at TEXT NOT NULL
);
"""


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(DEFAULT_SCHEMA)
    conn.commit()
    conn.close()


def upsert_device(
    path: Path,
    key: str,
    display_name: str,
    front_components: Any,
    label_components: Any,
    meta: dict | None = None,
) -> None:
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO devices(key, display_name, front_components, label_components, meta_json, created_at) VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(key) DO UPDATE SET display_name=excluded.display_name, front_components=excluded.front_components, label_components=excluded.label_components, meta_json=excluded.meta_json",
        (
            key,
            display_name,
            json.dumps(front_components, ensure_ascii=False),
            json.dumps(label_components, ensure_ascii=False),
            json.dumps(meta or {}, ensure_ascii=False),
            datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    conn.close()


def save_inspection(
    path: Path,
    *,
    barcode: str,
    session_id: str,
    device_type: str,
    front_status: str,
    front_components: Any,
    label_status: str,
    label_components: Any,
    barcode_status: str,
    overall: str,
    fail_phase: str = "",
    fail_reasons: list[str] | None = None,
) -> None:
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO inspections(
          barcode, session_id, device_type,
          front_status, front_components_json,
          label_status, label_components_json,
          barcode_status, overall, fail_phase, fail_reasons_json, inspected_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(barcode) DO UPDATE SET
          session_id=excluded.session_id,
          device_type=excluded.device_type,
          front_status=excluded.front_status,
          front_components_json=excluded.front_components_json,
          label_status=excluded.label_status,
          label_components_json=excluded.label_components_json,
          barcode_status=excluded.barcode_status,
          overall=excluded.overall,
          fail_phase=excluded.fail_phase,
          fail_reasons_json=excluded.fail_reasons_json,
          inspected_at=excluded.inspected_at
        """,
        (
            barcode,
            session_id,
            device_type or "",
            front_status,
            json.dumps(front_components, ensure_ascii=False),
            label_status,
            json.dumps(label_components, ensure_ascii=False),
            barcode_status,
            overall,
            fail_phase or "",
            json.dumps(fail_reasons or [], ensure_ascii=False),
            datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    conn.close()


def list_inspections(path: Path, limit: int = 12) -> list[dict]:
    if not path.is_file():
        return []
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    try:
        rows = cur.execute(
            """
            SELECT barcode, device_type, overall, fail_phase, inspected_at
            FROM inspections
            ORDER BY inspected_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return []
    conn.close()
    return [
        {
            "barcode": barcode or "",
            "device_type": device_type or "",
            "overall": overall or "",
            "fail_phase": fail_phase or "",
            "inspected_at": inspected_at or "",
        }
        for barcode, device_type, overall, fail_phase, inspected_at in rows
    ]


def load_profiles_into_db(path: Path, profiles: dict) -> None:
    if not profiles:
        return

    def _to_jsonable_components(components: Any) -> Any:
        # DeviceProfile.front_components/label_components: dict[str, ComponentSpec]
        if isinstance(components, dict):
            out: dict[str, Any] = {}
            for k, v in components.items():
                if hasattr(v, "count") and hasattr(v, "aliases"):
                    out[k] = {"count": int(getattr(v, "count")), "aliases": list(getattr(v, "aliases"))}
                else:
                    out[k] = v
            return out
        return components

    for key, profile in profiles.items():
        display = getattr(profile, "display_name", None) or (
            profile.get("display_name") if isinstance(profile, dict) else str(key)
        )
        front = getattr(profile, "front_components", None) or (
            profile.get("front_components") if isinstance(profile, dict) else []
        )
        label = getattr(profile, "label_components", None) or (
            profile.get("label_components") if isinstance(profile, dict) else []
        )
        upsert_device(
            path,
            key,
            display or "",
            _to_jsonable_components(front),
            _to_jsonable_components(label),
            {},
        )

