import json
import threading
import os
from typing import Optional

_LOCK = threading.Lock()
_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "statuses.json")


def _ensure_file():
    d = os.path.dirname(_PATH)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(_PATH):
        with open(_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)


def set_status(document_id: str, status: str, error: Optional[str] = None):
    _ensure_file()
    with _LOCK:
        with open(_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        entry = data.get(document_id, {})
        entry["status"] = status
        if error:
            entry["error"] = error
        data[document_id] = entry

        with open(_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)


def get_status(document_id: str):
    _ensure_file()
    with _LOCK:
        with open(_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

    return data.get(document_id, {"status": "uploaded"})
