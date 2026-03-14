from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from autoquant_cli.config import get_api_key, get_backend_base_url

logger = logging.getLogger(__name__)


def normalize_api_path(path: str) -> str:
    normalized = path.strip()
    if not normalized:
        raise RuntimeError("API path cannot be empty")
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    if normalized.startswith("/api/v1/"):
        return normalized
    if normalized == "/api/v1":
        return normalized
    return f"/api/v1{normalized}"


def post_json(path: str, payload: dict[str, Any]) -> Any:
    url = f"{get_backend_base_url()}{normalize_api_path(path)}"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-API-Key": get_api_key(),
    }
    request = Request(url, data=data, headers=headers, method="POST")
    logger.info("POST %s", url)
    try:
        with urlopen(request) as response:
            text = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = body
        try:
            payload = json.loads(body)
            if isinstance(payload, dict) and payload.get("detail"):
                detail = str(payload["detail"])
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"Backend request failed with status {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Backend request failed: {exc.reason}") from exc
    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Backend returned invalid JSON: {text}") from exc


def get_openapi_json() -> Any:
    url = f"{get_backend_base_url()}/openapi.json"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
        },
        method="GET",
    )
    logger.info("GET %s", url)
    try:
        with urlopen(request) as response:
            text = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend request failed with status {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Backend request failed: {exc.reason}") from exc
    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Backend returned invalid JSON: {text}") from exc


def get_run(run_id: str) -> dict[str, Any]:
    payload = {
        "run_ids": [run_id],
        "page": 1,
        "limit": 1,
        "sort_by": "created_at",
        "sort_order": "desc",
    }
    rows = post_json("/run/get", payload)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError(f"Run not found: {run_id}")
    return rows[0]
