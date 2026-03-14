from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from autoquant_cli.config import ENV_FILE_PATH, get_env


def _env_status(name: str) -> dict[str, Any]:
    value = get_env(name, required=False)
    return {
        "present": bool(value),
    }


def _ping_backend() -> dict[str, Any]:
    url = get_env("AUTOQUANT_API_URL", required=False)
    if not url:
        return {
            "configured": False,
            "ok": False,
            "error": "AUTOQUANT_API_URL is missing",
        }
    target = f"{url.rstrip('/')}/ping"
    request = Request(target, method="GET")
    try:
        with urlopen(request) as response:
            text = response.read().decode("utf-8")
        payload = json.loads(text) if text.strip() else {}
        return {
            "configured": True,
            "ok": True,
            "url": target,
            "response": payload,
        }
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "configured": True,
            "ok": False,
            "url": target,
            "status_code": exc.code,
            "error": body or exc.reason,
        }
    except URLError as exc:
        return {
            "configured": True,
            "ok": False,
            "url": target,
            "error": str(exc.reason),
        }


def health() -> dict[str, Any]:
    env = {
        "MASSIVE_API_KEY": _env_status("MASSIVE_API_KEY"),
        "AUTOQUANT_API_KEY": _env_status("AUTOQUANT_API_KEY"),
        "AUTOQUANT_API_URL": _env_status("AUTOQUANT_API_URL"),
    }
    env_ok = all(item["present"] for item in env.values())
    backend = _ping_backend()
    return {
        "status": "ok" if env_ok and backend["ok"] else "error",
        "env_vars_ok": env_ok,
        "env_file": str(ENV_FILE_PATH),
        "env_file_exists": ENV_FILE_PATH.exists(),
        "backend": backend,
    }
