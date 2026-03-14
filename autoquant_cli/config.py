from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def get_workspace_root() -> Path:
    value = os.getenv("AUTOQUANT_WORKSPACE", "").strip()
    if value:
        return Path(value).expanduser()
    return Path.home() / ".nanobot" / "workspace" / "autoquant"


ENV_FILE_PATH = get_workspace_root() / ".env"


def load_env() -> None:
    if ENV_FILE_PATH.exists():
        load_dotenv(ENV_FILE_PATH, override=False)


load_env()


def get_env(name: str, *fallback_names: str, required: bool = True) -> str | None:
    for key in (name, *fallback_names):
        value = os.getenv(key)
        if value:
            return value
    if not required:
        return None
    keys = ", ".join((name, *fallback_names))
    raise RuntimeError(f"Missing required environment variable: {keys}")


def get_backend_base_url() -> str:
    value = get_env("AUTOQUANT_API_URL")
    assert value is not None
    return value.rstrip("/")


def get_api_key() -> str:
    value = get_env("AUTOQUANT_API_KEY")
    assert value is not None
    return value


def get_massive_api_key() -> str:
    value = get_env("MASSIVE_API_KEY")
    assert value is not None
    return value
