from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_CODEX_ENV_KEY = "OPENAI_API_KEY"
_MODELS_CACHE_TTL_SECONDS = 300.0
_MODELS_CACHE: tuple[tuple[str, str, str], float, list[dict[str, str]]] | None = None


def codex_config_path(env: Mapping[str, str] | None = None) -> Path:
    values = os.environ if env is None else env
    home = values.get("CODEX_HOME")
    if home:
        return Path(home) / "config.toml"
    return Path.home() / ".codex" / "config.toml"


def _read_config(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    path = codex_config_path(env)
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid Codex config TOML at {path}: {exc}") from exc


def _active_provider_block(data: dict[str, Any]) -> dict[str, Any]:
    provider_id = str(data.get("model_provider") or "openai")
    providers = data.get("model_providers")
    if not isinstance(providers, dict):
        return {}
    provider = providers.get(provider_id)
    return provider if isinstance(provider, dict) else {}


def active_provider_env_key(env: Mapping[str, str] | None = None) -> str:
    provider = _active_provider_block(_read_config(env))
    env_key = provider.get("env_key")
    if isinstance(env_key, str) and env_key.strip():
        return env_key.strip()
    return DEFAULT_CODEX_ENV_KEY


def active_provider_api_key(env: Mapping[str, str] | None = None) -> str | None:
    values = os.environ if env is None else env
    env_key = active_provider_env_key(values)
    val = values.get(env_key)
    return val if val else None


def active_provider_spawn_model(env: Mapping[str, str] | None = None) -> str:
    data = _read_config(env)
    configured = str(data.get("model") or "").strip()
    entries = _active_provider_model_entries(data, env)
    ids = {entry["id"] for entry in entries}
    if configured and (not entries or configured in ids):
        return configured
    return _select_fallback_model(entries) or configured


def _active_provider_model_entries(
    data: dict[str, Any],
    env: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    values = os.environ if env is None else env
    provider = _active_provider_block(data)
    base_url = str(provider.get("base_url") or "").rstrip("/")
    env_key = active_provider_env_key(values)
    api_key = values.get(env_key, "").strip()
    if not base_url or not api_key:
        return []

    cache_key = (base_url, env_key, api_key[-8:])
    global _MODELS_CACHE
    if _MODELS_CACHE is not None:
        cached_key, cached_at, cached_entries = _MODELS_CACHE
        if cached_key == cache_key and time.monotonic() - cached_at < _MODELS_CACHE_TTL_SECONDS:
            return [dict(entry) for entry in cached_entries]

    first_url = f"{base_url}/models?pageSize=100"
    body = _fetch_models_page(first_url, api_key)
    if body is None:
        return []

    seen: set[str] = set()
    entries = list(_extract_model_entries(body, seen))
    total = body.get("total")
    if isinstance(total, int) and total > len(entries):
        pages_needed = min((total + 99) // 100, 3)
        for page_num in range(2, pages_needed + 1):
            page_body = _fetch_models_page(
                f"{base_url}/models?pageSize=100&pageNum={page_num}",
                api_key,
            )
            if page_body is None:
                break
            entries.extend(_extract_model_entries(page_body, seen))

    if entries:
        _MODELS_CACHE = (cache_key, time.monotonic(), [dict(entry) for entry in entries])
    return entries


def _fetch_models_page(url: str, api_key: str) -> dict[str, Any] | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # nosec B310
            body = json.loads(resp.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None
    return body if isinstance(body, dict) else None


def _extract_model_entries(body: dict[str, Any], seen: set[str]) -> list[dict[str, str]]:
    raw_items = body.get("data")
    if not isinstance(raw_items, list):
        return []
    entries: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        ident = str(item.get("id") or "").strip()
        if not ident or ident in seen:
            continue
        seen.add(ident)
        name = str(item.get("modelName") or item.get("name") or ident).strip()
        entries.append({"id": ident, "name": name})
    return entries


def _select_fallback_model(entries: list[dict[str, str]]) -> str:
    for needle in ("gpt-5.5", "codex", "gpt-5", "o4-mini", "o3"):
        for entry in entries:
            haystack = f"{entry['id']} {entry['name']}".lower()
            if needle in haystack:
                return entry["id"]
    return entries[0]["id"] if entries else ""


def codex_auth_status(env: dict[str, str] | None = None) -> dict[str, Any]:
    env_key = active_provider_env_key(env)
    api_key = active_provider_api_key(env)
    return {"env_var": env_key, "has_env_key": bool(api_key)}
