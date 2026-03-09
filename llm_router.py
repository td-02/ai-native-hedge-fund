from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests


PROVIDER_STATS: dict[str, dict[str, float]] = {
    "groq": {"calls": 0.0, "failures": 0.0, "avg_latency": 0.0},
    "gemini": {"calls": 0.0, "failures": 0.0, "avg_latency": 0.0},
    "openrouter": {"calls": 0.0, "failures": 0.0, "avg_latency": 0.0},
    "ollama": {"calls": 0.0, "failures": 0.0, "avg_latency": 0.0},
}

_LAST_PROVIDER_USED = "none"
_LAST_LATENCY_MS = 0.0


@dataclass
class _ProviderResult:
    provider: str
    text: str
    latency_ms: float


def _update_stats(provider: str, latency_ms: float, success: bool) -> None:
    rec = PROVIDER_STATS.setdefault(provider, {"calls": 0.0, "failures": 0.0, "avg_latency": 0.0})
    rec["calls"] += 1.0
    if not success:
        rec["failures"] += 1.0
    c = max(1.0, rec["calls"])
    rec["avg_latency"] = ((rec["avg_latency"] * (c - 1.0)) + latency_ms) / c


def _with_json_instruction(prompt: str, json_mode: bool) -> str:
    if not json_mode:
        return prompt
    suffix = " Respond ONLY with valid JSON, no markdown, no explanation."
    return f"{prompt}{suffix}"


def _validated_json_text(text: str) -> str:
    obj = json.loads(text)
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _compose_messages(prompt: str, system: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _call_groq(prompt: str, system: str, timeout: int) -> _ProviderResult:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": _compose_messages(prompt, system),
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    text = str(data["choices"][0]["message"]["content"])
    return _ProviderResult("groq", text, latency_ms)


def _call_gemini(prompt: str, system: str, timeout: int) -> _ProviderResult:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    merged_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
    body = {
        "contents": [{"parts": [{"text": merged_prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000},
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=body, timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    text = str(data["candidates"][0]["content"]["parts"][0]["text"])
    return _ProviderResult("gemini", text, latency_ms)


def _call_openrouter(prompt: str, system: str, timeout: int) -> _ProviderResult:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": _compose_messages(prompt, system),
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    text = str(data["choices"][0]["message"]["content"])
    return _ProviderResult("openrouter", text, latency_ms)


def _call_ollama(prompt: str, system: str, timeout: int) -> _ProviderResult:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3").strip() or "llama3"
    url = f"{base_url}/api/chat"
    body = {
        "model": model,
        "stream": False,
        "messages": _compose_messages(prompt, system),
        "options": {"temperature": 0.1},
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=body, timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    text = str(data.get("message", {}).get("content", ""))
    return _ProviderResult("ollama", text, latency_ms)


def llm_chat(
    prompt: str,
    system: str = "",
    json_mode: bool = False,
    timeout: int = 15,
) -> str:
    global _LAST_PROVIDER_USED, _LAST_LATENCY_MS
    prepared_prompt = _with_json_instruction(prompt, json_mode=json_mode)
    providers = [_call_groq, _call_gemini, _call_openrouter, _call_ollama]
    for fn in providers:
        provider_name = fn.__name__.replace("_call_", "")
        try:
            result = fn(prepared_prompt, system, timeout)
            text = result.text.strip()
            if json_mode:
                text = _validated_json_text(text)
            _update_stats(provider_name, result.latency_ms, success=True)
            _LAST_PROVIDER_USED = provider_name
            _LAST_LATENCY_MS = result.latency_ms
            return text
        except Exception:
            _update_stats(provider_name, 0.0, success=False)
            continue

    # Deterministic terminal fallback if all providers fail.
    _LAST_PROVIDER_USED = "none"
    _LAST_LATENCY_MS = 0.0
    if json_mode:
        return "{}"
    return ""


def llm_chat_with_audit(
    prompt: str,
    system: str = "",
    json_mode: bool = False,
    event_type: str = "llm_call",
    audit_logger: Any = None,
) -> str:
    from hashlib import sha256

    t0 = time.perf_counter()
    text = llm_chat(prompt=prompt, system=system, json_mode=json_mode, timeout=15)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    response_hash = sha256(text.encode("utf-8")).hexdigest()
    token_estimate = int(max(1, len(text) // 4))
    provider_used = _LAST_PROVIDER_USED
    if audit_logger is not None and hasattr(audit_logger, "append"):
        try:
            run_id = sha256(f"{event_type}|{response_hash}".encode("utf-8")).hexdigest()[:16]
            audit_logger.append(
                event_type=event_type,
                run_id=run_id,
                payload={
                    "provider_used": provider_used,
                    "latency_ms": latency_ms if latency_ms > 0 else _LAST_LATENCY_MS,
                    "token_estimate": token_estimate,
                    "response_hash": response_hash,
                    "json_mode": json_mode,
                },
            )
        except Exception:
            pass
    return text


def get_provider_stats() -> dict[str, dict[str, float]]:
    return {k: dict(v) for k, v in PROVIDER_STATS.items()}

