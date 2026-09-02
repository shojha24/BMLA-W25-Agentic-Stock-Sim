from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import requests

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        app_title: Optional[str] = None,
        timeout_s: int = 90,
        max_retries: int = 4,
    ):
        # Explicit argument wins; fall back to the environment.
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY not found. Add a line 'OPENROUTER_API_KEY=sk-or-...' "
                "to .env at the project root (a bare key with no VAR= prefix is not read "
                "by python-dotenv), or export it in your shell."
            )

        self.site_url = site_url
        self.app_title = app_title
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.last_usage: Dict[str, Any] = {}

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        json_mode: bool = True,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_title:
            headers["X-Title"] = self.app_title

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_err = ""
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout_s)
            except requests.RequestException as exc:
                last_err = f"network error: {exc}"
            else:
                if resp.status_code in RETRYABLE_STATUS:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                elif not resp.ok:
                    raise OpenRouterError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:500]}")
                else:
                    return self._parse(resp)

            if attempt < self.max_retries - 1:
                # Exponential backoff with jitter; free-tier models rate-limit often.
                time.sleep(min(2 ** attempt + random.random(), 20.0))

        raise OpenRouterError(f"OpenRouter failed after {self.max_retries} attempts. Last: {last_err}")

    def _parse(self, resp: requests.Response) -> str:
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise OpenRouterError(f"OpenRouter returned non-JSON body: {resp.text[:300]}") from exc

        # OpenRouter returns HTTP 200 with an {"error": ...} body for some failures.
        if isinstance(data, dict) and data.get("error"):
            raise OpenRouterError(f"OpenRouter error: {json.dumps(data['error'])[:500]}")

        self.last_usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(f"Unexpected OpenRouter payload: {json.dumps(data)[:500]}") from exc

        if content is None:
            reasoning = (data["choices"][0]["message"] or {}).get("reasoning")
            if reasoning:
                return str(reasoning)
            raise OpenRouterError("OpenRouter returned null content.")
        return content
