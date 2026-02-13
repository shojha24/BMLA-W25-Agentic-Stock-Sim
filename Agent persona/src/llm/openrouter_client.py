from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        app_title: Optional[str] = None,
        timeout_s: int = 60,
    ):
        # Pull from env if not passed
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found. Put it in a .env file at the project root "
                "or export it in your shell."
            )

        self.site_url = site_url
        self.app_title = app_title
        self.timeout_s = timeout_s
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Optional attribution headers (fine to leave)
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_title:
            headers["X-Title"] = self.app_title

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"]