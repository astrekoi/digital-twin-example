"""Pinata IPFS provider - pin JSON via the official REST API.

Endpoint: POST https://api.pinata.cloud/pinning/pinJSONToIPFS
Auth: Bearer JWT in Authorization header. Body: {"pinataContent": <json>, "pinataMetadata": {"name": <str>}}
Returns: {"IpfsHash": "<CID>", "PinSize": <int>, "Timestamp": "<iso>"}.

The provider returns IntegrationResult(status="ok", metadata={"cid": ...}) on
success, status="disabled" if not configured / not allowed, or status="error"
with message on any HTTP / JSON error.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from src.integrations.base import IntegrationResult
from src.integrations.ipfs import IPFSProvider

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 10.0
_RETRY_BACKOFFS_S = (1.0, 2.0)  # initial attempt + 2 retries


class PinataProvider(IPFSProvider):
    name = "pinata"

    def __init__(
        self,
        jwt: str | None,
        api_base: str = "https://api.pinata.cloud",
        allow_external_api_calls: bool = False,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._jwt = (jwt or "").strip()
        self._api_base = api_base.rstrip("/")
        self._allow = allow_external_api_calls
        self._timeout_s = timeout_s

    def is_configured(self) -> bool:
        return bool(self._jwt)

    def pin_json(self, payload: dict[str, Any], name: str) -> IntegrationResult:
        if not self._allow:
            return IntegrationResult(
                status="disabled",
                message="ALLOW_EXTERNAL_API_CALLS=false",
                provider=self.name,
            )
        if not self.is_configured():
            return IntegrationResult(
                status="disabled",
                message="PINATA_JWT is empty",
                provider=self.name,
            )

        url = f"{self._api_base}/pinning/pinJSONToIPFS"
        headers = {
            "Authorization": f"Bearer {self._jwt}",
            "Content-Type": "application/json",
        }
        body = {"pinataContent": payload, "pinataMetadata": {"name": name}}

        last_err: str | None = None
        with httpx.Client(timeout=self._timeout_s) as client:
            for attempt, backoff in enumerate((0.0,) + _RETRY_BACKOFFS_S):
                if backoff > 0:
                    time.sleep(backoff)
                try:
                    resp = client.post(url, headers=headers, json=body)
                except httpx.TimeoutException as exc:
                    last_err = f"timeout: {exc}"
                    continue
                except httpx.HTTPError as exc:
                    last_err = f"http error: {exc}"
                    continue
                if resp.status_code >= 500 or resp.status_code == 429:
                    last_err = f"http {resp.status_code}: {resp.text[:200]}"
                    continue
                if resp.status_code >= 400:
                    return IntegrationResult(
                        status="error",
                        message=f"http {resp.status_code}: {resp.text[:200]}",
                        provider=self.name,
                    )
                try:
                    data = resp.json()
                except ValueError:
                    last_err = "non-json response"
                    continue
                cid = data.get("IpfsHash")
                if not cid:
                    last_err = f"no IpfsHash in response: {data}"
                    continue
                return IntegrationResult(
                    status="ok",
                    message="pinned",
                    provider=self.name,
                    metadata={
                        "cid": cid,
                        "pin_size": data.get("PinSize"),
                        "timestamp": data.get("Timestamp"),
                    },
                )
        return IntegrationResult(
            status="error",
            message=last_err or "unknown pinata error",
            provider=self.name,
        )
