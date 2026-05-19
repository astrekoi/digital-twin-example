"""Pinata provider - mock httpx via respx."""
from __future__ import annotations

import respx
from httpx import Response

from src.integrations.ipfs.pinata import PinataProvider


def test_pinata_disabled_when_external_api_off():
    p = PinataProvider(jwt="x", allow_external_api_calls=False)
    res = p.pin_json({"hello": 1}, "key1")
    assert res.status == "disabled"


def test_pinata_disabled_when_no_jwt():
    p = PinataProvider(jwt="", allow_external_api_calls=True)
    res = p.pin_json({"hello": 1}, "key1")
    assert res.status == "disabled"


@respx.mock
def test_pinata_success_returns_cid():
    p = PinataProvider(jwt="testjwt", allow_external_api_calls=True)
    respx.post("https://api.pinata.cloud/pinning/pinJSONToIPFS").mock(
        return_value=Response(200, json={"IpfsHash": "QmTEST123", "PinSize": 42, "Timestamp": "x"})
    )
    res = p.pin_json({"hello": 1}, "key1")
    assert res.status == "ok"
    assert res.metadata["cid"] == "QmTEST123"


@respx.mock
def test_pinata_4xx_returns_error_no_retry():
    p = PinataProvider(jwt="testjwt", allow_external_api_calls=True)
    respx.post("https://api.pinata.cloud/pinning/pinJSONToIPFS").mock(
        return_value=Response(401, text="bad token")
    )
    res = p.pin_json({"hello": 1}, "key1")
    assert res.status == "error"
    assert "401" in res.message


@respx.mock
def test_pinata_5xx_retries_then_succeeds():
    p = PinataProvider(jwt="testjwt", allow_external_api_calls=True, timeout_s=2.0)
    route = respx.post("https://api.pinata.cloud/pinning/pinJSONToIPFS")
    route.side_effect = [
        Response(503, text="busy"),
        Response(200, json={"IpfsHash": "QmRetry"}),
    ]
    res = p.pin_json({"a": 1}, "k")
    assert res.status == "ok"
    assert res.metadata["cid"] == "QmRetry"
