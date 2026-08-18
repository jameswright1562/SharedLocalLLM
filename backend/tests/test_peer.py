from __future__ import annotations

import pytest

from sharedlocalllm_backend.errors import BackendError
from sharedlocalllm_backend.peer import PEER_PORT, _parse_endpoint


def test_manual_peer_endpoint_defaults_to_peer_port() -> None:
    assert _parse_endpoint("10.10.10.2") == ("10.10.10.2", PEER_PORT)
    assert _parse_endpoint("10.10.10.2:50000") == ("10.10.10.2", 50000)


def test_manual_peer_endpoint_rejects_hostnames() -> None:
    with pytest.raises(BackendError):
        _parse_endpoint("example.com")
