"""Network workaround for hosts whose IPv6 route does not work.

On such a host every Python HTTPS request waits for the IPv6 connection attempt to
time out before trying IPv4, which turns a model download into a long stall.
Setting ``LEGALRAG_IPV4=1`` restricts name resolution to IPv4 for this process.
"""

from __future__ import annotations

import os
import socket

_original = socket.getaddrinfo


def _ipv4_only(host, port, family=0, *args, **kwargs):
    return _original(host, port, socket.AF_INET, *args, **kwargs)


def configure() -> None:
    if os.environ.get("LEGALRAG_IPV4") == "1":
        socket.getaddrinfo = _ipv4_only
