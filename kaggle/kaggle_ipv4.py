"""Run the Kaggle CLI with name resolution restricted to IPv4.

    python kaggle/kaggle_ipv4.py kernels push -p <folder> --accelerator NvidiaTeslaT4
    python kaggle/kaggle_ipv4.py datasets version -p <folder> -m "<message>"

On a network that advertises IPv6 routes it cannot carry, the CLI's uploads block
indefinitely while opening a connection to the storage host. This wrapper is the CLI
unchanged apart from that restriction, with a watchdog that prints a traceback and
exits if a call makes no progress for ten minutes.
"""

from __future__ import annotations

import faulthandler
import socket
import sys

_resolve = socket.getaddrinfo


def _ipv4_only(host, port, family=0, *args, **kwargs):
    return _resolve(host, port, socket.AF_INET, *args, **kwargs)


def main() -> None:
    socket.getaddrinfo = _ipv4_only
    faulthandler.dump_traceback_later(600, exit=True)
    from kaggle.cli import main as kaggle_main

    sys.argv = ["kaggle", *sys.argv[1:]]
    kaggle_main()


if __name__ == "__main__":
    main()
