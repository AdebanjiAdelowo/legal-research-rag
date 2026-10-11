"""Print the account's weekly GPU quota: allowed, used and remaining.

    /opt/anaconda3/bin/python kaggle/quota.py

Check this before submitting a run. A run that outlasts the remaining quota is cut
off, and an interrupted run may not keep its output.
"""

from __future__ import annotations

import socket

_resolve = socket.getaddrinfo


def seconds(value: str) -> float:
    """Kaggle returns durations such as ``'6933.122060.0s'``; keep the leading number."""
    head = value.rstrip("s").split(".")
    return float(".".join(head[:2])) if len(head) > 1 else float(head[0])


def main() -> None:
    socket.getaddrinfo = lambda h, p, f=0, *a, **k: _resolve(h, p, socket.AF_INET, *a, **k)
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types import kernels_api_service as t

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        stats = client.kernels.kernels_api_client.get_accelerator_quota_statistics(
            t.ApiGetAcceleratorQuotaStatisticsRequest()
        ).to_dict()
    gpu = stats["gpuQuota"]
    allowed, used = seconds(gpu["totalTimeAllowed"]), seconds(gpu["timeUsed"])
    remaining = (allowed - used) / 3600
    print(
        f"GPU quota: {allowed / 3600:.1f} h allowed, {used / 3600:.2f} h used, "
        f"{remaining:.2f} h remaining; refreshes {stats['quotaRefreshTime']}"
    )


if __name__ == "__main__":
    main()
