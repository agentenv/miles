import dataclasses
import logging
from urllib.parse import quote

import httpx
import sglang_router
from packaging.version import parse

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.utils.http_utils import GeneralHttpClientProvider

logger = logging.getLogger(__name__)

ROUTER_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


@dataclasses.dataclass(frozen=True)
class SGLangRouterApiClient:
    router_url: str

    async def add_worker(
        self,
        worker_url: str,
        worker_type: WorkerType,
        bootstrap_port: int | None = None,
        *,
        cordoned: bool = False,
    ):
        """Register a worker; ``cordoned`` (miles router only) registers it without making it selectable."""
        payload = {
            "url": worker_url,
            "worker_type": worker_type.value,
        }
        if worker_type == WorkerType.PREFILL:
            payload["bootstrap_port"] = bootstrap_port
        if cordoned:
            payload["cordoned"] = True
        response = await GeneralHttpClientProvider.client().post(
            f"{self.router_url}/workers",
            json=payload,
            timeout=ROUTER_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

    # Cordon and in-flight counts are miles router endpoints (see miles/router/router.py); sgl-router has neither.
    async def cordon_worker(self, worker_url: str) -> None:
        response = await GeneralHttpClientProvider.client().post(
            f"{self.router_url}/cordon_worker", json={"url": worker_url}, timeout=ROUTER_REQUEST_TIMEOUT
        )
        response.raise_for_status()

    async def uncordon_worker(self, worker_url: str) -> None:
        response = await GeneralHttpClientProvider.client().post(
            f"{self.router_url}/uncordon_worker", json={"url": worker_url}, timeout=ROUTER_REQUEST_TIMEOUT
        )
        response.raise_for_status()

    async def get_worker_inflight(self) -> dict[str, int]:
        response = await GeneralHttpClientProvider.client().get(
            f"{self.router_url}/worker_inflight", timeout=ROUTER_REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()["inflight"]

    async def remove_worker(self, worker_url: str) -> None:
        response = None
        if parse(sglang_router.__version__) < parse("0.3.0"):
            quoted_worker_url = quote(worker_url, safe="")
            response = await GeneralHttpClientProvider.client().delete(
                f"{self.router_url}/workers/{quoted_worker_url}",
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        else:
            workers_response = await GeneralHttpClientProvider.client().get(
                f"{self.router_url}/workers",
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
            workers_response.raise_for_status()
            all_workers = workers_response.json()["workers"]
            for worker in all_workers:
                if worker["url"] == worker_url:
                    worker_id = worker["id"]
                    response = await GeneralHttpClientProvider.client().delete(
                        f"{self.router_url}/workers/{worker_id}",
                        timeout=ROUTER_REQUEST_TIMEOUT,
                    )
                    break
            else:
                logger.warning(f"Worker {worker_url} not found in router during shutdown.")

        if response is not None:
            response.raise_for_status()
