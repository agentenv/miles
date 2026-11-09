import pytest
import requests

from tests.fast.router.test_router import RouterEnv, make_router_config, router_env  # noqa: F401

from miles.router.router import MilesRouter
from miles.utils.http_utils import find_available_port


@pytest.fixture
def router() -> MilesRouter:
    return MilesRouter(make_router_config(find_available_port(20000)), verbose=False)


class TestCordonSelection:
    def test_a_cordoned_worker_is_never_selected_but_keeps_its_count(self, router):
        router.worker_request_counts = {"http://w1:8000": 0, "http://w2:8000": 5}
        router.cordoned_workers = {"http://w1:8000"}

        assert router._use_url() == "http://w2:8000"
        assert router.worker_request_counts == {"http://w1:8000": 0, "http://w2:8000": 6}

    def test_in_flight_requests_of_a_cordoned_worker_still_finish(self, router):
        router.worker_request_counts = {"http://w1:8000": 0, "http://w2:8000": 0}
        url = router._use_url()
        router.cordoned_workers.add(url)

        assert router.worker_inflight() == {url: 1, next(u for u in router.worker_request_counts if u != url): 0}
        router._finish_url(url)
        assert router.worker_request_counts[url] == 0

    def test_all_workers_cordoned_or_dead_raises(self, router):
        router.worker_request_counts = {"http://w1:8000": 0, "http://w2:8000": 0}
        router.cordoned_workers = {"http://w1:8000"}
        router.dead_workers = {"http://w2:8000"}

        with pytest.raises(RuntimeError, match="No healthy workers"):
            router._use_url()

    def test_uncordon_puts_the_worker_back(self, router):
        router.worker_request_counts = {"http://w1:8000": 0, "http://w2:8000": 3}
        router.cordoned_workers = {"http://w1:8000"}
        router.cordoned_workers.discard("http://w1:8000")

        assert router._use_url() == "http://w1:8000"


class TestCordonEndpoints:
    def test_cordon_uncordon_and_inflight_over_http(self, router_env: RouterEnv):  # noqa: F811
        url = "http://127.0.0.1:30021"
        other = "http://127.0.0.1:30022"
        for u in (url, other):
            requests.post(f"{router_env.url}/add_worker", params={"url": u}, timeout=5.0).raise_for_status()
        router_env.router.worker_request_counts[url] = 2

        r = requests.post(f"{router_env.url}/cordon_worker", params={"url": url}, timeout=5.0)
        r.raise_for_status()
        assert url in router_env.router.cordoned_workers
        assert url in router_env.router.worker_request_counts

        r = requests.get(f"{router_env.url}/worker_inflight", timeout=5.0)
        r.raise_for_status()
        assert r.json() == {"inflight": {url: 2, other: 0}, "cordoned": [url]}

        requests.post(f"{router_env.url}/uncordon_worker", json={"url": url}, timeout=5.0).raise_for_status()
        assert url not in router_env.router.cordoned_workers

    def test_cordoning_an_unknown_worker_is_a_404(self, router_env: RouterEnv):  # noqa: F811
        r = requests.post(f"{router_env.url}/cordon_worker", params={"url": "http://127.0.0.1:1"}, timeout=5.0)
        assert r.status_code == 404
        assert not router_env.router.cordoned_workers

    def test_removing_a_worker_clears_its_cordon(self, router_env: RouterEnv):  # noqa: F811
        url = "http://127.0.0.1:30023"
        requests.post(f"{router_env.url}/add_worker", params={"url": url}, timeout=5.0).raise_for_status()
        requests.post(f"{router_env.url}/cordon_worker", params={"url": url}, timeout=5.0).raise_for_status()

        requests.post(f"{router_env.url}/remove_worker", params={"url": url}, timeout=5.0).raise_for_status()

        assert url not in router_env.router.cordoned_workers


class TestRouterApiClientCordon:
    async def test_the_client_drives_cordon_and_reads_inflight(self, router_env: RouterEnv):  # noqa: F811
        from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient

        url = "http://127.0.0.1:30031"
        requests.post(f"{router_env.url}/add_worker", params={"url": url}, timeout=5.0).raise_for_status()
        router_env.router.worker_request_counts[url] = 4
        client = SGLangRouterApiClient(router_url=router_env.url)

        await client.cordon_worker(worker_url=url)
        assert router_env.router.cordoned_workers == {url}
        assert await client.get_worker_inflight() == {url: 4}

        await client.uncordon_worker(worker_url=url)
        assert router_env.router.cordoned_workers == set()


class TestRegistrationGenerations:
    def test_a_request_of_a_previous_registration_does_not_decrement_the_new_one(
        self, router_env: RouterEnv  # noqa: F811
    ):
        """remove + re-add of the same url: the old in-flight request must not take the new registration's slot."""
        router = router_env.router
        url = "http://127.0.0.1:30041"
        requests.post(f"{router_env.url}/add_worker", params={"url": url}, timeout=5.0).raise_for_status()
        old_generation = router.worker_generations[url]
        assert router._use_url() == url

        requests.post(f"{router_env.url}/remove_worker", params={"url": url}, timeout=5.0).raise_for_status()
        requests.post(f"{router_env.url}/add_worker", params={"url": url}, timeout=5.0).raise_for_status()
        assert router._use_url() == url
        assert router.worker_request_counts[url] == 1

        router._finish_url(url, generation=old_generation)
        assert router.worker_request_counts[url] == 1

        router._finish_url(url, generation=router.worker_generations[url])
        assert router.worker_request_counts[url] == 0


class TestCordonedRegistration:
    def test_add_worker_can_register_a_worker_cordoned(self, router_env: RouterEnv):  # noqa: F811
        router = router_env.router
        url, other = "http://127.0.0.1:30051", "http://127.0.0.1:30052"
        requests.post(f"{router_env.url}/add_worker", params={"url": other}, timeout=5.0).raise_for_status()
        requests.post(
            f"{router_env.url}/add_worker", params={"url": url, "cordoned": "1"}, timeout=5.0
        ).raise_for_status()

        assert url in router.worker_request_counts and router.cordoned_workers == {url}
        assert {router._use_url() for _ in range(3)} == {other}

        requests.post(f"{router_env.url}/uncordon_worker", params={"url": url}, timeout=5.0).raise_for_status()
        assert router._use_url() == url

    async def test_the_client_passes_cordoned_on_the_legacy_api_only(self, router_env: RouterEnv):  # noqa: F811
        from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient

        client = SGLangRouterApiClient(router_url=router_env.url)
        url = "http://127.0.0.1:30053"
        await client.add_worker(worker_url=url, worker_type="regular", use_legacy_api=True, cordoned=True)
        assert router_env.router.cordoned_workers == {url}
        with pytest.raises(ValueError, match="Miles router"):
            await client.add_worker(worker_url=url, worker_type="regular", use_legacy_api=False, cordoned=True)


class TestRejoinInFlightSemantics:
    def test_a_drain_of_a_rejoined_url_does_not_count_the_old_registration(self, router_env: RouterEnv):  # noqa: F811
        """Documented semantics: in-flight counts (and drains) cover the current registration only."""
        router = router_env.router
        url = "http://127.0.0.1:30054"
        requests.post(f"{router_env.url}/add_worker", params={"url": url}, timeout=5.0).raise_for_status()
        old_generation = router.worker_generations[url]
        router._use_url()  # one request in flight on the first registration

        requests.post(f"{router_env.url}/remove_worker", params={"url": url}, timeout=5.0).raise_for_status()
        requests.post(f"{router_env.url}/add_worker", params={"url": url}, timeout=5.0).raise_for_status()

        assert router.worker_inflight()[url] == 0  # the old request is invisible to a drain of the new registration
        router._finish_url(url, generation=old_generation)
        assert router.worker_inflight()[url] == 0
