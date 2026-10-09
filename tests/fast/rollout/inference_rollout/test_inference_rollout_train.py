import asyncio
import logging
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.rollout.inference_rollout import inference_rollout_train


class TestAbort:
    async def test_one_unresponsive_worker_does_not_stop_abort_cleanup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One failed worker abort still runs the agent hook and returns pending partial groups."""
        args = Namespace(
            partial_rollout=True,
            sglang_router_ip="router",
            sglang_router_port=30000,
            use_miles_router=True,
        )
        state = SimpleNamespace(args=args, aborted=False)
        sample = SimpleNamespace(response="partial", metadata={})
        hook_calls: list[Namespace] = []
        posted_urls: list[str] = []
        hook_finished = asyncio.Event()

        async def fake_get(url: str) -> dict[str, list[str]]:
            return {"urls": ["http://healthy", "http://unresponsive"]}

        async def fake_post(url: str, payload: dict[str, bool]) -> None:
            posted_urls.append(url)
            if "unresponsive" in url:
                raise ConnectionError("worker cannot answer")

        async def fake_agent_abort_hook(hook_args: Namespace) -> None:
            hook_calls.append(hook_args)
            hook_finished.set()

        async def finish_group() -> list[SimpleNamespace]:
            await hook_finished.wait()
            return [sample]

        pending = asyncio.create_task(finish_group())
        await asyncio.sleep(0)
        monkeypatch.setattr(inference_rollout_train, "get", fake_get)
        monkeypatch.setattr(inference_rollout_train, "post", fake_post)
        monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", fake_agent_abort_hook)

        with caplog.at_level(logging.WARNING, logger=inference_rollout_train.__name__):
            aborted_groups = await inference_rollout_train.abort(state, {pending}, rollout_id=17)

        assert posted_urls == ["http://healthy/abort_request", "http://unresponsive/abort_request"]
        assert hook_calls == [args]
        assert aborted_groups == [[sample]]
        assert sample.metadata["start_rollout_id"] == 17
        assert "Failed to abort worker at http://unresponsive: worker cannot answer" in caplog.text

    async def test_discarded_groups_are_tallied_without_partial_rollout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without partial rollout the drained groups are counted on
        args.rollout_abort_discard_stats; a drained group that raised counts as
        unknown and does not fail the abort."""
        args = Namespace(
            partial_rollout=False,
            sglang_router_ip="router",
            sglang_router_port=30000,
            use_miles_router=True,
        )
        state = SimpleNamespace(args=args, aborted=False)

        async def fake_get(url: str) -> dict[str, list[str]]:
            return {"urls": ["http://w"]}

        async def fake_post(url: str, payload: dict[str, bool]) -> None:
            return None

        async def fake_hook(hook_args: Namespace) -> None:
            return None

        async def group(lengths: list[int]) -> list[SimpleNamespace]:
            return [SimpleNamespace(response="x", response_length=n, metadata={}) for n in lengths]

        async def broken() -> list[SimpleNamespace]:
            raise RuntimeError("sandbox lease is not live")

        pendings = {asyncio.create_task(group([3, 4])), asyncio.create_task(group([10])),
                    asyncio.create_task(broken())}
        monkeypatch.setattr(inference_rollout_train, "get", fake_get)
        monkeypatch.setattr(inference_rollout_train, "post", fake_post)
        monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", fake_hook)

        assert await inference_rollout_train.abort(state, pendings, rollout_id=3) == []
        assert args.rollout_abort_discard_stats == {
            "groups": 3, "samples": 3, "response_tokens": 17, "unknown_groups": 1}
