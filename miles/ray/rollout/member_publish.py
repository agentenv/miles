"""Member weight publish: update only a set of engine cells, then commit the version once all serving cells agree.

``placement_group.update_weights`` publishes to every cell and sets the rollout executor's weight version. A member
publish (``update_member_weights``) loads the weights into the named cells only; the other cells keep serving what
they serve, so it leaves the executor alone. Once the caller brought every serving cell to one version,
``commit_weight_version`` checks that under the controller lock and sets the executor's version.
"""

from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.ray.train_actor import WeightUpdateOutput
from miles.utils.workers.worker_handle import BaseWorkerHandle


class MemberPublishFailedError(RuntimeError):
    """Some members did not take the weights; ``end_update_weights`` took them out of service."""

    def __init__(self, failed_cell_ids: tuple[str, ...], weight_version: int | None) -> None:
        self.failed_cell_ids, self.weight_version = failed_cell_ids, weight_version
        super().__init__(f"member cells {list(failed_cell_ids)} failed their weight update and left service")


async def update_member_weights(
    actor_model,
    inference_controller: BaseWorkerHandle,
    *,
    members: list[str],
    expected_epoch: int,
    admit_cordoned: bool = False,
    rollout_id: int | None = None,
    trainer_model_id: str | None = None,
) -> int | None:
    """Publish the actor weights to ``members`` (cell ids) only and return the weight version they now hold.

    ``expected_epoch`` is the membership epoch the member set was planned for. With ``admit_cordoned`` the cells
    become Serving cordoned; the caller verifies them and opens them with ``admit_cells``.
    """
    info: UpdatableEngines = await inference_controller.start_update_weights(
        model_id=trainer_model_id, members=members, expected_epoch=expected_epoch, admit_cordoned=admit_cordoned
    )
    try:
        output: WeightUpdateOutput = await actor_model.update_weights(info=info, rollout_id=rollout_id)
    except BaseException:
        await inference_controller.abort_update_weights()
        raise
    await inference_controller.end_update_weights(
        snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes, failed_cell_ids=output.failed_cell_ids
    )
    if output.failed_cell_ids:
        raise MemberPublishFailedError(tuple(output.failed_cell_ids), output.weight_version)
    return output.weight_version


async def commit_weight_version(
    rollout_executor: BaseWorkerHandle,
    inference_controller: BaseWorkerHandle,
    *,
    weight_version: int,
    expected_epoch: int,
    trainer_model_id: str | None = None,
) -> None:
    """Set the executor's weight version once every Serving cell reports it, at ``expected_epoch``."""
    await inference_controller.start_commit_weight_version(
        weight_version=weight_version, expected_epoch=expected_epoch, model_id=trainer_model_id
    )
    try:
        await rollout_executor.set_weight_version(weight_version, trainer_model_id=trainer_model_id)
    finally:
        await inference_controller.end_commit_weight_version()
