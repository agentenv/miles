from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


def _layer_ids_from_moe_layer_freq(models) -> list[int]:
    """Global layer ids of the local MoE layers by ``config.moe_layer_freq``."""
    layer_indices = []
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if isinstance(config.moe_layer_freq, int):
                if layer_id % config.moe_layer_freq != 0:
                    continue
            elif isinstance(config.moe_layer_freq, list):
                assert len(config.moe_layer_freq) == config.num_layers
                if config.moe_layer_freq[layer_id] == 0:
                    continue
            layer_indices.append(layer_id)
    return layer_indices


def _layer_ids_from_routers(models) -> list[int] | None:
    """Global layer ids of the routers that registered a replay stream, in
    module (= construction = stream) order; None when a router lacks its
    layer number.

    A hybrid model (e.g. Nemotron-H: Mamba, attention and MoE layers from a
    ``hybrid_layer_pattern``) has MoE routers on only some layers, and its
    bridge config keeps ``moe_layer_freq = 1``, so the frequency rule would
    map every layer.  Each router knows its 1-based global ``layer_number``."""
    layer_indices = []
    for model in models:
        for module in model.modules():
            if getattr(module, "routing_replay", None) is None:
                continue
            layer_number = getattr(module, "layer_number", None)
            if not isinstance(layer_number, int) or layer_number < 1:
                return None
            layer_indices.append(layer_number - 1)
    return layer_indices


def register_replay_list_moe(replay_list, replay_data, *, models, **_kwargs):
    """Map replay streams to Megatron MoE layers using the local model layout.

    ``replay_data[:, layer]`` is indexed by the global layer id (SGLang
    records routed experts per absolute layer).  The frequency rule decides
    when it yields one layer per stream; otherwise (hybrid models) the
    routers' own layer numbers do.  A layer count that still differs from the
    stream count is an error, never a silent misalignment."""
    layer_indices = _layer_ids_from_moe_layer_freq(models)
    if len(layer_indices) != len(replay_list):
        from_routers = _layer_ids_from_routers(models)
        if from_routers is not None and len(from_routers) == len(replay_list):
            layer_indices = from_routers
    if len(layer_indices) != len(replay_list):
        raise RuntimeError(
            f"routing replay: {len(replay_list)} replay streams but {len(layer_indices)} local MoE layers "
            f"({layer_indices[:8]}...); the layer mapping does not fit this model"
        )
    if layer_indices and max(layer_indices) >= replay_data.shape[1]:
        raise RuntimeError(
            f"routing replay: MoE layer {max(layer_indices)} is outside the rollout's "
            f"{replay_data.shape[1]} recorded layers"
        )

    for replay_idx, layer_idx in enumerate(layer_indices):
        layer_data = replay_data[:, layer_idx]
        replay_list[replay_idx].record(layer_data)
