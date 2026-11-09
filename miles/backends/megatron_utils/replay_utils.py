from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles_plugins.models.kimi_k3.partial_experts import map_replay_ids_to_slots
from miles_plugins.models.kimi_k3.permute import get_active_slot_layout


def register_replay_list_moe(replay_list, replay_data, *, models, **_kwargs):
    """Map replay streams to Megatron MoE layers using the local model layout."""
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

    # inference engines report original expert ids; a partial-experts trainer routes in slot ids
    layout = get_active_slot_layout()
    for replay_idx, layer_idx in enumerate(layer_indices):
        layer_data = map_replay_ids_to_slots(layout, layer_idx, replay_data[:, layer_idx])
        replay_list[replay_idx].record(layer_data)
