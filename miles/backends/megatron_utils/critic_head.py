"""Where a critic's value head goes (no megatron imports; unit-testable on CPU).

Bridge-built multimodal wrappers (e.g. ``Qwen3VLModel`` for Qwen3.5) keep the
LM head on ``model.language_model.output_layer``; the wrapper's forward never
reads a top-level ``output_layer``.  Assigning the value head on the wrapper
leaves the vocab head live, so the critic predicts logits, and the HF load maps
``language_model.output_layer.weight -> lm_head.weight`` -- which a tied
checkpoint (``tie_word_embeddings=true``) does not contain because the critic
provider disables ``share_embeddings_and_output_weights``.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch


def critic_head_owner(model: torch.nn.Module) -> torch.nn.Module:
    """Module whose ``output_layer`` the value head must replace."""
    inner = getattr(model, "language_model", None)
    if isinstance(inner, torch.nn.Module) and hasattr(inner, "output_layer"):
        return inner
    return model


def iter_value_heads(chunk: torch.nn.Module, head_cls: type) -> Iterator[tuple[torch.nn.Module, str, torch.nn.Module]]:
    """(parent, attr, head) for each ``head_cls`` child of ``chunk`` or of its ``language_model``."""
    owners = [chunk]
    owner = critic_head_owner(chunk)
    if owner is not chunk:
        owners.append(owner)
    for parent in owners:
        for name, head in list(parent.named_children()):
            if isinstance(head, head_cls):
                yield parent, name, head
