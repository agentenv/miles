"""Which Kimi-K3.1 layers carry a DSA indexer (no torch / Megatron imports, so CPU tests load it)."""

from collections.abc import Iterable


def checked_dsa_layers(dsa_layers: Iterable[int], *, num_layers: int, kda_layers: Iterable[int]) -> list[int]:
    """The 1-based DSA layers of a model with ``num_layers`` layers: every MLA (non-KDA) layer up to
    ``num_layers``, nothing else. A checkpoint cut down to its first layers may keep the full model's
    ``dsa_layers`` table or trim it; both describe the same layers once cut at ``num_layers``."""
    kda = set(kda_layers)
    expected = [n for n in range(1, num_layers + 1) if n not in kda]
    actual = sorted(n for n in dsa_layers if n <= num_layers)
    assert actual == expected, f"dsa_layers {actual} != the MLA layers {expected} of a {num_layers}-layer model"
    return actual
