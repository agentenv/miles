import importlib.util
from pathlib import Path

import pytest

# Load the module file directly: the package __init__ imports Megatron, which CPU CI does not install.
_PATH = Path(__file__).resolve().parents[5] / "miles_plugins" / "models" / "kimi_k31" / "dsa_layout.py"
_SPEC = importlib.util.spec_from_file_location("_dsa_layout_under_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
checked_dsa_layers = _MODULE.checked_dsa_layers

# Kimi-K3.1-Pro: KDA on every layer but each fourth and the last
KDA = tuple(n for n in range(1, 94) if n % 4 != 0 and n != 93)
FULL_DSA = [n for n in range(1, 94) if n not in KDA]


def test_the_full_model_table():
    assert checked_dsa_layers(FULL_DSA, num_layers=93, kda_layers=KDA) == FULL_DSA


@pytest.mark.parametrize("table", [FULL_DSA, [4]])
def test_a_four_layer_cut_keeps_or_trims_the_table(table):
    assert checked_dsa_layers(table, num_layers=4, kda_layers=KDA) == [4]


def test_a_table_that_misses_an_mla_layer_is_refused():
    with pytest.raises(AssertionError, match="MLA layers"):
        checked_dsa_layers([4], num_layers=8, kda_layers=KDA)
    with pytest.raises(AssertionError):
        checked_dsa_layers([3, 4], num_layers=4, kda_layers=KDA)
