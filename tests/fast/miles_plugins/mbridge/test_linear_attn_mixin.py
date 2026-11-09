import importlib.util
from pathlib import Path

# Load the module file directly: the package __init__ imports mbridge, which CPU CI does not install.
_PATH = Path(__file__).resolve().parents[4] / "miles_plugins" / "mbridge" / "linear_attn.py"
_SPEC = importlib.util.spec_from_file_location("_linear_attn_under_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
LinearAttentionBridgeMixin = _MODULE.LinearAttentionBridgeMixin


class _Base:
    def __init__(self):
        self.seen_kwargs = None

    def _weight_name_mapping_mcore_local_to_global(self, model, consider_ep: bool = True):
        self.seen_kwargs = {"consider_ep": consider_ep}
        return {"a.weight": "a.weight", "b.weight_layout_version": "b.weight_layout_version"}


class _Bridge(LinearAttentionBridgeMixin, _Base):
    pass


def test_mixin_forwards_keyword_options_to_the_base_bridge():
    bridge = _Bridge()
    mapping = bridge._weight_name_mapping_mcore_local_to_global(object(), consider_ep=False)
    assert bridge.seen_kwargs == {"consider_ep": False}
    assert mapping == {"a.weight": "a.weight"}


def test_mixin_keeps_the_base_default_when_no_option_is_passed():
    bridge = _Bridge()
    bridge._weight_name_mapping_mcore_local_to_global(object())
    assert bridge.seen_kwargs == {"consider_ep": True}
