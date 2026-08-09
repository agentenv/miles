import logging

import torch

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    import miles_plugins.megatron_bridge  # noqa: F401
except Exception as _e:  # best-effort; not every environment uses megatron.bridge
    logging.warning("miles megatron.bridge plugins failed to load: %s", _e)

# Yeto ships the pinned DeepSeek-V4 registration separately because the
# NVIDIA Megatron-Bridge version in this image predates V4, while Miles already
# contains the V4 MCore implementation.  Ray actor processes import this module
# independently of the controller, so registration must happen here as well as
# in Yeto's learner entry point.  Ordinary Miles installations without Yeto
# remain unaffected.
try:
    from yeto.rl.deepseek_v4_bridge import ensure_deepseek_v4_bridge
except ModuleNotFoundError as _e:
    if (_e.name or "").split(".", 1)[0] != "yeto":
        raise
else:
    ensure_deepseek_v4_bridge()

logging.getLogger("megatron").setLevel(logging.WARNING)
