"""Process-wide patches the Megatron training stack needs before it runs.

Importing this module (once per process; later imports are no-ops) wraps
``deep_ep.Buffer.__init__`` so torch_memory_saver does not track the DeepEP
buffers.

This used to run in the package ``__init__``, so every process that touched a
light submodule (``megatron_config``, ``checkpoint_tracker``) imported
``deep_ep`` -- including RayWorkerManager, which unpickles placement group and
worker-spec objects whose modules import those submodules. The trainer actor and
the model provider, which build the models whose dispatchers allocate DeepEP
buffers, import this module instead.
"""

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
