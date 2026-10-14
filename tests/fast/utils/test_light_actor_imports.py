"""Light Ray actors must not import the Megatron training stack.

Each actor process pays its imports at start-up; on a training image the
Megatron stack (``megatron.core``, ``megatron.bridge``, ``modelopt``, DeepEP)
costs tens of seconds. The light actors only need the CPU-side utilities and the
light configuration modules of ``megatron_utils`` (``megatron_config``,
``checkpoint_request``), so they must stay free of the rest.
"""

import json
import subprocess
import sys

LIGHT_ACTOR_MODULES = (
    "miles.utils.workers.command_actor",
    "miles.utils.workers.ray_worker_manager",
    "miles.ray.rollout.inference_controller",
    "miles.utils.http_utils",
)
FORBIDDEN_PREFIXES = (
    "megatron.core",
    "megatron.bridge",
    "modelopt",
    "deep_ep",
    "miles_plugins.megatron_bridge",
    "miles.backends.megatron_utils.runtime_patches",
    "miles.backends.megatron_utils.actor",
)


def test_light_actor_modules_do_not_import_the_training_stack():
    """Each light actor would pay the training stack's import time before it can serve."""
    # A fresh interpreter: other tests in this process may already have
    # imported the heavy modules.
    code = (
        "import importlib, json, sys\n"
        f"for name in {LIGHT_ACTOR_MODULES!r}:\n"
        "    importlib.import_module(name)\n"
        f"print(json.dumps(sorted(m for m in sys.modules if m.startswith({FORBIDDEN_PREFIXES!r}))))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert json.loads(out.strip().splitlines()[-1]) == []
