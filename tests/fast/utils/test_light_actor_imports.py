"""Light Ray actors must not import the Megatron training stack.

Each actor process pays its imports at start-up; on the ports image
``miles.backends.megatron_utils`` costs 28-31 s and ``modelopt`` 21-30 s
(coldstart-gpu-review §6.2).  The light actors only need the CPU-side
utilities, so their modules must stay free of both.
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
FORBIDDEN_PREFIXES = ("miles.backends.megatron_utils", "megatron.bridge", "modelopt")


def test_light_actor_modules_do_not_import_the_training_stack():
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


def test_old_train_step_types_path_still_works():
    from miles.backends.megatron_utils.ft import types as old
    from miles.utils import train_step_types as new

    assert old.TrainStepOutcome is new.TrainStepOutcome
    assert old.TrainStepOutput is new.TrainStepOutput
