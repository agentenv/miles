# Keep this package import light: light actors (RayWorkerManager) load
# ``megatron_config`` / ``checkpoint_tracker`` through it.  The heavy
# process-wide patches live in ``runtime_patches``.
import logging

logging.getLogger("megatron").setLevel(logging.WARNING)
