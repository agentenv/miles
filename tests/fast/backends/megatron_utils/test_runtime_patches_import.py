from tests.fast.import_isolation_utils import modules_imported_by

# RayWorkerManager unpickles PlacementGroupInfo and worker specs, whose modules import the
# light megatron_utils submodules; the package __init__ must not pull in the DeepEP patch.
_UNPICKLED_BY_LIGHT_ACTORS = ("miles.backends.megatron_utils.megatron_config", "miles.ray.specs.train")


class TestRuntimePatchesStayOutOfLightActors:
    def test_the_light_megatron_utils_modules_do_not_import_deep_ep(self):
        """Importing deep_ep costs every light actor its CUDA extension load before it can serve."""
        for module in _UNPICKLED_BY_LIGHT_ACTORS:
            modules = modules_imported_by(module)

            assert "miles.backends.megatron_utils.runtime_patches" not in modules
            assert "deep_ep" not in modules
