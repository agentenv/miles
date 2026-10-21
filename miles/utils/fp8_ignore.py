"""FP8 ignore list: the modules SGLang keeps in BF16 under an FP8 ``quantization_config``.

The trainer-side quantizer (``quantize_params_fp8``) and the offline converter
(``tools/convert_hf_to_fp8.py``) both follow it, so the weights sent to SGLang
have the precision of the layers SGLang built from the same config.
"""


def ignored_modules(quantization_config) -> tuple[str, ...]:
    """The modules SGLang keeps unquantized, as its Fp8Config reads them.

    SGLang takes ``ignored_layers``, else ``modules_to_not_convert``, and keeps
    each name with and without the ``model.`` prefix.
    """
    names = quantization_config.get("ignored_layers")
    if names is None:
        names = quantization_config.get("modules_to_not_convert")
    bases = {name.removeprefix("model.") for name in names or ()}
    return tuple(sorted(bases | {f"model.{base}" for base in bases}))


def is_ignored(name, ignored) -> bool:
    """Whether the module of HF parameter ``name`` is ignored (SGLang ``_module_path_match``)."""
    module = name.removesuffix(".weight")
    return any(
        module == entry or module.startswith(entry + ".") or f".{entry}." in f".{module}."
        for entry in (e.rstrip(".") for e in ignored)
    )
