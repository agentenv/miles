from pydantic import Field

from miles.utils.args.enhanced_argparse_namespace import SerializableConfigDict
from miles.utils.args.schema import BaseConfig


class YetoExtensionConfig(BaseConfig):
    """Opaque context of an out-of-tree orchestrator that drives Miles as a library (yeto).

    Miles never reads it and declares no CLI flag for it. The orchestrator sets it in ``preprocess_args``
    (or with ``AllConfig.model_copy(update=...)`` before any component config is sliced). Every leaf config
    carries it, so trainer ranks, rollout workers and custom hooks see the value the orchestrator froze.
    Values must be JSON-serializable (``SerializableConfigDict``) for the RPC and payload transports.
    """

    yeto_ext: SerializableConfigDict = Field(default_factory=dict)
