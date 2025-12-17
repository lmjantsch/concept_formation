from .global_primitives import CompatibilityMapping
from .caching.caching_fn import Qwen3CachingFN

HF_ID_CACHING_FN_MAPPING = {
    "Qwen/Qwen3-4B": CompatibilityMapping(
        default = Qwen3CachingFN,
        compatible= [Qwen3CachingFN]
    )
}

DEFAULT_MODEL_KWARGS = {
    "trust_remote_code": True,
    "attn_implementation": 'eager',
    "dispatch": True
}
