"""Qwen Megakernel decode helpers for text and Qwen3-TTS.

CUDA extension builds are deferred until the corresponding decoder module is
imported. This keeps the text and TTS build variants from compiling on ordinary
package import.
"""

__all__ = [
    "load_weights",
    "Decoder",
    "generate",
    "MegakernelTTSEngine",
    "TTSConfig",
    "MegakernelTTSService",
]


def __getattr__(name):
    if name in {"load_weights", "Decoder", "generate"}:
        from qwen_megakernel import model

        return getattr(model, name)
    if name in {"MegakernelTTSEngine", "TTSConfig"}:
        from qwen_megakernel import tts_engine

        return getattr(tts_engine, name)
    if name == "MegakernelTTSService":
        from qwen_megakernel import pipecat_tts

        return pipecat_tts.MegakernelTTSService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
