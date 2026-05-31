"""Pipecat TTS service using the megakernel TTS engine.

This module provides a custom Pipecat TTS service that uses the megakernel
for both the talker decoder and code predictor, achieving real-time streaming
speech synthesis with TTFC < 90ms and RTF < 0.3.

Usage in a Pipecat pipeline:
    tts = MegakernelTTSService(model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    pipeline = Pipeline([..., tts, ...])
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, AsyncIterator, Optional

import numpy as np

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from .tts_engine import MegakernelTTSEngine, TTSConfig

logger = logging.getLogger(__name__)


class MegakernelTTSService(TTSService):
    """Pipecat TTS service backed by the megakernel TTS engine.

    Streams audio chunks as they're generated — does NOT buffer the full
    utterance before sending.

    Performance targets:
        TTFC (time to first audio chunk): < 90 ms
        RTF (real-time factor): < 0.3
    """

    def __init__(
        self,
        *,
        model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        vocoder_path: Optional[str] = None,
        device: str = "cuda",
        chunk_frames: int = 10,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 2048,
        subtalker_do_sample: bool = True,
        subtalker_temperature: float = 0.9,
        subtalker_top_k: int = 50,
        warmup_profile: str = "full",
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        settings = kwargs.pop(
            "settings",
            TTSSettings(model=model_path, voice=None, language=None),
        )
        super().__init__(sample_rate=sample_rate or 24000, settings=settings, **kwargs)

        self._config = TTSConfig(
            model_path=model_path,
            vocoder_path=vocoder_path or model_path,
            chunk_frames=chunk_frames,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            subtalker_do_sample=subtalker_do_sample,
            subtalker_temperature=subtalker_temperature,
            subtalker_top_k=subtalker_top_k,
            warmup_profile=warmup_profile,
        )
        self._device = device
        self._engine: Optional[MegakernelTTSEngine] = None

    def can_generate_metrics(self) -> bool:
        return True

    def _ensure_engine(self):
        """Lazily initialize the TTS engine."""
        if self._engine is None:
            init_started = time.perf_counter()
            logger.info(
                "Initializing Megakernel TTS engine model=%s device=%s chunk_frames=%s",
                self._config.model_path,
                self._device,
                self._config.chunk_frames,
            )
            self._engine = MegakernelTTSEngine(config=self._config, device=self._device)
            self._engine.initialize()
            logger.info(
                "Megakernel TTS engine initialized duration_ms=%.1f sample_rate=%s",
                (time.perf_counter() - init_started) * 1000,
                self._engine.sample_rate,
            )

    async def warmup(self):
        """Initialize and warm the underlying TTS engine before first speech."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_engine)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate streaming speech from text using the megakernel.

        Yields audio frames as codec frames are generated and decoded,
        pushing audio to the pipeline chunk by chunk.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        utterance_started = time.perf_counter()
        first_chunk_at: Optional[float] = None
        chunk_count = 0
        audio_bytes = 0
        sample_rate = self._engine.sample_rate if self._engine else 24000
        text_chars = len(text)

        try:
            logger.info(
                "TTS utterance started context_id=%s chars=%s",
                context_id,
                text_chars,
            )
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            # Run streaming synthesis in a thread (megakernel is synchronous/GPU-bound)
            async def audio_chunk_iterator() -> AsyncIterator[bytes]:
                """Generate audio chunks as PCM16 bytes via streaming synthesis."""
                nonlocal audio_bytes, chunk_count, first_chunk_at, sample_rate
                loop = asyncio.get_running_loop()

                # Ensure engine is initialized
                await loop.run_in_executor(None, self._ensure_engine)
                engine = self._engine
                sample_rate = engine.sample_rate

                # Run the streaming synthesis
                previous_chunk_at = time.perf_counter()
                async for audio_chunk, sr in engine.synthesize_streaming(
                    text, chunk_frames=self._config.chunk_frames
                ):
                    # Convert float32 numpy array to PCM16 bytes
                    chunk_ready_at = time.perf_counter()
                    pcm16 = _float32_to_pcm16(audio_chunk)
                    if first_chunk_at is None:
                        first_chunk_at = chunk_ready_at
                        logger.info(
                            "TTS first chunk context_id=%s ttfc_ms=%.1f sample_rate=%s",
                            context_id,
                            (first_chunk_at - utterance_started) * 1000,
                            sr,
                        )
                    chunk_count += 1
                    audio_bytes += len(pcm16)
                    sample_rate = sr
                    chunk_audio_ms = len(audio_chunk) / sr * 1000 if sr else 0.0
                    chunk_gap_ms = (chunk_ready_at - previous_chunk_at) * 1000
                    logger.info(
                        "TTS chunk metrics context_id=%s chunk=%s samples=%s audio_ms=%.1f "
                        "wall_gap_ms=%.1f bytes=%s total_bytes=%s sample_rate=%s",
                        context_id,
                        chunk_count,
                        len(audio_chunk),
                        chunk_audio_ms,
                        chunk_gap_ms,
                        len(pcm16),
                        audio_bytes,
                        sr,
                    )
                    previous_chunk_at = chunk_ready_at
                    yield pcm16

            async for frame in self._stream_audio_frames_from_iterator(
                audio_chunk_iterator(),
                in_sample_rate=self._engine.sample_rate if self._engine else 24000,
                context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame

        except Exception as e:
            logger.error(f"{self} TTS exception: {e}")
            yield ErrorFrame(error=f"Megakernel TTS error: {e}")
        finally:
            elapsed_s = time.perf_counter() - utterance_started
            audio_duration_s = audio_bytes / 2 / sample_rate if sample_rate else 0.0
            rtf = elapsed_s / audio_duration_s if audio_duration_s else 0.0
            logger.info(
                "TTS metrics context_id=%s chars=%s chunks=%s ttfc_ms=%s "
                "audio_ms=%.1f duration_ms=%.1f rtf=%.3f bytes=%s",
                context_id,
                text_chars,
                chunk_count,
                f"{(first_chunk_at - utterance_started) * 1000:.1f}"
                if first_chunk_at
                else "n/a",
                audio_duration_s * 1000,
                elapsed_s * 1000,
                rtf,
                audio_bytes,
            )
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()


def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 audio [-1, 1] to 16-bit PCM bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()
