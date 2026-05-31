#!/usr/bin/env python3
"""Pipecat WebRTC voice bot: browser mic -> Deepgram -> OpenAI -> Megakernel TTS."""

import asyncio
import logging
import os
import time
from pathlib import Path

from dotenv import dotenv_values

logger = logging.getLogger(__name__)


SYSTEM_INSTRUCTION = (
    "You are a concise, natural voice assistant. Keep spoken answers short and "
    "conversational. Do not use bullets unless the user asks for a list. Avoid "
    "emojis, decorative symbols, and text that is awkward to speak aloud. Only "
    "mention the custom local TTS system if the user asks about your voice or "
    "implementation."
)


def load_env_files() -> None:
    """Load .env and .env.local without overriding variables already exported."""
    repo_root = Path(__file__).resolve().parent
    values = {
        **dotenv_values(repo_root / ".env"),
        **dotenv_values(repo_root / ".env.local"),
    }
    for key, value in values.items():
        if value is not None and key not in os.environ:
            os.environ[key] = value


def require_env(*names: str) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variable(s): {joined}. "
            "Set them in the shell, .env, or .env.local."
        )


def create_tts_service():
    from qwen_megakernel.pipecat_tts import MegakernelTTSService

    model_path = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    device = os.getenv("QWEN_TTS_DEVICE", "cuda")
    chunk_frames = int(os.getenv("QWEN_TTS_CHUNK_FRAMES", "10"))
    logger.info(
        "Initializing Megakernel TTS service model=%s device=%s chunk_frames=%s",
        model_path,
        device,
        chunk_frames,
    )
    return MegakernelTTSService(
        model_path=model_path,
        device=device,
        chunk_frames=chunk_frames,
    )


def _client_label(client) -> str:
    for attr in ("id", "client_id", "participant_id", "session_id"):
        value = getattr(client, attr, None)
        if value:
            return str(value)
    return type(client).__name__


async def run_bot(transport):
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import LLMMessagesAppendFrame, LLMRunFrame
    from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.openai.llm import OpenAILLMService

    require_env("DEEPGRAM_API_KEY", "OPENAI_API_KEY")

    logger.info("Initializing Deepgram STT service")
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    logger.info("Initializing OpenAI LLM service model=%s", llm_model)
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model=llm_model,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.4,
            max_completion_tokens=160,
        ),
    )

    tts = create_tts_service()
    logger.info("Starting Megakernel TTS warmup")
    warmup_started = time.perf_counter()
    await tts.warmup()
    logger.info(
        "Megakernel TTS warmup complete duration_ms=%.1f",
        (time.perf_counter() - warmup_started) * 1000,
    )

    logger.info("Creating LLM context aggregators")
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    logger.info("Building voice pipeline: transport_in -> STT -> LLM -> TTS -> transport_out")
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            observers=[TranscriptionLogObserver()],
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected client=%s; queueing initial greeting", _client_label(client))
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    [
                        {
                            "role": "developer",
                            "content": "Greet the user briefly and say you are ready.",
                        }
                    ]
                ),
                LLMRunFrame(),
            ]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected client=%s; cancelling pipeline task", _client_label(client))
        await task.cancel()

    runner = PipelineRunner()
    logger.info("Starting Pipecat pipeline runner")
    try:
        await runner.run(task)
    finally:
        logger.info("Pipecat pipeline runner stopped")


async def bot(runner_args):
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.runner.utils import create_transport
    from pipecat.transports.base_transport import TransportParams
    from pipecat.transports.daily.transport import DailyParams

    logger.info("Creating runner transport for args=%s", type(runner_args).__name__)
    transport = await create_transport(
        runner_args,
        {
            "daily": lambda: DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
                vad_analyzer=SileroVADAnalyzer(),
            ),
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        },
    )
    await run_bot(transport)


if __name__ == "__main__":
    load_env_files()
    logging.basicConfig(level=logging.INFO)

    from pipecat.runner.run import main

    main()
