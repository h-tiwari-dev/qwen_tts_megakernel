## Qwen 0.6B Megakernel for RTX 5090

This megakernel is aggressively optimized for Qwen3-0.6B (bf16) shapes to be run on an RTX 5090.

More details on this blogpost: https://blog.alpindale.net/posts/5090_decode_optimization/


| Backend      | tok/s  | ms/tok | Speedup |
|--------------|--------|--------|---------|
| PyTorch (HF) | 123.3  | 8.11   | 1.00x   |
| Megakernel   | 1036.3 | 0.99   | 8.40x   |


To use this:

```bash
uv pip install -r requirements.txt
python -m qwen_megakernel.bench
```

Not tested on any other GPU, and likely won't run or work. Needs at least CUDA 12.8.

### Qwen3-TTS / Pipecat

This fork also includes a Qwen3-TTS integration path:

- `qwen_megakernel/build_tts.py` builds a TTS variant of the extension with a
  `3072`-way codec head.
- `qwen_megakernel/model_tts.py` loads Qwen3-TTS talker/code-predictor weights
  and wraps the megakernel decode calls.
- `qwen_megakernel/tts_engine.py` streams text-to-audio chunks.
- `qwen_megakernel/pipecat_tts.py` exposes a Pipecat `TTSService`.

Standalone TTS:

```bash
python demo_tts.py "Hello, this is a test." --output /tmp/tts.wav
```

Streaming TTS with TTFC/RTF reporting:

```bash
python demo_pipeline.py --text "Hello from the streaming pipeline."
```

Pipecat text-only demo:

```bash
python demo_voice_agent.py --text-only
```

Pipecat WebRTC voice demo:

```bash
cp .env.example .env
# Fill in DEEPGRAM_API_KEY and OPENAI_API_KEY.
uv pip install -r requirements.txt
python bot.py -t webrtc --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860/client` in a browser. The audio path is:

```text
Browser mic -> Pipecat WebRTC -> Deepgram STT -> OpenAI gpt-4.1-mini
  -> Megakernel TTS -> browser audio output
```

The WebRTC runner loads `.env` and `.env.local`, requires
`DEEPGRAM_API_KEY` and `OPENAI_API_KEY`, and defaults to:

```bash
OPENAI_MODEL=gpt-4.1-mini
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
QWEN_TTS_CHUNK_FRAMES=10
```

Before running the full browser demo, the lightweight checks are:

```bash
python3 -m py_compile bot.py qwen_megakernel/pipecat_tts.py qwen_megakernel/tts_engine.py
python3 -c "import qwen_megakernel; print('ok')"
python demo_tts.py "Hello, this is a test." --output /tmp/tts.wav
python demo_pipeline.py --text "Hello from the streaming pipeline."
```

The WebRTC bot logs startup and runtime metrics at `INFO`, including service
initialization, TTS warmup time, client connect/disconnect, pipeline
start/stop, first-audio latency, generated audio duration, wall-clock TTS
duration, RTF, chunk count, and output bytes. Per-chunk TTS details are emitted
at `DEBUG` if deeper streaming diagnostics are needed.

Implementation notes and risks are tracked in
[`QWEN3_TTS_PIPECAT_PLAN.md`](QWEN3_TTS_PIPECAT_PLAN.md). The current
implementation status and runbook are in
[`QWEN3_TTS_IMPLEMENTATION_README.md`](QWEN3_TTS_IMPLEMENTATION_README.md).


### Credits
Based on Elliot Arledge's [MegaQwen](https://github.com/Infatoshi/MegaQwen) for the RTX 3090 GPU.
