## Qwen 0.6B Megakernel for RTX 5090

This megakernel is aggressively optimized for Qwen3-0.6B (bf16) shapes to be run on an RTX 5090.

More details on this blogpost: https://blog.alpindale.net/posts/5090_decode_optimization/

| Backend      | tok/s  | ms/tok | Speedup |
| ------------ | ------ | ------ | ------- |
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
# Fill in DEEPGRAM_API_KEY, OPENAI_API_KEY, and DAILY_API_KEY if using Daily.
uv pip install -r requirements.txt
python bot.py -t webrtc --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860/client` in a browser. The audio path is:

```text
Browser mic -> Pipecat WebRTC -> Deepgram STT -> OpenAI gpt-4.1-mini
  -> Megakernel TTS -> browser audio output
```

Pipecat Daily voice demo for cloud/Vast.ai:

```bash
export DAILY_API_KEY=your-daily-api-key
python bot.py -t daily --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860/daily` locally, or the mapped Vast TCP port
for internal `7860` plus `/daily`. Daily creates the room and handles WebRTC
NAT traversal, so this avoids the SmallWebRTC ICE failures that happen on
shared/NATed cloud hosts.

The WebRTC runner loads `.env` and `.env.local`, requires
`DEEPGRAM_API_KEY` and `OPENAI_API_KEY`. Daily mode also requires
`DAILY_API_KEY`. Defaults:

```bash
OPENAI_MODEL=gpt-4.1-mini
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
QWEN_TTS_CHUNK_FRAMES=10
QWEN_TTS_WARMUP_PROFILE=fast
QWEN_TTS_STREAMING_MODE=full_decode
QWEN_TTS_BACKEND=megakernel
QWEN_TTS_REF_AUDIO=https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav
QWEN_TTS_REF_TEXT="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
QWEN_TTS_VOICE_PROMPT_CACHE=1
QWEN_TTS_VOICE_PROMPT_CACHE_DIR=~/.cache/qwen_megakernel
QWEN_TTS_REQUIRE_REF_PROMPT=false
QWEN_TTS_X_VECTOR_ONLY=false
QWEN_TTS_DO_SAMPLE=false
QWEN_TTS_TEMPERATURE=0.9
QWEN_TTS_TOP_K=50
QWEN_TTS_SUBTALKER_DO_SAMPLE=false
QWEN_TTS_SUBTALKER_TEMPERATURE=0.9
QWEN_TTS_SUBTALKER_TOP_K=50
```

Use `QWEN_TTS_WARMUP_PROFILE=full` for more exhaustive startup warmup if first
utterance latency matters more than process startup time. The default TTS
settings favor voice stability while the megakernel prompt path is being
validated: deterministic sampling, cached voice-clone prompting, and
full-utterance vocoder decode. Set
`QWEN_TTS_STREAMING_MODE=chunked` for the low-latency streaming path after
audio quality is confirmed. The default reference audio/text are the official
Qwen3-TTS voice-clone sample; set `QWEN_TTS_REF_AUDIO=` and
`QWEN_TTS_REF_TEXT=` to disable reference prompting. Reference prompt failures
are non-fatal by default so the bot can still start; set
`QWEN_TTS_REQUIRE_REF_PROMPT=true` to fail fast instead. Vocoder failures remain
fatal for synthesis because the service refuses to emit silent audio. The
default `fast` warmup profile skips vocoder decode warmup; `full` runs a
vocoder decode check and disables the vocoder if that check fails. In
`full_decode` mode, a missing megakernel vocoder falls back to official
Qwen voice-clone synthesis when reference audio/text are configured.

For Base-model voice cloning on the megakernel path, provide a reference clip
and its transcript:

```bash
export QWEN_TTS_REF_AUDIO=/path/to/reference.wav
export QWEN_TTS_REF_TEXT="Exact transcript of the reference audio."
```

This builds the official Qwen3-TTS voice-clone prompt once during warmup,
injects the speaker embedding into the megakernel talker prefill, and uses ICL
reference text/code conditioning. If you only want the x-vector speaker
embedding and no reference-code ICL, set `QWEN_TTS_X_VECTOR_ONLY=true`.

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
duration, RTF, per-chunk wall gaps, per-chunk audio duration, chunk count, and
output byte counts. File and console logs default to `INFO`; set
`QWEN_TTS_FILE_LOG_LEVEL=DEBUG` or `QWEN_TTS_PIPECAT_FILE_LOG_LEVEL=DEBUG` only
when byte/frame-level diagnostics are needed.

Implementation notes and risks are tracked in
[`QWEN3_TTS_PIPECAT_PLAN.md`](QWEN3_TTS_PIPECAT_PLAN.md). The current
implementation status and runbook are in
[`QWEN3_TTS_IMPLEMENTATION_README.md`](QWEN3_TTS_IMPLEMENTATION_README.md).

### Credits

Based on Elliot Arledge's [MegaQwen](https://github.com/Infatoshi/MegaQwen) for the RTX 3090 GPU.

---

• To expose your app port on Vast.ai, add a Docker port mapping when creating/editing the instance.

For this repo’s WebRTC bot on internal port 7860, use Docker create options:

-p 7860:7860 -e OPEN_BUTTON_PORT=7860

Then run the bot inside the instance bound to all interfaces:

python bot.py -t webrtc --host 0.0.0.0 --port 7860

After the instance starts, open IP Port Info in Vast.ai and find the mapping for 7860/tcp, for example:

65.130.162.74:33526 -> 7860/tcp

Then open:

http://65.130.162.74:33526/client

Important details:

- The external port will usually be random.
- Do not use localhost from your laptop; use the Vast public IP and mapped external port.
- The app must bind to 0.0.0.0, not 127.0.0.1.
- OPEN_BUTTON_PORT=7860 makes the Vast “Open” button target the mapped external port for internal 7860.
