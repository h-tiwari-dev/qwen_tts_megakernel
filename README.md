# Qwen3-TTS Megakernel Voice Agent

This is my RTX 5090 Qwen3-TTS take-home implementation. I started from
AlpinDale's Qwen3 decode megakernel and adapted it so the Qwen3-TTS talker
decoder can run through the CUDA megakernel, then wired that TTS path into a
Pipecat voice-agent pipeline.

The end-to-end voice path is:

```text
Browser/Daily mic
  -> Pipecat transport
  -> Deepgram STT
  -> OpenAI LLM
  -> Megakernel-backed Qwen3-TTS service
  -> Pipecat audio output
```

The original megakernel benchmark from this repo is still available:

| Backend      | tok/s  | ms/tok | Speedup |
| ------------ | ------ | ------ | ------- |
| PyTorch (HF) | 123.3  | 8.11   | 1.00x   |
| Megakernel   | 1036.3 | 0.99   | 8.40x   |

More context on the original kernel: https://blog.alpindale.net/posts/5090_decode_optimization/

## What I Changed

I added a Qwen3-TTS path alongside the original text decode path:

- `qwen_megakernel/build_tts.py` builds a separate TTS extension named
  `qwen_megakernel_tts_C`.
- `csrc/kernel.cu` supports a configurable codec vocab size and a sentinel
  input-token mode so layer 0 can consume a precomputed hidden embedding.
- `qwen_megakernel/model_tts.py` loads Qwen3-TTS talker/code-predictor weights
  and exposes megakernel-backed decode wrappers.
- `qwen_megakernel/tts_engine.py` orchestrates text tokenization, talker
  prefill, codec-frame generation, optional voice-clone prompting, vocoder
  decode, and streaming audio chunks.
- `qwen_megakernel/pipecat_tts.py` exposes the engine as a Pipecat `TTSService`.
- `bot.py` runs the full Pipecat voice pipeline.

The megakernel is used for the Qwen3-TTS talker decoder. I also route the
5-layer code predictor through the same extension because it was a practical
bottleneck during integration. The official Qwen components are still used for
text/tokenizer utilities, voice-clone prompt construction, and vocoder/audio
decode.

## Current Voice Defaults

The default voice settings are optimized for lower first-audio latency:

```bash
OPENAI_MODEL=gpt-4.1-mini
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
QWEN_TTS_CHUNK_FRAMES=2
QWEN_TTS_WARMUP_PROFILE=fast
QWEN_TTS_STREAMING_MODE=chunked
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

`chunked` mode pushes audio to Pipecat as chunks are decoded instead of waiting
for a full utterance. If the vocoder path is unstable on a given environment,
set:

```bash
QWEN_TTS_STREAMING_MODE=full_decode
```

In `full_decode` mode, if the megakernel vocoder path is unavailable and
reference audio/text are configured, the service falls back to official Qwen
voice-clone synthesis rather than emitting silence.

Reference prompt failures are non-fatal by default so the bot can still start.
Set `QWEN_TTS_REQUIRE_REF_PROMPT=true` if you want startup to fail hard when the
reference prompt cannot be built. Set both of these to empty strings to disable
reference prompting:

```bash
QWEN_TTS_REF_AUDIO=
QWEN_TTS_REF_TEXT=
```

## Environment

This is intended for:

- NVIDIA RTX 5090 / Blackwell `sm_120a`
- CUDA 12.8+
- PyTorch CUDA 12.8 build
- Python 3.10+

Install:

```bash
pip install -r requirements.txt
```

If PyTorch needs to be installed manually:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Quick GPU sanity check:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Run It

Original text megakernel benchmark:

```bash
python -m qwen_megakernel.bench
```

Standalone TTS to WAV:

```bash
python demo_tts.py "Hello, this is a test." --output /tmp/tts.wav
```

Streaming TTS demo with TTFC/RTF reporting:

```bash
python demo_pipeline.py --text "Hello from the streaming pipeline."
```

Text-only Pipecat TTS service test:

```bash
python demo_voice_agent.py --text-only
```

Full Pipecat WebRTC demo:

```bash
export DEEPGRAM_API_KEY=...
export OPENAI_API_KEY=...
python bot.py -t webrtc --host 0.0.0.0 --port 7860
```

Then open:

```text
http://localhost:7860/client
```

Daily mode for Vast/cloud:

```bash
export DAILY_API_KEY=...
export DEEPGRAM_API_KEY=...
export OPENAI_API_KEY=...
python bot.py -t daily --host 0.0.0.0 --port 7860
```

I use Daily as the browser/WebRTC interface for the hosted demo. The direct
Pipecat WebRTC transport is useful locally, but on Vast/cloud the browser is
usually behind NAT/port mapping and ICE can be brittle. Daily creates the room,
handles WebRTC signaling/NAT traversal, and gives me a stable URL for the human
participant. The logs print the Daily room URL; join that URL from the browser.

This is also the command I use for whole-pipeline metrics: the bot attaches
Pipecat's metrics observer and user-to-bot latency observer, so service timings
and end-to-end voice-turn latency are emitted directly in the run log.

## Vast.ai Port Mapping

For the WebRTC app on internal port `7860`, create the instance with:

```bash
-p 7860:7860 -e OPEN_BUTTON_PORT=7860
```

Run the bot bound to all interfaces:

```bash
python bot.py -t webrtc --host 0.0.0.0 --port 7860
```

Then use the external Vast port mapping, for example:

```text
65.130.162.74:33526 -> 7860/tcp
```

Open:

```text
http://65.130.162.74:33526/client
```

Do not use `localhost` from your laptop for a Vast instance. The app must bind
to `0.0.0.0`.

## Validation Order

I use this order on the RTX 5090 machine:

1. `python3 -m py_compile bot.py qwen_megakernel/pipecat_tts.py qwen_megakernel/tts_engine.py`
2. `python -m qwen_megakernel.bench`
3. `python demo_tts.py "Hello" --output /tmp/tts.wav`
4. `python demo_pipeline.py --text "Hello from streaming."`
5. `python benchmark.py --runs 3`
6. `python demo_voice_agent.py --text-only`
7. Full browser/Daily Pipecat voice demo

Detailed benchmark entry points:

```bash
python -m benchmarks.measure_tok_s
python -m benchmarks.measure_ttfc
python -m benchmarks.measure_rtf
python -m benchmarks.measure_e2e
```

The full Daily/WebRTC command logs whole voice-turn metrics directly; I do not
run a separate parser. Look for `Voice pipeline latency user_to_bot_ms=...` and
`Voice pipeline latency breakdown:` in the bot log.

## Metrics and Logs

The bot logs startup and runtime metrics at `INFO`, including:

- whole user-to-bot response latency from Pipecat's `UserBotLatencyObserver`
- per-service metrics from Pipecat's `MetricsLogObserver`
- TTS warmup duration
- client connect/disconnect
- first-audio latency / TTFC
- per-chunk wall gap
- per-chunk audio duration
- total generated audio duration
- wall-clock TTS duration
- RTF
- chunk count
- output bytes

Logs default to `INFO`. Use these only when debugging low-level Pipecat or audio
frame behavior:

```bash
QWEN_TTS_FILE_LOG_LEVEL=DEBUG
QWEN_TTS_PIPECAT_FILE_LOG_LEVEL=DEBUG
```

For the final report I would include two groups of metrics:

- **Megakernel/TTS metrics** from `benchmark.py` and `benchmarks.measure_*`:
  decode tok/s, TTFC, RTF, chunk count, and inter-chunk timing.
- **Whole pipeline metrics** from the normal `python bot.py -t daily ...` run:
  user-to-bot latency, first bot speech latency, service TTFB/processing
  breakdowns, TTS service latency, TTS RTF, and end-to-end turn notes from the
  real Pipecat/Daily demo.

## What To Submit

For the take-home I would include:

- this repo with the build/run instructions above,
- a short demo video showing me joining the voice agent, speaking, and hearing
  the agent reply,
- benchmark output for megakernel tok/s, TTFC, RTF, and end-to-end latency,
- the environment used for the run: GPU, CUDA, PyTorch, model revision, and env
  vars,
- an honest note about any fallbacks used during the recording.

## Current Caveats

- The talker decode path is megakernel-backed. The official Qwen vocoder is
  still used for codec-to-waveform audio.
- `chunked` mode is the default because the assignment requires streaming audio
  into Pipecat. `full_decode` remains available as a stability fallback.
- The voice-clone prompt is cached in `~/.cache/qwen_megakernel` after the first
  successful build.
- The code predictor is accelerated through the megakernel wrapper, but the
  per-group `2048`-way heads are still outside the kernel.
- I did not implement full Qwen3-TTS M-RoPE in CUDA. The current implementation
  follows the pragmatic integration path and uses a frame cap to avoid runaway
  generation if EOS behavior is unreliable.
- The service refuses to emit silent audio. If the vocoder fails and no official
  fallback is available, the Pipecat TTS frame returns an error instead.

## Files To Review

- CUDA kernel changes: `csrc/kernel.cu`, `csrc/torch_bindings.cpp`
- TTS extension build: `qwen_megakernel/build_tts.py`
- Qwen3-TTS weights and wrappers: `qwen_megakernel/model_tts.py`
- TTS orchestration: `qwen_megakernel/tts_engine.py`
- Pipecat service: `qwen_megakernel/pipecat_tts.py`
- Voice bot: `bot.py`

## Credits

Based on Elliot Arledge's MegaQwen / Qwen megakernel work:

- https://github.com/AlpinDale/qwen_megakernel
- https://blog.alpindale.net/posts/5090_decode_optimization/
