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

Implementation notes and risks are tracked in
[`QWEN3_TTS_PIPECAT_PLAN.md`](QWEN3_TTS_PIPECAT_PLAN.md). The current
implementation status and runbook are in
[`QWEN3_TTS_IMPLEMENTATION_README.md`](QWEN3_TTS_IMPLEMENTATION_README.md).


### Credits
Based on Elliot Arledge's [MegaQwen](https://github.com/Infatoshi/MegaQwen) for the RTX 3090 GPU.
