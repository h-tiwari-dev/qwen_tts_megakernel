# Qwen3-TTS Megakernel Implementation README

This repo now has a first-pass Qwen3-TTS integration on top of AlpinDale's
Qwen3-0.6B RTX 5090 megakernel.

The implementation goal is narrow: use the CUDA megakernel as the fast decode
backend for the Qwen3-TTS talker path, keep the official Qwen3-TTS components
around it, and stream audio through a Pipecat-compatible TTS service.

## What Changed

### CUDA Kernel

`csrc/kernel.cu` now supports two TTS requirements:

- `LDG_VOCAB_SIZE` is compile-time configurable instead of always being
  `151936`.
- `token_id < 0` is a sentinel that makes layer 0 read from `hidden_buffer`
  instead of doing an embedding table lookup.

The sentinel path is needed because Qwen3-TTS often feeds a fused embedding into
the talker decoder, not a single token id.

### TTS Build Variant

`qwen_megakernel/build_tts.py` builds a separate TTS extension:

```text
extension name: qwen_megakernel_tts_C
LDG_VOCAB_SIZE: 3072
LDG_LM_NUM_BLOCKS: 16
target arch: sm_120a
```

The distinct extension name avoids collisions with the original text extension
`qwen_megakernel_C`.

### TTS Runtime

New runtime files:

- `qwen_megakernel/model_tts.py`
  - loads Qwen3-TTS safetensors,
  - packs talker weights into the megakernel layer struct,
  - exposes `TTSDecoder`,
  - exposes `CodePredictorKernel` using the same kernel with `num_layers=5`,
  - implements text projection helpers.

- `qwen_megakernel/tts_engine.py`
  - orchestrates tokenization, prefill, talker decode, code predictor, vocoder,
    and streaming audio chunks.

- `qwen_megakernel/pipecat_tts.py`
  - exposes `MegakernelTTSService`, a Pipecat `TTSService` whose `run_tts()`
    yields PCM audio frames. Current Pipecat creates the TTS context/start frame
    before calling `run_tts()` and closes the context after generated audio
    finishes.

### Demos and Benchmarks

New entry points:

- `demo_tts.py`: text to WAV.
- `demo_pipeline.py`: streaming TTS demo with TTFC/RTF reporting.
- `demo_voice_agent.py`: Pipecat voice-agent demo and text-only mode.
- `benchmark.py`: aggregate TTS benchmark.
- `benchmarks/`: detailed TTFC, RTF, tok/s, and end-to-end measurements.
- `test_e2e.py`, `test_cp_kernel.py`, `validate_kernel.py`: validation helpers.

## Expected Environment

This code is intended for:

- NVIDIA RTX 5090 / Blackwell `sm_120a`
- CUDA 12.8+
- PyTorch CUDA 12.8 build
- Python 3.10+

Useful setup check:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch needs to be installed manually for CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Run Commands

Original text megakernel benchmark:

```bash
python -m qwen_megakernel.bench
```

Standalone Qwen3-TTS synthesis:

```bash
python demo_tts.py "Hello, this is a test." --output /tmp/tts.wav
```

Streaming Qwen3-TTS demo:

```bash
python demo_pipeline.py --text "Hello from the streaming pipeline."
```

Main benchmark:

```bash
python benchmark.py --runs 3
```

Detailed benchmarks:

```bash
python -m benchmarks.measure_tok_s
python -m benchmarks.measure_ttfc
python -m benchmarks.measure_rtf
python -m benchmarks.measure_e2e
```

Pipecat text-only demo:

```bash
python demo_voice_agent.py --text-only
```

Full Pipecat voice agent:

```bash
export DEEPGRAM_API_KEY=...
export OPENAI_API_KEY=...
python demo_voice_agent.py --transport websocket --port 8765
```

## Validation Order

Use this order on the RTX 5090 machine:

1. Install dependencies.
2. Run `python -m qwen_megakernel.bench` to confirm the original text path.
3. Run `python demo_tts.py "Hello" --output /tmp/tts.wav`.
4. Run `python demo_pipeline.py --text "Hello from streaming."`.
5. Run `python benchmark.py --runs 3`.
6. Run `python demo_voice_agent.py --text-only`.
7. Only after TTS works, try the full STT -> LLM -> TTS Pipecat pipeline.

## Static Review Notes

The CUDA/build integration was reviewed for extension naming and operator
namespace consistency:

- `csrc/torch_bindings.cpp` registers ops under `TORCH_EXTENSION_NAME`.
- `qwen_megakernel/build.py` still loads the original text extension as
  `qwen_megakernel_C`.
- `qwen_megakernel/build_tts.py` loads the TTS extension as
  `qwen_megakernel_tts_C`.
- `qwen_megakernel/model.py` calls `torch.ops.qwen_megakernel_C.*`.
- `qwen_megakernel/model_tts.py` and `test_cp_kernel.py` call
  `torch.ops.qwen_megakernel_tts_C.decode`.
- `qwen_megakernel/__init__.py` defers CUDA compilation until the text or TTS
  modules are imported.

Python syntax checks pass for the new modules, demos, benchmarks, and validation
helpers. CUDA compilation was not run in this editing environment.

## Known Limitations

- Full CUDA verification was not run in the local editing environment because it
  does not have PyTorch/CUDA/RTX 5090 available.
- The implementation follows the reference repo's pragmatic path and does not
  implement full Qwen3-TTS M-RoPE in the CUDA kernel.
- EOS/stop behavior may be unreliable if M-RoPE mismatch affects long sequences.
  The engine uses a word-count-based frame cap as a fallback.
- The code predictor is accelerated through the megakernel, but still computes
  per-group `2048`-way heads outside the megakernel.
- The talker path returns tokens through the existing output-token mechanism,
  which includes some synchronization. Further optimization could add a
  no-head/no-sync hidden-state decode op for predictor-only use cases.

## Files To Review First If Something Breaks

- CUDA compile or extension issues:
  - `qwen_megakernel/build_tts.py`
  - `csrc/kernel.cu`
  - `csrc/torch_bindings.cpp`

- Missing or mismatched Qwen3-TTS weights:
  - `qwen_megakernel/model_tts.py`

- Bad audio quality or no audio:
  - `qwen_megakernel/tts_engine.py`
  - Qwen3-TTS prefill/token formatting
  - vocoder loading in `_load_vocoder`

- Pipecat frame issues:
  - `qwen_megakernel/pipecat_tts.py`
  - `demo_voice_agent.py`

## Related Planning Doc

See `QWEN3_TTS_PIPECAT_PLAN.md` for the detailed design rationale, risks, and
benchmarking plan.
