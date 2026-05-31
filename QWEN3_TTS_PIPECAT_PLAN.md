# Qwen3-TTS Pipecat Integration Plan

This note turns the take-home prompt and the reference implementation docs into a
concrete implementation plan for this repo.

Reference docs reviewed:

- `/private/tmp/qwen-megakernel-tts-docs/README.md`
- `/private/tmp/qwen-megakernel-tts-docs/docs/01-gpu-setup.md`
- `/private/tmp/qwen-megakernel-tts-docs/docs/02-kernel-adaptation.md`
- `/private/tmp/qwen-megakernel-tts-docs/docs/03-tts-pipeline-and-pipecat.md`
- `/private/tmp/qwen-megakernel-tts-docs/docs/04-performance-optimization.md`
- `/private/tmp/qwen-megakernel-tts-docs/docs/05-key-insights.md`

## Goal

Adapt AlpinDale's `qwen_megakernel` so it can serve as the decode backend for
Qwen3-TTS, then stream generated audio through a Pipecat voice agent pipeline.

The primary required acceleration target is the Qwen3-TTS talker decoder. The
performance-critical optional target is the 5-layer code predictor. The prompt
says the codebook generator is not the main target, but the reference docs show
that leaving the code predictor in PyTorch can dominate RTF.

The task is not to replace the chatbot LLM in Pipecat, and it is not initially
to port the speech tokenizer/vocoder to CUDA.

## Intended End-to-End Pipeline

```text
Mic audio
  -> Pipecat STT
  -> normal chat LLM
  -> text response
  -> custom Qwen3-TTS service
       -> Qwen3-TTS text/prompt preprocessing
       -> megakernel-backed talker decode
       -> code predictor: PyTorch first, megakernel for performance milestone
       -> official Qwen3-TTS speech tokenizer / vocoder
       -> streaming int16 PCM audio chunks
  -> Pipecat audio output
```

Qwen3-TTS generation shape:

- the talker produces the first codec group for each audio frame,
- the code predictor expands that into the remaining 15 codec groups,
- one complete speech frame has 16 code groups,
- speech frames are generated at roughly 12 Hz / 12.5 Hz,
- the vocoder converts code frames to 24 kHz PCM.

## What Carries Over From the Current Megakernel

The useful part of the current repo is the hand-written CUDA transformer decode
body:

```text
embedding/input state
  -> RMSNorm
  -> QKV projection
  -> RoPE
  -> attention with KV cache
  -> O projection
  -> MLP
  -> final norm
```

The Qwen3-TTS 0.6B talker is close to the current Qwen3-0.6B shape:

- 28 layers
- hidden size 1024
- intermediate size 3072
- 16 attention heads
- 8 KV heads
- head dim 128
- bfloat16-compatible weights

That makes the transformer body a plausible reuse target.

More concretely, the expensive part of the current CUDA kernel is not tied to
natural-language tokens. It is a fixed-shape decoder-only transformer block that
streams one position through:

```text
RMSNorm -> QKV matvec -> Q/K norm -> RoPE -> KV-cache attention
        -> O projection -> residual/post-norm -> gate/up/down MLP
```

Qwen3-TTS reuses a Qwen3-style decoder backbone for the talker, but changes the
surrounding contract. Instead of:

```text
text token id -> text embedding -> transformer -> 151936-way text LM head
```

the TTS path is closer to:

```text
text/codec conditioning embedding -> talker transformer -> 3072-way codec head
```

So the likely reusable part is the layer loop and its matvec/attention/MLP
implementation. The parts that need adaptation are the input embedding path, the
output head, the RoPE table configuration, and exact Qwen3-TTS prefill/position
semantics.

The current kernel also accepts `num_layers` at runtime rather than baking 28
layers into the CUDA loop. That matters because the code predictor is a smaller
transformer-style decoder with 5 layers. If RTF misses target with a PyTorch code
predictor, reuse the same compiled kernel with:

- `num_layers=5`,
- separate packed code-predictor weights,
- a separate code-predictor KV cache,
- the code predictor's own embedding/head handling.

This should be treated as a second integration target, not as automatic reuse.
The layer math can likely carry over, but the code predictor still needs its own
weight mapping, cache allocation, input construction, and output-token handling.

## Required Kernel / Runtime Changes

### 1. TTS Build Mode

Add a TTS-specific build path, likely `qwen_megakernel/build_tts.py`, that
changes compile-time constants for the codec head:

```text
LDG_VOCAB_SIZE: 151936 -> 3072
LDG_LM_NUM_BLOCKS: text-vocab scale -> around 16 blocks
```

The original text LM head scans `151936 x 1024`. The Qwen3-TTS talker codec head
is much smaller, about `3072 x 1024`, so the original text-vocab reduction shape
is wasteful.

Use a distinct extension name for the TTS build, `qwen_megakernel_tts_C`, so the
text and TTS variants can coexist in one Python process without fighting over
the `torch.ops` namespace.

### 2. Weight Loading

The current loader uses:

```python
AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
```

Qwen3-TTS uses Qwen3-TTS-specific model classes and a different state dict
layout. A new loader needs to:

- load the official Qwen3-TTS model,
- inspect and map talker layer weights,
- pack talker layer pointers into CUDA-readable structs,
- pack codec-head weights,
- keep official PyTorch modules available for the non-accelerated parts,
- optionally pack the code predictor's 5-layer weights for the performance
  milestone.

### 3. Precomputed Embedding Sentinel

Plain text decode receives one token id and does an embedding-table lookup. TTS
decode input is often a combined embedding, not a single token id: previous 16
codec group embeddings plus the current/advancing text embedding.

In the original text path, layer 0 can cheaply do:

```text
input_token_id -> embed_weight[input_token_id] -> layer 0 input
```

That works because every decode step has exactly one discrete text token. In the
Qwen3-TTS frame loop, the next talker input is already a fused vector assembled
from multiple sources:

```text
sum(codec_embedding[group_i, code_i] for i in 0..15)
  + current/trailing text embedding
  + any fixed prompt or special-token contribution required by the official path
```

There is no single row in `embed_weight` that represents this sum. Launching a
separate CUDA kernel every frame just to build the fused embedding would add
avoidable launch overhead in the hottest path. The sentinel approach lets Python
or a lightweight existing Torch path prepare the fused 1024-wide bf16 vector,
then lets the megakernel consume it as if it were the normal layer-0 input.

Add a small sentinel path in CUDA:

```c
// token_id >= 0: normal embedding table lookup
// token_id == -1: use hidden_buffer/precomputed embedding as layer input
const __nv_bfloat16 *embed_row =
    (input_token_id >= 0) ? embed_weight + input_token_id * HIDDEN_SIZE
                          : hidden_buffer;
```

Python prepares the combined embedding into the expected buffer and passes
`token_id=-1`. Keep the normal token-id path intact for prefill and compatibility.

Implementation constraints:

- write the precomputed embedding before launching the decode kernel,
- keep it on GPU as bf16 and contiguous with length `HIDDEN_SIZE`,
- avoid `.cpu()`, `.item()`, or host-side copies in the per-frame path,
- make sure the buffer used as layer-0 input is not overwritten before layer 0
  reads it,
- keep the standard `token_id >= 0` path for ordinary token prefill steps and
  for regression testing against the original text model.

Using `hidden_buffer` is convenient because the kernel already has it and layer 0
can read from it directly. The main caveat is lifetime/aliasing: later layers
also write their outputs into `hidden_buffer`, so the fused input only needs to
survive until layer 0 has consumed it. That is fine for a single decode launch,
but the Python wrapper should treat the buffer as scratch and rewrite it before
each sentinel step.

### 4. Codec Head Instead of Text LM Head

The current repo decodes text tokens:

```text
hidden[1024] -> lm_head[151936 x 1024] -> argmax text token
```

Qwen3-TTS talker decode should produce the first speech codec group:

```text
hidden[1024] -> codec_head[3072 x 1024] -> first codec token/group
```

The large text LM-head kernel should become a smaller codec-head path.

### 5. RoPE Tables and M-RoPE Risk

Qwen3-TTS uses a different RoPE base, reported in the reference docs as
`1_000_000` instead of `10_000`. Regenerate the Python-side `cos` and `sin`
tables accordingly.

The base megakernel implements standard RoPE, not Qwen3-TTS M-RoPE. For
text-only/simple TTS inputs, standard RoPE may be good enough for a demo, but
EOS/stop behavior and longer utterance quality can degrade. If M-RoPE is not
implemented, document this clearly and use a bounded word-count-based frame cap
as a temporary workaround.

## Qwen3-TTS Pipeline Details To Match

The official generation format matters for quality.

Implementation notes from the reference docs:

- prefill is an 8-step sequence,
- it includes special "thinking" tokens reported as `2155`, `2156`, `2157`,
- the generation loop advances a text embedding per frame,
- strip official chat-template tail tokens from the text stream before feeding
  the frame loop,
- generic padding or extra template tokens can produce degraded or garbage
  audio.

Correctness work should follow the official Qwen3-TTS source code exactly before
judging model quality.

## Code Predictor Strategy

Initial correctness milestone:

```text
Talker transformer decode: CUDA megakernel
Code predictor: official PyTorch/Qwen code
Speech tokenizer/vocoder: official PyTorch/Qwen code
```

Performance milestone:

```text
Talker transformer decode: CUDA megakernel
Code predictor: same megakernel with num_layers=5
Speech tokenizer/vocoder: official PyTorch/Qwen code
```

The reference docs report that PyTorch code predictor took about `179 ms/frame`,
which misses the RTF target by itself. Reusing the megakernel with `num_layers=5`
reportedly reduced this to about `10.9 ms/frame`. Treat this as a likely
required optimization for a competitive submission, even though it is beyond the
prompt's narrow "talker decoder" wording.

Keep all per-frame and per-code tokens on GPU. Avoid `.item()` inside decode
loops; use tensor outputs such as `argmax(keepdim=True)` or GPU-side sampled
tokens to avoid CPU/GPU synchronization.

## Streaming TTS Service Shape

Expose a custom Pipecat TTS service:

```text
Custom Pipecat TTSService
  run_tts(text) yields:
    TTSStartedFrame
    TTSAudioRawFrame chunks with int16 PCM audio
    TTSStoppedFrame
```

Operational requirements:

- the frame generator must be a real generator or async generator using `yield`,
  not `list.append` followed by `return`,
- emit the first decoded frame immediately for low TTFC,
- batch later chunks, for example around 10 frames, for efficiency,
- convert vocoder `float32` output to Pipecat-compatible 16-bit PCM,
- include a text-only demo mode so TTS latency and audio quality can be tested
  without mic, STT, or external LLM keys,
- include a full voice-agent demo using STT -> LLM -> megakernel TTS -> audio
  output.

## Warmup Requirements

Warmup is part of the product, not just benchmarking. The reference docs report
large first-call penalties from vocoder initialization and sampling ops.

Run startup warmups for:

- megakernel compilation/decode path,
- talker argmax path,
- code predictor path if used,
- sampling path: `softmax`, `topk`, `multinomial`,
- vocoder decode with representative dummy code-frame sizes.

Precompute static embeddings for:

- role tokens,
- special TTS tokens,
- codec tags,
- fixed prompt markers,
- any fused constant embeddings used by the prefill path.

## Implementation Steps

1. Verify the original megakernel benchmark on RTX 5090:
   `python3 -m qwen_megakernel.bench`.
2. Get official Qwen3-TTS 0.6B generation running locally.
3. Dump Qwen3-TTS talker state dict keys and tensor shapes.
4. Confirm talker layer shapes match the existing megakernel assumptions.
5. Add the TTS build mode with `LDG_VOCAB_SIZE=3072` and smaller codec-head
   block count.
6. Add the TTS-specific Python wrapper and talker weight packer.
7. Regenerate RoPE tables with the TTS RoPE base.
8. Add the negative-token embedding sentinel path.
9. Adapt the CUDA output head from text vocab projection to codec projection.
10. Match official Qwen3-TTS prefill and text-token handling exactly.
11. Verify deterministic talker hidden/logit/first-codec outputs against the
    official PyTorch path on a short prompt.
12. Build a functional hybrid pipeline with PyTorch code predictor and vocoder.
13. Benchmark RTF. If RTF misses target, pack the 5-layer code predictor and run
    it through the megakernel with `num_layers=5`.
14. Expose a streaming TTS interface: text in, PCM chunks out.
15. Build Pipecat text-only and full voice-agent demos.
16. Benchmark and document final numbers.

## Validation Plan

Correctness checks:

- compare megakernel talker hidden/logit outputs against official PyTorch for
  deterministic short prompts,
- validate first codec group parity before testing waveform quality,
- validate full 16-code frame generation,
- confirm generated audio has no trailing chat-template artifacts,
- confirm no `.item()` or CPU syncs inside per-frame/per-code hot loops,
- confirm streaming starts before full utterance generation completes.

Audio checks:

- short, medium, and long text samples,
- listen for glitches, dropped frames, template artifacts, and long-tail
  degradation,
- document whether M-RoPE is implemented or skipped.

Pipecat checks:

- `TTSStartedFrame` appears before audio,
- `TTSAudioRawFrame` chunks are emitted incrementally,
- `TTSStoppedFrame` appears after generation,
- text-only demo works without external API keys,
- full voice pipeline works with configured STT/LLM providers.

## Benchmarking Plan

Use synchronized GPU timing for measured GPU sections:

```python
torch.cuda.synchronize()
start = time.perf_counter()
# measured work
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

Run warmups before benchmark measurements. Exclude one-time JIT/model download
costs from per-request numbers, but document startup cost separately.

Report:

- original megakernel text decode tok/s,
- Qwen3-TTS talker decode step time,
- code predictor time per frame, PyTorch vs megakernel if implemented,
- TTFC from text input to first PCM frame emitted,
- RTF as generation wall time divided by generated audio duration,
- end-to-end voice-agent latency.

TTFC breakdown:

```text
tokenization
text embedding/projection
prefill
first talker decode
first code predictor
first vocoder decode
Pipecat frame emission
```

RTF breakdown:

```text
talker decode per frame
code predictor per frame
embedding work per frame
vocoder amortized cost
Pipecat/audio framing overhead
```

Reference-doc numbers to use as sanity checks, not guarantees:

- streaming TTFC around `81.6 ms`,
- streaming RTF around `0.234`,
- talker decode around `1 ms/step`,
- code predictor around `10.9 ms/frame` when run through the megakernel,
- PyTorch code predictor around `179 ms/frame`.

## Setup Expectations

Target environment:

- NVIDIA RTX 5090 / Blackwell `sm_120a`,
- CUDA 12.8+,
- PyTorch with CUDA 12.8 support,
- Python 3.10+; Python 3.12 worked in the reference docs.

Useful setup checks:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -m qwen_megakernel.bench
```

For a CUDA 12.8 PyTorch wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Risks and Challenges

- Qwen3-TTS talker input is a combined embedding, not always a single token id.
- Prefill format is specific and partly undocumented; exact source-code matching
  matters for audio quality.
- M-RoPE is not implemented in the base megakernel.
- If M-RoPE is skipped, EOS/stop detection may be unreliable.
- TTS may require sampling rather than greedy decoding for good quality.
- CPU/GPU syncs inside the code predictor loop can erase latency gains.
- TTFC may be dominated by prefill, code predictor, vocoder, or first-call
  warmup rather than the talker transformer.
- Buffering generated frames before yielding will fail the streaming
  requirement.
- Talker-only acceleration may not hit RTF if the code predictor remains in
  PyTorch.

## Success Criteria

A good submission should show:

- original Qwen3 megakernel still works,
- Qwen3-TTS audio is emitted incrementally as Pipecat audio frames,
- the talker decoder uses the megakernel path,
- the code predictor path is either measured honestly in PyTorch or accelerated
  with the megakernel,
- warmups are implemented and benchmark methodology is explicit,
- TTFC, RTF, decode tok/s, and end-to-end latency are reported,
- the README clearly states which parts are accelerated and which remain in
  PyTorch,
- limitations such as skipped M-RoPE, EOS heuristics, or quality issues are
  documented.

The core claim should be narrow and testable: the AlpinDale transformer
megakernel has been adapted to sit inside Qwen3-TTS's talker decode path while
the rest of the Qwen3-TTS and Pipecat pipeline remains functional and streams
audio frame-by-frame.
