"""Full TTS pipeline: text → talker (megakernel) → code predictor → vocoder → audio.

This module orchestrates all components:
  1. Text tokenization and embedding
  2. Prefill phase (via megakernel step-by-step)
  3. Autoregressive decode (megakernel talker + megakernel code predictor)
  4. Vocoder decode (codec tokens → waveform)

Streaming: yields audio chunks as codec frames accumulate.
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Generator, Optional

import numpy as np
import torch

from .model_tts import (
    CODEC_BOS,
    CODEC_EOS,
    CODEC_NOTHINK,
    CODEC_PAD,
    CODEC_THINK_BOS,
    CODEC_THINK_EOS,
    HIDDEN_SIZE,
    NUM_CODE_GROUPS,
    TTS_BOS,
    TTS_EOS,
    TTS_PAD,
    CodePredictorKernel,
    TextProjection,
    TTSDecoder,
    load_tts_weights,
)


def _import_transformers_auto(name: str):
    """Import HF auto classes with a clearer error for broken environments."""
    try:
        import transformers
        value = getattr(transformers, name, None)
        if value is not None:
            return value
    except Exception as exc:
        top_level_error = exc
    else:
        top_level_error = None

    fallback_modules = {
        "AutoTokenizer": "transformers.models.auto.tokenization_auto",
        "AutoConfig": "transformers.models.auto.configuration_auto",
        "AutoModel": "transformers.models.auto.modeling_auto",
    }
    module_name = fallback_modules.get(name)
    if module_name is not None:
        try:
            module = __import__(module_name, fromlist=[name])
            return getattr(module, name)
        except Exception as exc:
            fallback_error = exc
        else:
            fallback_error = None
    else:
        fallback_error = None

    details = []
    if top_level_error is not None:
        details.append(f"top-level import failed: {top_level_error}")
    if fallback_error is not None:
        details.append(f"fallback import failed: {fallback_error}")
    detail = "; ".join(details) or f"{name} was not exported by transformers"
    raise ImportError(
        f"Could not import {name} from Hugging Face transformers ({detail}). "
        "Reinstall the supported stack with: "
        "python -m pip install --force-reinstall 'transformers==4.57.3' 'qwen-tts==0.1.1'"
    )


def _normalize_torch_device(device: str) -> str:
    return "cuda:0" if device == "cuda" else device


def _load_qwen3_tts_model(model_path: str, device: str):
    """Load Qwen3TTSModel for one-shot reference prompt building."""
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        model_path,
        low_cpu_mem_usage=False,
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _voice_prompt_cache_path(cfg: "TTSConfig") -> Optional[Path]:
    cache_setting = os.getenv("QWEN_TTS_VOICE_PROMPT_CACHE", "1").strip().lower()
    if cache_setting in {"0", "false", "no", "off"}:
        return None
    cache_dir = Path(
        os.getenv("QWEN_TTS_VOICE_PROMPT_CACHE_DIR", "~/.cache/qwen_megakernel")
    ).expanduser()
    payload = {
        "model_path": cfg.model_path,
        "ref_audio": cfg.ref_audio,
        "ref_text": cfg.ref_text,
        "x_vector_only_mode": cfg.x_vector_only_mode,
        "format": 1,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:24]
    return cache_dir / f"voice_prompt_{digest}.pt"


@dataclass
class TTSConfig:
    """Configuration for the TTS engine."""
    model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    vocoder_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    sample_rate: int = 24000
    chunk_frames: int = 10       # ~0.8 sec per chunk at 12.5 Hz
    # Generation params
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048
    # Code predictor params
    subtalker_do_sample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    # Startup warmup profile: "full" for production latency, "fast" for debugging.
    warmup_profile: str = "full"
    # Optional Base-model voice clone prompt. When set, the megakernel path
    # uses the official qwen-tts prompt builder once, then reuses its tensors.
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only_mode: bool = False


class MegakernelTTSEngine:
    """TTS engine using the megakernel for both talker and code predictor.

    Architecture:
        text → tokenizer → text_embedding + text_projection → prefill
        → megakernel decode loop:
            talker.step_with_embed() → first codebook + hidden state
            code_predictor.predict() → remaining 15 codebooks (megakernel-accelerated)
            sum(all_codec_embeds) + trailing_text → next input
        → vocoder.decode(codec_frames) → audio waveform
    """

    def __init__(self, config: Optional[TTSConfig] = None, device: str = "cuda"):
        self.config = config or TTSConfig()
        self.device = device
        self._initialized = False
        self._init_t0 = None
        self._voice_clone_prompt = None

    def _log(self, message: str):
        """Print an engine progress message with elapsed init time when available."""
        if self._init_t0 is None:
            print(f"[TTS] {message}", flush=True)
            return
        elapsed = time.perf_counter() - self._init_t0
        print(f"[TTS +{elapsed:7.2f}s] {message}", flush=True)

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _timed(self, label: str, fn):
        self._log(f"{label}...")
        t0 = time.perf_counter()
        result = fn()
        self._sync()
        self._log(f"{label} done in {time.perf_counter() - t0:.2f}s")
        return result

    def initialize(self):
        """Load all model components. Call once before generation."""
        if self._initialized:
            return

        cfg = self.config
        self._init_t0 = time.perf_counter()
        self._log("Initializing MegakernelTTSEngine")

        # Load weights
        weights = self._timed(
            f"Loading TTS weights from {cfg.model_path}",
            lambda: load_tts_weights(cfg.model_path, device=self.device, verbose=True),
        )

        # Initialize components (TTSDecoder triggers JIT compilation)
        self.talker = self._timed("Initializing talker decoder / compiling TTS extension", lambda: TTSDecoder(weights=weights))
        self.text_projection = self._timed("Initializing text projection", lambda: TextProjection(weights, device=self.device))
        # Use megakernel-accelerated code predictor (~18x faster than PyTorch)
        self.code_predictor = self._timed("Initializing code predictor kernel wrapper", lambda: CodePredictorKernel(weights, device=self.device))

        # Codec embedding tables (for summing all codebook group embeddings)
        self._talker_embed = weights["embed_weight"]  # [3072, 1024] - group 0
        self._cp_embeds = []  # groups 1-15
        for g in range(NUM_CODE_GROUPS - 1):
            self._cp_embeds.append(
                weights["code_predictor"][f"codec_embedding.{g}.weight"]  # [2048, 1024]
            )
        # Stacked list of all 16 embedding tables (talker first, then cp_embeds).
        # Used in the decode loop for a single enumerated pass with in-place add_.
        self._all_embed_tables = [self._talker_embed] + self._cp_embeds

        # Load tokenizer (text)
        self._log("Loading text tokenizer...")
        AutoTokenizer = _import_transformers_auto("AutoTokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        self._log("Text tokenizer loaded")

        # Load speech tokenizer (vocoder)
        self._timed(f"Loading vocoder from {cfg.vocoder_path}", lambda: self._load_vocoder(cfg.vocoder_path))

        # Vocoder self-test: run a minimal decode to catch hard failures
        # (e.g. version mismatches) before the first real utterance.
        # Use small random codes matching the real call shape [T, NUM_CODE_GROUPS].
        # A test failure is logged but does NOT disable the vocoder — the real
        # warmup synthesis pass below will surface any persistent failure.
        if self.speech_tokenizer is not None:
            try:
                dummy_codes = torch.randint(1, 100, (1, NUM_CODE_GROUPS), dtype=torch.long, device=self.device)
                _ = self.speech_tokenizer.decode([{"audio_codes": dummy_codes}])
                self._log("Vocoder self-test passed")
            except Exception as exc:
                self._log(f"Vocoder self-test warning ({exc}); will verify again during warmup")

        if cfg.ref_audio:
            try:
                self._timed("Building Qwen3-TTS voice clone prompt", self._load_voice_clone_prompt)
            except Exception as exc:
                self._voice_clone_prompt = None
                if _env_bool("QWEN_TTS_REQUIRE_REF_PROMPT", False):
                    raise
                self._log(
                    "Warning: Qwen3-TTS voice clone prompt failed; continuing "
                    f"without reference prompting. Set QWEN_TTS_REQUIRE_REF_PROMPT=true "
                    f"to fail fast. Error: {type(exc).__name__}: {exc}"
                )

        # Precompute constant embeddings (TTS special tokens + role tokens + codec tags)
        self._log("Precomputing constant embeddings...")
        t0 = time.perf_counter()
        with torch.no_grad():
            special_ids = torch.tensor([TTS_PAD, TTS_BOS, TTS_EOS], device=self.device)
            special_embeds = self.text_projection.embed_text_ids(special_ids)
            self._cached_tts_embeds = {
                "pad": special_embeds[0:1],  # [1, 1024]
                "bos": special_embeds[1:2],  # [1, 1024]
                "eos": special_embeds[2:3],  # [1, 1024]
            }
            self._tts_pad_embed = special_embeds[0].to(torch.bfloat16)  # [1024]

            # Precompute role token embeddings (<|im_start|> assistant \n)
            role_text = "<|im_start|>assistant\n"
            role_ids = self.tokenizer.encode(role_text, return_tensors="pt")[0][:3].to(self.device)
            self._cached_role_embeds = self.text_projection.embed_text_ids(role_ids)  # [3, 1024]

            # Precompute codec tag + TTS fused embeddings (official Qwen3-TTS format)
            # Codec: [nothink, think_bos, think_eos, codec_pad, codec_bos]
            codec_ids = torch.tensor([
                CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS,
                CODEC_PAD, CODEC_BOS,
            ], device=self.device)
            codec_embeds = torch.nn.functional.embedding(codec_ids, self._talker_embed)  # [5, 1024]

            # Fuse: [(tts_pad+nothink), (tts_pad+think_bos), (tts_pad+think_eos), (tts_bos+codec_pad)]
            tts_prefix = torch.cat([
                special_embeds[0:1].expand(3, -1),  # pad × 3
                special_embeds[1:2],                  # bos × 1
            ], dim=0)  # [4, 1024]
            self._cached_fused_tags = tts_prefix + codec_embeds[:4]  # [4, 1024]

            # Precompute codec BOS embedding (last in codec sequence)
            self._cached_codec_bos = codec_embeds[4:5]  # [1, 1024]
        self._sync()
        self._log(f"Constant embeddings ready in {time.perf_counter() - t0:.2f}s")

        # Warm up both deterministic and sampling paths. First calls are slow due
        # to CUDA JIT/cublas and PyTorch sampling kernel initialization.
        warmup_profile = self.config.warmup_profile.lower()
        if warmup_profile == "off":
            warmup_modes = []
        elif warmup_profile == "fast":
            warmup_modes = [False]
        else:
            warmup_modes = [False, False, True, True, True]
        self._log(
            "Warming up talker/code predictor "
            f"(profile={warmup_profile}, {len(warmup_modes)} pass(es))"
        )
        for i, do_sample in enumerate(warmup_modes, start=1):
            label = "sampling" if do_sample else "argmax"
            self._log(
                f"Warmup {i}/{len(warmup_modes)} ({label}): "
                "resetting talker and running first decode step..."
            )
            t0 = time.perf_counter()
            self.talker.reset()
            _, h = self.talker.step(CODEC_BOS)
            self._sync()
            self._log(
                f"Warmup {i}/{len(warmup_modes)} ({label}): "
                "running code predictor..."
            )
            self.code_predictor.predict(
                h, 0, self._talker_embed,
                do_sample=do_sample, temperature=0.9, top_k=50,
            )
            self._sync()
            self._log(
                f"Warmup {i}/{len(warmup_modes)} ({label}) done "
                f"in {time.perf_counter() - t0:.2f}s"
            )
        self.talker.reset()

        # Run a full synthesis warmup using the actual voice-clone prefill path
        # so that prefill_parallel, cuBLAS GEMMs, and the vocoder are all hot
        # before the first real utterance. Without this, the first call pays
        # ~6s of cold-start cost (CUDA kernel JIT, cuBLAS plan caching, HBM
        # cold pages) even though the talker/CP decode steps above are warm.
        if warmup_profile != "off":
            self._log("Warming up full synthesis path (voice-clone prefill + vocoder)...")
            t0 = time.perf_counter()
            try:
                warmup_text = "Hello."
                # Consume the full generator to drive all GPU work
                for _ in self._generate_codec_frames(warmup_text):
                    pass
                self.talker.reset()
                self._sync()
                self._log(f"Full synthesis warmup done in {time.perf_counter() - t0:.2f}s")
            except Exception as exc:
                self._log(f"Full synthesis warmup failed (non-fatal): {exc}")
                self.talker.reset()
        if self.speech_tokenizer is not None:
            if warmup_profile == "off":
                vocoder_warmup_frames = []
            elif warmup_profile == "fast":
                vocoder_warmup_frames = []
            else:
                vocoder_warmup_frames = [5]
            self._log(
                "Warming up vocoder "
                f"(profile={warmup_profile}, {len(vocoder_warmup_frames)} pass(es): "
                f"{vocoder_warmup_frames} codec frame batches)"
            )
            for i, n in enumerate(vocoder_warmup_frames, start=1):
                self._log(
                    f"Vocoder warmup {i}/{len(vocoder_warmup_frames)} "
                    f"({n} frame(s)): creating dummy codec codes..."
                )
                t0 = time.perf_counter()
                dummy_codes = torch.randint(0, 2048, (n, NUM_CODE_GROUPS), dtype=torch.long, device=self.device)
                self._log(
                    f"Vocoder warmup {i}/{len(vocoder_warmup_frames)} "
                    "decoding dummy audio..."
                )
                try:
                    self.speech_tokenizer.decode([{"audio_codes": dummy_codes}])
                    self._sync()
                    self._log(
                        f"Vocoder warmup {i}/{len(vocoder_warmup_frames)} "
                        f"done in {time.perf_counter() - t0:.2f}s"
                    )
                except Exception as exc:
                    self._log(
                        "Vocoder warmup decode failed; marking vocoder unavailable "
                        f"to avoid runtime crashes: {type(exc).__name__}: {exc}"
                    )
                    self.speech_tokenizer = None
                    self.sample_rate = self.config.sample_rate
                    break
        else:
            self._log("Skipping vocoder warmup because vocoder is unavailable")
        self._sync()

        self._initialized = True
        self._log("MegakernelTTSEngine initialized")

    def _load_voice_clone_prompt(self):
        """Build and cache official Qwen3-TTS Base voice-clone prompt tensors."""
        cfg = self.config
        if not cfg.ref_audio:
            self._voice_clone_prompt = None
            return
        if not cfg.x_vector_only_mode and not cfg.ref_text:
            raise ValueError(
                "QWEN_TTS_REF_TEXT is required for megakernel voice-clone ICL mode. "
                "Set QWEN_TTS_X_VECTOR_ONLY=true to use only the speaker embedding."
            )

        import torch
        cache_path = _voice_prompt_cache_path(cfg)
        if cache_path and cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location=self.device)
                self._voice_clone_prompt = {
                    "ref_code": None
                    if cached.get("ref_code") is None
                    else cached["ref_code"].to(self.device).long(),
                    "ref_spk_embedding": cached["ref_spk_embedding"].to(self.device).to(torch.bfloat16),
                    "x_vector_only_mode": bool(cached["x_vector_only_mode"]),
                    "icl_mode": bool(cached["icl_mode"]),
                    "ref_text": cached.get("ref_text"),
                }
                self._log(f"Loaded Qwen3-TTS voice clone prompt cache from {cache_path}")
                return
            except Exception as exc:
                self._log(f"Ignoring unreadable voice clone prompt cache {cache_path}: {exc}")

        model = _load_qwen3_tts_model(cfg.model_path, self.device)
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=cfg.ref_audio,
            ref_text=cfg.ref_text,
            x_vector_only_mode=cfg.x_vector_only_mode,
        )
        if not prompt_items:
            raise ValueError("Qwen3-TTS voice clone prompt builder returned no prompt items")

        item = prompt_items[0]
        ref_code = None if item.ref_code is None else item.ref_code.to(self.device).long()
        ref_spk_embedding = item.ref_spk_embedding.to(self.device).to(torch.bfloat16)
        self._voice_clone_prompt = {
            "ref_code": ref_code,
            "ref_spk_embedding": ref_spk_embedding,
            "x_vector_only_mode": bool(item.x_vector_only_mode),
            "icl_mode": bool(item.icl_mode),
            "ref_text": item.ref_text,
        }

        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "ref_code": None if ref_code is None else ref_code.detach().cpu(),
                        "ref_spk_embedding": ref_spk_embedding.detach().cpu(),
                        "x_vector_only_mode": bool(item.x_vector_only_mode),
                        "icl_mode": bool(item.icl_mode),
                        "ref_text": item.ref_text,
                    },
                    cache_path,
                )
                self._log(f"Saved Qwen3-TTS voice clone prompt cache to {cache_path}")
            except Exception as exc:
                self._log(f"Could not save voice clone prompt cache {cache_path}: {exc}")

        del model
        torch.cuda.empty_cache()

    def _load_vocoder(self, vocoder_path: str):
        """Load the speech tokenizer for codec → waveform decoding."""
        target_device = _normalize_torch_device(self.device)
        try:
            from qwen_tts import Qwen3TTSTokenizer

            self.speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
                vocoder_path,
                device_map=target_device,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            self.sample_rate = self.speech_tokenizer.get_output_sample_rate()
            self._log(f"Vocoder loaded via Qwen3TTSTokenizer (sample rate: {self.sample_rate} Hz)")
            return
        except Exception as e:
            self._log(f"Official vocoder load failed, trying manual loader: {e}")

        # Fallback: load the speech tokenizer model directly from the
        # speech_tokenizer/ subfolder, bypassing AutoFeatureExtractor.
        try:
            # Monkey-patch transformers if needed (qwen_tts compat with transformers 5.x)
            import transformers.utils.generic
            if not hasattr(transformers.utils.generic, 'check_model_inputs'):
                def _check_model_inputs(*args, **kwargs):
                    def decorator(func):
                        return func
                    return decorator
                transformers.utils.generic.check_model_inputs = _check_model_inputs

            AutoConfig = _import_transformers_auto("AutoConfig")
            AutoModel = _import_transformers_auto("AutoModel")
            from qwen_tts.core import (
                Qwen3TTSTokenizerV2Config,
                Qwen3TTSTokenizerV2Model,
            )

            # Register speech tokenizer model type
            try:
                AutoConfig.register('qwen3_tts_tokenizer_12hz', Qwen3TTSTokenizerV2Config)
                AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
            except ValueError:
                pass  # Already registered

            # Load speech tokenizer from subfolder
            model = AutoModel.from_pretrained(
                vocoder_path,
                subfolder='speech_tokenizer',
                dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            )
            try:
                model = model.to(target_device)
            except NotImplementedError as exc:
                self._log(
                    f"Could not move vocoder to {target_device}; continuing on "
                    f"loader default device: {exc}"
                )

            # Create Qwen3TTSTokenizer wrapper (skip feature_extractor — not needed for decode)
            from qwen_tts import Qwen3TTSTokenizer
            self.speech_tokenizer = Qwen3TTSTokenizer()
            self.speech_tokenizer.model = model
            self.speech_tokenizer.feature_extractor = None
            self.speech_tokenizer.config = model.config
            self.speech_tokenizer.device = next(model.parameters()).device
            self.sample_rate = self.speech_tokenizer.get_output_sample_rate()
            self._log(f"Vocoder loaded (sample rate: {self.sample_rate} Hz)")
            return
        except Exception as e:
            self._log(f"Vocoder load failed: {e}")

        self.speech_tokenizer = None
        self.sample_rate = self.config.sample_rate
        self._log("Warning: Vocoder unavailable. Audio output will be silence")

    @torch.no_grad()
    def synthesize(self, text: str, ref_audio: Optional[np.ndarray] = None) -> tuple[np.ndarray, int]:
        """Non-streaming synthesis. Returns (waveform, sample_rate)."""
        self.initialize()
        codec_frames = list(self._generate_codec_frames(text))
        if not codec_frames:
            return np.array([], dtype=np.float32), self.sample_rate
        return self._decode_to_audio(codec_frames)

    async def synthesize_streaming(
        self,
        text: str,
        chunk_frames: Optional[int] = None,
    ) -> AsyncGenerator[tuple[np.ndarray, int], None]:
        """Streaming synthesis. Yields (audio_chunk, sample_rate) as frames accumulate."""
        self.initialize()
        chunk_size = chunk_frames or self.config.chunk_frames
        buffer = []
        first_chunk = True
        chunk_idx = 0
        self._log(f"Starting streaming synthesis: chunk_frames={chunk_size}, text_len={len(text)}")

        for frame in self._generate_codec_frames(text):
            buffer.append(frame)
            # Use smaller first chunk (1 frame) for fast TTFC, then normal chunk size
            target = 1 if first_chunk else chunk_size
            if len(buffer) >= target:
                chunk_idx += 1
                self._log(f"Decoding audio chunk {chunk_idx} from {len(buffer)} codec frame(s)...")
                t0 = time.perf_counter()
                audio, sr = self._decode_to_audio(buffer)
                self._log(f"Audio chunk {chunk_idx} decoded in {time.perf_counter() - t0:.2f}s ({len(audio)} samples)")
                buffer = []
                first_chunk = False
                yield audio, sr
                await asyncio.sleep(0)

        if buffer:
            chunk_idx += 1
            self._log(f"Decoding final audio chunk {chunk_idx} from {len(buffer)} codec frame(s)...")
            t0 = time.perf_counter()
            audio, sr = self._decode_to_audio(buffer)
            self._log(f"Final audio chunk {chunk_idx} decoded in {time.perf_counter() - t0:.2f}s ({len(audio)} samples)")
            yield audio, sr

    def _generate_codec_frames(self, text: str) -> Generator[torch.Tensor, None, None]:
        """Run the talker + code predictor to generate codec frames.

        Each frame is a tensor of shape [NUM_CODE_GROUPS] (int64).
        Yields frames one at a time for streaming support.
        """
        cfg = self.config
        self.talker.reset()
        self._log("Preparing codec frame generator")
        t_gen_start = time.perf_counter()

        # Tokenize only the content text (role tokens are precomputed)
        # Format: <|im_start|>assistant\n TEXT <|im_end|>\n<|im_start|>assistant\n
        # Tokens: [role(3)] [text...] [<|im_end|>(1) \n(1) <|im_start|>(1) assistant(1) \n(1)]
        formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = self.tokenizer.encode(formatted_text, return_tensors="pt")[0]
        content_ids = text_ids[3:].to(self.device)
        self._log(f"Tokenized text: total_tokens={text_ids.numel()}, content_tokens={content_ids.numel()}")

        # Embed content tokens (single batched call — role/tags/special are precomputed)
        t0 = time.perf_counter()
        content_embeds = self.text_projection.embed_text_ids(content_ids)
        self._sync()
        self._log(f"Projected text embeddings in {time.perf_counter() - t0:.2f}s")
        first_text_with_bos = content_embeds[:1] + self._cached_codec_bos

        voice_prompt = self._voice_clone_prompt
        if voice_prompt is not None:
            prefill_embeds, trailing_text = self._build_voice_clone_prefill(
                text_ids=text_ids,
                content_embeds=content_embeds,
                first_text_with_bos=first_text_with_bos,
                voice_prompt=voice_prompt,
            )
        else:
            # Build prefill: [role(3), fused_tags(4), first_text+bos(1)] = 8 steps
            prefill_embeds = torch.cat([
                self._cached_role_embeds,
                self._cached_fused_tags,
                first_text_with_bos,
            ], dim=0)  # [8, 1024]

            # Trailing text: content tokens[1:-5] + tts_eos
            # Strip last 5 format tokens: <|im_end|>\n<|im_start|>assistant\n
            trailing_text = torch.cat([
                content_embeds[1:-5],
                self._cached_tts_embeds["eos"],
            ], dim=0)

        # Phase 1: Prefill — feed all prefill embeddings through the talker.
        # Parallel path runs the full 28-layer forward over N positions in one
        # pass via cuBLAS GEMMs + flash-attention; sequential path queues N
        # megakernel decode launches with a single end-of-loop sync. Toggle
        # with QWEN_TTS_PARALLEL_PREFILL=0 if numerical drift is suspected.
        use_parallel = _env_bool("QWEN_TTS_PARALLEL_PREFILL", True)
        mode = "parallel" if use_parallel else "sequential"
        self._log(f"Running talker prefill ({prefill_embeds.shape[0]} step(s), {mode})...")
        t0 = time.perf_counter()
        if use_parallel:
            first_token, hidden = self.talker.prefill_parallel(prefill_embeds)
        else:
            first_token, hidden = self.talker.prefill_with_embeds(prefill_embeds)
        self._log(f"Talker prefill done in {time.perf_counter() - t0:.2f}s")

        # Phase 2: Autoregressive decode
        trailing_idx = 0
        tts_pad_embed = self._tts_pad_embed

        if first_token is None or hidden is None:
            return
        if first_token == CODEC_EOS:
            self._log(
                "Talker prefill produced CODEC_EOS; falling back to legacy "
                "CODEC_BOS decode step"
            )
            first_token, hidden = self.talker.step(CODEC_BOS)
            self._sync()
        self._log(f"Talker prefill produced first token={first_token}")

        prev_token = first_token

        # Estimate max frames from text length
        # English speech: ~2.5 words/sec. At 12.5 codec frames/sec,
        # each word ≈ 5 frames. Use 2x margin since EOS is unreliable.
        word_count = max(len(text.split()), 1)
        estimated_speech_sec = word_count / 2.5
        max_frames = max(int(estimated_speech_sec * 12.5 * 2.0), 25)
        max_frames = min(max_frames, cfg.max_new_tokens)
        self._log(f"Generating up to {max_frames} codec frame(s) for {word_count} word(s)")

        for step in range(max_frames):
            if prev_token == CODEC_EOS:
                self._log(f"Stopping at frame {step}: talker emitted CODEC_EOS")
                break

            # Run code predictor (megakernel-accelerated)
            if step < 3 or (step + 1) % 10 == 0:
                self._log(f"Generating codec frame {step + 1}/{max_frames}...")
            frame_t0 = time.perf_counter()
            all_codes = self.code_predictor.predict(
                talker_hidden=hidden,
                first_codebook_token=prev_token,
                talker_embed_weight=self._talker_embed,
                do_sample=cfg.subtalker_do_sample,
                temperature=cfg.subtalker_temperature,
                top_k=cfg.subtalker_top_k,
            )  # [NUM_CODE_GROUPS] int64
            self._sync()
            if step < 3 or (step + 1) % 10 == 0:
                self._log(f"Codec frame {step + 1}/{max_frames} ready in {time.perf_counter() - frame_t0:.2f}s")

            yield all_codes

            # Compute next input: sum of all codec group embeddings.
            # Use in-place add_ to avoid 15 intermediate tensor allocations.
            embed_sum = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device=self.device)
            for g, table in enumerate(self._all_embed_tables):
                embed_sum.add_(torch.nn.functional.embedding(all_codes[g:g + 1], table).squeeze(0))

            # Add trailing text embedding
            if trailing_idx < trailing_text.shape[0]:
                embed_sum = embed_sum + trailing_text[trailing_idx].to(torch.bfloat16)
                trailing_idx += 1
            else:
                if trailing_idx == trailing_text.shape[0]:
                    self._log(f"Trailing text exhausted at frame {step}; padding remaining frames")
                    trailing_idx += 1  # advance so this only logs once
                embed_sum = embed_sum + tts_pad_embed

            prev_token, hidden = self.talker.step_with_embed(embed_sum)
        self._log(f"Codec frame generation finished in {time.perf_counter() - t_gen_start:.2f}s")

    def _build_voice_clone_prefill(
        self,
        *,
        text_ids: torch.Tensor,
        content_embeds: torch.Tensor,
        first_text_with_bos: torch.Tensor,
        voice_prompt: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the official Base-model speaker/ICL prefill for megakernel decode."""
        speaker_embed = voice_prompt["ref_spk_embedding"].view(1, -1)
        if speaker_embed.shape[-1] != HIDDEN_SIZE:
            raise ValueError(
                f"Unexpected speaker embedding size {speaker_embed.shape[-1]}; "
                f"expected {HIDDEN_SIZE}"
            )

        codec_ids = torch.tensor([
            CODEC_NOTHINK,
            CODEC_THINK_BOS,
            CODEC_THINK_EOS,
            CODEC_PAD,
            CODEC_BOS,
        ], device=self.device)
        codec_embeds = torch.nn.functional.embedding(codec_ids, self._talker_embed)
        codec_input_embedding = torch.cat([
            codec_embeds[0:3],
            speaker_embed.to(torch.bfloat16),
            codec_embeds[3:5],
        ], dim=0)

        # Official prompt: tts_pad for tag/speaker tokens, then tts_bos for
        # codec_pad. The final codec_bos is fused with target/ICL text below.
        text_side = torch.cat([
            self._cached_tts_embeds["pad"].expand(codec_input_embedding.shape[0] - 2, -1),
            self._cached_tts_embeds["bos"],
        ], dim=0)
        prompt_prefix = text_side + codec_input_embedding[:-1]
        prefill_prefix = torch.cat([self._cached_role_embeds, prompt_prefix], dim=0)

        ref_code = voice_prompt.get("ref_code")
        if ref_code is not None and voice_prompt.get("icl_mode"):
            ref_text = voice_prompt.get("ref_text")
            ref_formatted = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
            ref_ids = self.tokenizer.encode(ref_formatted, return_tensors="pt")[0].to(self.device)

            target_text_ids = text_ids[3:-5].to(self.device)
            ref_text_ids = ref_ids[3:-2].to(self.device)
            icl_text_ids = torch.cat([ref_text_ids, target_text_ids], dim=0)
            icl_text_embed = self.text_projection.embed_text_ids(icl_text_ids)
            icl_text_embed = torch.cat([icl_text_embed, self._cached_tts_embeds["eos"]], dim=0)

            ref_code = ref_code.to(self.device).long()
            ref_codec_embed = self._codec_frame_embeds(ref_code)
            codec_bos = codec_embeds[4:5]
            codec_embed = torch.cat([codec_bos, ref_codec_embed], dim=0)

            if icl_text_embed.shape[0] > codec_embed.shape[0]:
                icl_input_embed = icl_text_embed[:codec_embed.shape[0]] + codec_embed
                trailing_text = icl_text_embed[codec_embed.shape[0]:]
            else:
                pad_count = codec_embed.shape[0] - icl_text_embed.shape[0]
                if pad_count:
                    icl_text_embed = torch.cat([
                        icl_text_embed,
                        self._cached_tts_embeds["pad"].expand(pad_count, -1),
                    ], dim=0)
                icl_input_embed = icl_text_embed + codec_embed
                trailing_text = self._cached_tts_embeds["pad"]

            return torch.cat([prefill_prefix, icl_input_embed], dim=0), trailing_text

        trailing_text = torch.cat([
            content_embeds[1:-5],
            self._cached_tts_embeds["eos"],
        ], dim=0)
        return torch.cat([prefill_prefix, first_text_with_bos], dim=0), trailing_text

    def _codec_frame_embeds(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum group embeddings for a sequence of full codec frames."""
        embeds = torch.nn.functional.embedding(codes[:, 0], self._talker_embed)
        for g in range(NUM_CODE_GROUPS - 1):
            embeds = embeds + torch.nn.functional.embedding(codes[:, g + 1], self._cp_embeds[g])
        return embeds

    def _decode_to_audio(self, codec_frames: list[torch.Tensor]) -> tuple[np.ndarray, int]:
        """Decode codec frames to audio waveform."""
        if not codec_frames:
            return np.array([], dtype=np.float32), self.sample_rate

        audio_codes = torch.stack(codec_frames, dim=0)

        if self.speech_tokenizer is not None:
            wavs, sr = self.speech_tokenizer.decode([{"audio_codes": audio_codes}])
            wav = wavs[0]
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().float().cpu().numpy()
            return np.asarray(wav, dtype=np.float32), sr
        raise RuntimeError(
            "Qwen3-TTS vocoder is unavailable (speech_tokenizer is None). "
            "This means _load_vocoder() failed at startup — look for "
            "'Vocoder load failed' in the engine init log above. "
            "Common fix: pip install --upgrade qwen-tts transformers"
        )

    def get_metrics(self) -> dict:
        """Return performance metrics from the last generation."""
        return {
            "sample_rate": self.sample_rate,
            "position": self.talker.position if self._initialized else 0,
        }
