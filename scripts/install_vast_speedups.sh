#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Install optional Vast.ai/container speedups for Qwen TTS megakernel.

Usage:
  scripts/install_vast_speedups.sh [options]

Options:
  --skip-flash-attn      Do not install flash-attn.
  --max-jobs N           Limit flash-attn build jobs. Default: 4.
  --preload-models       Download the default Qwen text and TTS models.
  --help                 Show this help.

Environment:
  HF_HOME                Hugging Face cache directory.
  HF_HUB_ENABLE_HF_TRANSFER=1 is exported by this script for this process.
EOF
}

log() {
  printf '\n==> %s\n' "$*"
}

warn() {
  printf '\nWARNING: %s\n' "$*" >&2
}

SKIP_FLASH_ATTN=0
PRELOAD_MODELS=0
MAX_JOBS="${MAX_JOBS:-4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-flash-attn)
      SKIP_FLASH_ATTN=1
      shift
      ;;
    --max-jobs)
      if [[ $# -lt 2 ]]; then
        warn "--max-jobs requires a value"
        exit 2
      fi
      MAX_JOBS="$2"
      shift 2
      ;;
    --preload-models)
      PRELOAD_MODELS=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      warn "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

log "Python and CUDA environment"
python - <<'PY'
import shutil
import sys

print(f"python={sys.version.split()[0]}")
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name()}")
except Exception as exc:
    print(f"torch_check_error={type(exc).__name__}: {exc}")

print(f"nvcc={shutil.which('nvcc') or 'missing'}")
PY

log "Installing Python packaging helpers"
python -m pip install -U pip setuptools wheel packaging ninja

log "Installing Hugging Face transfer speedup"
python -m pip install -U huggingface_hub hf_transfer

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

if [[ "$SKIP_FLASH_ATTN" -eq 0 ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    warn "nvcc was not found. flash-attn may fail unless a compatible wheel exists."
    warn "Use a CUDA devel image for reliable source builds, or rerun with --skip-flash-attn."
  fi

  log "Installing flash-attn with MAX_JOBS=${MAX_JOBS}"
  MAX_JOBS="$MAX_JOBS" python -m pip install -U flash-attn --no-build-isolation
else
  log "Skipping flash-attn install"
fi

if [[ "$PRELOAD_MODELS" -eq 1 ]]; then
  log "Preloading Qwen models into Hugging Face cache"
  huggingface-cli download Qwen/Qwen3-0.6B
  huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base
fi

log "Installed package versions"
python - <<'PY'
import importlib.metadata as md

for name in ["huggingface_hub", "hf_transfer", "flash_attn"]:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=not installed")
PY

cat <<EOF

Done.

For this shell, these Hugging Face settings were exported:
  HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}
  HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}
  HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}

To keep them for future shells, add them to your shell profile or container env:
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export HF_HUB_DOWNLOAD_TIMEOUT=60
  export HF_HUB_ETAG_TIMEOUT=60
EOF
