#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip packaging wheel ninja

INFO_JSON="$(
python - <<'PY'
import json
import platform
import sys

import torch

py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_version = torch.__version__.split("+")[0]
torch_mm = ".".join(torch_version.split(".")[:2])
cuda = torch.version.cuda or ""
cuda_digits = cuda.replace(".", "")
cuda_major = cuda.split(".")[0] if cuda else ""
abi = "TRUE" if torch.compiled_with_cxx11_abi() else "FALSE"

machine = platform.machine().lower()
if sys.platform.startswith("linux"):
    plat = "linux_x86_64" if machine in ("x86_64", "amd64") else f"linux_{machine}"
elif sys.platform.startswith("win"):
    plat = "win_amd64"
else:
    raise SystemExit(f"Unsupported platform for flash-attn wheel lookup: {sys.platform}")

print(json.dumps({
    "py_tag": py_tag,
    "torch_version": torch_version,
    "torch_mm": torch_mm,
    "cuda": cuda,
    "cuda_digits": cuda_digits,
    "cuda_major": cuda_major,
    "abi": abi,
    "plat": plat,
}))
PY
)"

echo "Detected environment:"
echo "$INFO_JSON" | python -m json.tool

WHEEL_URL="$(
INFO_JSON="$INFO_JSON" python - <<'PY'
import json
import os
import re
import sys
import urllib.request

info = json.loads(os.environ["INFO_JSON"])
headers = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "flash-attn-wheel-installer",
}
token = os.getenv("GITHUB_TOKEN")
if token:
    headers["Authorization"] = f"Bearer {token}"

assets = []
for page in range(1, 6):
    req = urllib.request.Request(
        f"https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=100&page={page}",
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        releases = json.load(resp)
    if not releases:
        break
    for rel in releases:
        for asset in rel.get("assets", []):
            name = asset.get("name", "")
            if name.endswith(".whl") and name.startswith("flash_attn-"):
                assets.append((rel.get("tag_name", ""), name, asset.get("browser_download_url", "")))

py_tag = info["py_tag"]
plat = info["plat"]
torch_mm = info["torch_mm"]
torch_version = info["torch_version"]
abi = info["abi"]
cuda_digits = info["cuda_digits"]
cuda_major = info["cuda_major"]

cuda_patterns = []
if cuda_digits:
    cuda_patterns.append(f"cu{cuda_digits}")
if cuda_major:
    cuda_patterns.append(f"cu{cuda_major}")
    cuda_patterns.append(f"cu{cuda_major}0")
cuda_patterns = list(dict.fromkeys(cuda_patterns))

def score(name):
    if py_tag not in name or plat not in name:
        return -1
    if not any(cu in name for cu in cuda_patterns):
        return -1
    if f"torch{torch_version}" in name:
        score_value = 180
    elif f"torch{torch_mm}" in name:
        score_value = 160
    else:
        return -1
    if f"cxx11abi{abi}" in name:
        score_value += 20
    elif "cxx11abi" not in name:
        score_value += 5
    version_match = re.search(r"flash_attn-([0-9][^+]+)", name)
    if version_match:
        parts = re.findall(r"\d+", version_match.group(1))
        score_value += sum(int(part) for part in parts[:3])
    return score_value

matches = sorted(
    [(score(name), tag, name, url) for tag, name, url in assets if score(name) >= 0],
    reverse=True,
)

if not matches:
    print("NO_MATCH")
    sys.exit(0)

_, tag, name, url = matches[0]
print(url)
print(f"Selected {tag}: {name}", file=sys.stderr)
PY
)"

if [[ "$WHEEL_URL" == "NO_MATCH" || -z "$WHEEL_URL" ]]; then
  WHEEL_URL="$(
  INFO_JSON="$INFO_JSON" python - <<'PY'
import json
import os

info = json.loads(os.environ["INFO_JSON"])
allow_community = os.getenv("ALLOW_COMMUNITY_FLASH_ATTN_WHEELS", "1") != "0"

# Official FlashAttention releases can lag new CUDA/PyTorch stacks. This
# community wheel is linked from Dao-AILab/flash-attention issue #2442 and
# covers the common CUDA 13 / Torch 2.11 / Python 3.12 Linux stack.
community_wheels = [
    {
        "cuda_major": "13",
        "torch_mm": "2.11",
        "py_tag": "cp312",
        "plat": "linux_x86_64",
        "abi": "TRUE",
        "url": (
            "https://github.com/adithyaxx/flash-attention/releases/download/v2.8.3/"
            "flash_attn-2.8.3%2Bcu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
        ),
    },
]

if allow_community:
    for wheel in community_wheels:
        if all(str(info.get(key)) == value for key, value in wheel.items() if key != "url"):
            print(wheel["url"])
            break
    else:
        print("NO_MATCH")
else:
    print("NO_MATCH")
PY
  )"
  if [[ "$WHEEL_URL" != "NO_MATCH" && -n "$WHEEL_URL" ]]; then
    echo "Using matching community prebuilt wheel."
    echo "Set ALLOW_COMMUNITY_FLASH_ATTN_WHEELS=0 to disable this fallback."
  fi
fi

if [[ "$WHEEL_URL" == "NO_MATCH" || -z "$WHEEL_URL" ]]; then
  echo "No exact official prebuilt wheel found. Falling back to faster source build."
  echo "If RAM is limited, keep MAX_JOBS low."
  MAX_JOBS="${MAX_JOBS:-4}" python -m pip install flash-attn --no-build-isolation
else
  echo "Downloading wheel:"
  echo "$WHEEL_URL"
  WHEEL_PATH="/tmp/$(basename "$WHEEL_URL")"
  curl -L --fail --retry 3 -o "$WHEEL_PATH" "$WHEEL_URL"
  python -m pip install --force-reinstall "$WHEEL_PATH"
fi

python - <<'PY'
import flash_attn

print("flash_attn ok", getattr(flash_attn, "__version__", "unknown"))
PY
