#!/usr/bin/env bash
# freeimage-local installer
# Builds stable-diffusion.cpp for your machine and downloads Z-Image Turbo (int8).
#
#   ./install.sh            build + download everything
#   ./install.sh --build    only (re)build the engine
#   ./install.sh --models   only download / verify the models
#
# Set HF_TOKEN to authenticate Hugging Face downloads (rate limits / gated repos):
#   HF_TOKEN=hf_xxx ./install.sh --models
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common.sh
source "$ROOT/scripts/common.sh"

# stable-diffusion.cpp version this repo is tested against. Override with SDCPP_REF=master.
SDCPP_REPO="https://github.com/leejet/stable-diffusion.cpp.git"
SDCPP_REF="${SDCPP_REF:-532d2c15c430a785a712b2d94281f04e733008a4}"

# name | url | sha256 | bytes
MODELS_COMMON=(
  "diffusion|https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/main/split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors|be517ebd47c912a5626a588e1aeea43e6be4a43c0cdcd2b48a2a780d9f358635|6201001296"
  "vae|https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/main/split_files/vae/ae.safetensors|afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38|335304388"
)
# CUDA (and Metal 24 GB+) keep the Q8 text encoder. Metal ≤18 GB uses Q4 so DiT fits.
MODEL_TE_Q8="text-encoder|https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf|eed555233267a33c7e8ee31682762cc7751b3f6d224039086e0e846f05fffa5d|4280405792"
MODEL_TE_Q4="text-encoder|https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf|f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a|2497281312"

model_list() {
  detect_platform
  MODELS=("${MODELS_COMMON[@]}")
  if [ "$FI_PLATFORM" = metal ] && [ "${FI_UNIFIED_GB:-0}" -gt 0 ] && [ "$FI_UNIFIED_GB" -le 18 ]; then
    MODELS+=("$MODEL_TE_Q4")
  else
    MODELS+=("$MODEL_TE_Q8")
  fi
}

need() { command -v "$1" >/dev/null 2>&1 || die "missing '$1'. $2"; }

build_engine() {
  step "Checking your machine"
  detect_platform
  info "platform: $FI_PLATFORM ($FI_DEVICE)"

  need git "Install git first."
  need cmake "macOS: brew install cmake   ·   Ubuntu: sudo apt install cmake"
  local gen=()
  command -v ninja >/dev/null 2>&1 && gen=(-G Ninja)

  local flags=(-DCMAKE_BUILD_TYPE=Release)
  case "$FI_PLATFORM" in
    metal)
      flags+=(-DSD_METAL=ON)
      ;;
    cuda)
      need nvcc "Install the CUDA toolkit (12.8+ for RTX 50-series): https://developer.nvidia.com/cuda-downloads"
      local cuda_ver; cuda_ver="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
      info "CUDA toolkit: $cuda_ver, GPU architecture: sm_$FI_CUDA_ARCH"
      if [ "$FI_CUDA_ARCH" -ge 120 ] && ! version_ge "$cuda_ver" 12.8; then
        die "Blackwell (sm_$FI_CUDA_ARCH, e.g. RTX 50-series / GB10) needs CUDA 12.8 or newer; you have $cuda_ver."
      fi
      # CUDA 13 dropped offline compile for pre-Turing (sm < 75).
      if version_ge "$cuda_ver" 13.0 && [ "$FI_CUDA_ARCH" -lt 75 ]; then
        die "CUDA $cuda_ver cannot target sm_$FI_CUDA_ARCH (needs sm_75+ / Turing or newer). Use CUDA 12.x or a newer GPU."
      fi
      flags+=(-DSD_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$FI_CUDA_ARCH")
      # Older CUDA releases reject the newest gcc. Use the newest supported g++ if one is installed.
      # CUDA 13 commonly pairs with gcc 15; try that first, then fall back.
      for v in 15 14 13 12; do
        if command -v "g++-$v" >/dev/null 2>&1; then
          flags+=("-DCMAKE_CUDA_HOST_COMPILER=$(command -v "g++-$v")"); break
        fi
      done
      ;;
    *)
      die "No supported GPU found. freeimage-local needs an Apple Silicon Mac or an NVIDIA GPU with CUDA (Linux / WSL2)."
      ;;
  esac

  step "Getting stable-diffusion.cpp ($SDCPP_REF)"
  local src="$FI_HOME/src/stable-diffusion.cpp"
  if [ ! -d "$src/.git" ]; then
    git clone --quiet "$SDCPP_REPO" "$src"
  fi
  git -C "$src" fetch --quiet origin
  git -C "$src" checkout --quiet "$SDCPP_REF" 2>/dev/null || git -C "$src" checkout --quiet "origin/$SDCPP_REF"
  git -C "$src" submodule update --init --recursive --quiet
  info "at $(git -C "$src" rev-parse --short HEAD)"

  local jobs; jobs="$(cpu_count)"
  step "Configuring cmake"
  info "log: $FI_HOME/build.log"
  cmake -S "$src" -B "$src/build" "${gen[@]}" "${flags[@]}" 2>&1 | tee "$FI_HOME/build.log" \
    || { tail -30 "$FI_HOME/build.log"; die "cmake configure failed. Full log: $FI_HOME/build.log"; }

  if [ "$FI_PLATFORM" = cuda ]; then
    step "Building engine ($jobs jobs — CUDA often 15–40 min; live progress below)"
  else
    step "Building engine ($jobs jobs)"
  fi
  # Stream compile lines to the terminal and keep the full log for failures.
  cmake --build "$src/build" --config Release -j "$jobs" --target sd-cli sd-server 2>&1 | tee -a "$FI_HOME/build.log" \
    || { tail -30 "$FI_HOME/build.log"; die "build failed. Full log: $FI_HOME/build.log"; }
  mkdir -p "$FI_HOME/bin"
  cp "$src/build/bin/sd-cli" "$src/build/bin/sd-server" "$FI_HOME/bin/"
  printf '%s\n' "$FI_PLATFORM" >"$FI_HOME/platform"
  ok "engine ready: $FI_HOME/bin"
}

download_models() {
  model_list
  local total=0 entry
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r _ _ _ bytes <<<"$entry"
    total=$(( total + bytes ))
  done
  step "Downloading models ($(human "$total"), resumable)"
  need curl "Install curl first."
  mkdir -p "$FI_HOME/models"
  local auth=()
  if [ -n "${HF_TOKEN:-}" ]; then
    auth=(-H "Authorization: Bearer ${HF_TOKEN}")
    info "using HF_TOKEN for Hugging Face downloads"
  fi
  local name url sha bytes file
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r name url sha bytes <<<"$entry"
    file="$FI_HOME/models/$(basename "$url")"
    if [ -f "$file" ] && [ "$(file_size "$file")" = "$bytes" ]; then
      info "$name: already downloaded"
    else
      info "$name: $(basename "$url") ($(human "$bytes"))"
      curl -fL --retry 5 --retry-delay 3 -C - --progress-bar "${auth[@]+"${auth[@]}"}" -o "$file" "$url" \
        || die "download failed for $name. Run ./install.sh --models again to resume."
    fi
    if [ "${FI_SKIP_VERIFY:-0}" != 1 ]; then
      printf '  verifying… '
      [ "$(sha256 "$file")" = "$sha" ] || { echo; die "$name checksum mismatch. Delete $file and run ./install.sh --models"; }
      echo ok
    fi
  done
  ok "models ready: $FI_HOME/models"
}

case "${1:-all}" in
  --build)  build_engine ;;
  --models) download_models ;;
  all|"")   build_engine; download_models ;;
  -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "unknown option: $1 (try --help)" ;;
esac

if [ "${1:-all}" = all ]; then
  echo
  ok "All set. Try:"
  echo "    ./fi \"a red fox in fresh snow, golden hour\""
  echo "    ./fi serve        # web UI on http://localhost:7861"
fi
