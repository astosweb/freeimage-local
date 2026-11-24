# shellcheck shell=bash disable=SC2034
# Shared helpers for install.sh and fi. Sourced, not executed.

FI_HOME="${FI_HOME:-$ROOT/.fi}"
mkdir -p "$FI_HOME"

if [ -t 1 ]; then B=$'\033[1m'; D=$'\033[2m'; G=$'\033[32m'; R=$'\033[31m'; N=$'\033[0m'; else B=; D=; G=; R=; N=; fi
step() { printf '\n%s▸ %s%s\n' "$B" "$1" "$N"; }
info() { printf '  %s%s%s\n' "$D" "$1" "$N"; }
ok()   { printf '%s✓%s %s\n' "$G" "$N" "$1"; }
die()  { printf '%s✗ %s%s\n' "$R" "$1" "$N" >&2; exit 1; }

cpu_count() { getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; }
file_size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }
sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
human() { awk -v b="$1" 'BEGIN { split("B KB MB GB", u); i = 1; while (b >= 1024 && i < 4) { b /= 1024; i++ } printf "%.1f %s", b, u[i] }'; }
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

# Sets FI_PLATFORM (metal | cuda | none), FI_DEVICE, FI_CUDA_ARCH, FI_VRAM_MB, FI_UNIFIED_GB.
detect_platform() {
  FI_PLATFORM=none; FI_DEVICE="no supported GPU"; FI_CUDA_ARCH=0; FI_VRAM_MB=0; FI_UNIFIED_GB=0
  if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
    FI_PLATFORM=metal
    # sysctl can fail under sandbox/TCC; never let empty memsize blow up $(( )).
    local brand mem_bytes mem_label
    brand="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
    mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
    if [ -n "$mem_bytes" ]; then
      FI_UNIFIED_GB=$(( mem_bytes / 1073741824 ))
      mem_label="$FI_UNIFIED_GB GB unified memory"
    else
      mem_label="unified memory"
    fi
    FI_DEVICE="$brand, $mem_label"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    FI_PLATFORM=cuda
    local q; q="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits | head -1)"
    FI_DEVICE="$(echo "$q" | cut -d, -f1 | sed 's/^ *//')"
    FI_CUDA_ARCH="$(echo "$q" | cut -d, -f2 | tr -d ' .')"
    FI_VRAM_MB="$(echo "$q" | cut -d, -f3 | tr -d ' ')"
    FI_DEVICE="$FI_DEVICE, $(( FI_VRAM_MB / 1024 )) GB VRAM"
  fi
}

# Pick the text-encoder file for this platform. Metal ≤18 GB prefers Q4 to leave room for DiT.
select_llm() {
  local m="$1"
  if [ "$FI_PLATFORM" = metal ]; then
    if [ "${FI_UNIFIED_GB:-0}" -gt 0 ] && [ "$FI_UNIFIED_GB" -le 18 ] && [ -f "$m/Qwen3-4B-Q4_K_M.gguf" ]; then
      printf '%s\n' "$m/Qwen3-4B-Q4_K_M.gguf"
      return
    fi
    # 24 GB+ can keep Q8 on Metal; fall back to Q4 if that's all install fetched.
    if [ -f "$m/Qwen3-4B-Q8_0.gguf" ]; then
      printf '%s\n' "$m/Qwen3-4B-Q8_0.gguf"
      return
    fi
    if [ -f "$m/Qwen3-4B-Q4_K_M.gguf" ]; then
      printf '%s\n' "$m/Qwen3-4B-Q4_K_M.gguf"
      return
    fi
  fi
  printf '%s\n' "$m/Qwen3-4B-Q8_0.gguf"
}

# Engine flags tuned per platform.
engine_args() {
  detect_platform
  local m="$FI_HOME/models"
  local llm; llm="$(select_llm "$m")"
  FI_ARGS=(
    --diffusion-model "$m/z_image_turbo_int8_convrot.safetensors"
    --vae "$m/ae.safetensors"
    --llm "$llm"
    --cfg-scale 1.0
  )
  if [ "$FI_PLATFORM" = cuda ]; then
    FI_ARGS+=(--sage-attn)
    # int8 weights (6.2 GB) + text encoder (4.3 GB) only fit together on 12 GB+ cards.
    # Below that, weights live in RAM and stream into VRAM as needed.
    if [ "$FI_VRAM_MB" -lt 12000 ] || [ "${FI_OFFLOAD:-0}" = 1 ]; then FI_ARGS+=(--offload-to-cpu); fi
  elif [ "$FI_PLATFORM" = metal ]; then
    # Unified memory: never --offload-to-cpu (duplicates residency, can OOM).
    # mmap lets Metal map weights through shared buffers instead of copying.
    FI_ARGS+=(--mmap)
    # On ≤18 GB Macs, auto-fit spills the 4 GB text encoder to disk and the DiT
    # graph-cuts every step (~25 s/it). Pin DiT on Metal; keep TE/VAE weights in
    # system memory so sampling owns the GPU working set.
    if [ "${FI_UNIFIED_GB:-0}" -gt 0 ] && [ "$FI_UNIFIED_GB" -le 18 ]; then
      FI_ARGS+=(--backend "diffusion=MTL0,te=cpu,vae=MTL0")
      FI_ARGS+=(--params-backend "te=cpu,vae=cpu")
      # Leave headroom inside Apple's recommended working set for activations.
      FI_ARGS+=(--max-vram 10)
    fi
    # Auto probe often leaves flash-attn off on Metal after a weak micro-bench;
    # force it for the diffusion model (TE stays on CPU above).
    FI_ARGS+=(--flash-attn "diffusion=on")
  fi
}

require_install() {
  detect_platform
  [ -x "$FI_HOME/bin/sd-cli" ] || die "engine not built yet. Run ./install.sh"
  [ -f "$FI_HOME/models/z_image_turbo_int8_convrot.safetensors" ] || die "models missing. Run ./install.sh --models"
  local llm; llm="$(select_llm "$FI_HOME/models")"
  [ -f "$llm" ] || die "text encoder missing ($(basename "$llm")). Run ./install.sh --models"
}
