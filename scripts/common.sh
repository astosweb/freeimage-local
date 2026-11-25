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

# Digits-only; empty/non-numeric → 0. Safe for $(( )) and -ge/-lt tests.
fi_uint() { case "$1" in ''|*[!0-9]*) printf '0' ;; *) printf '%s' "$1" ;; esac; }

# System RAM in MiB from /proc (Linux) or sysctl (macOS). 0 if unknown.
system_mem_mb() {
  local kb=0
  if [ -r /proc/meminfo ]; then
    kb="$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo 2>/dev/null || echo 0)"
  elif command -v sysctl >/dev/null 2>&1; then
    local bytes; bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    kb=$(( bytes / 1024 ))
  fi
  fi_uint "$(( $(fi_uint "$kb") / 1024 ))"
}

# Sets FI_PLATFORM (metal | cuda | none), FI_DEVICE, FI_CUDA_ARCH, FI_VRAM_MB, FI_UNIFIED_GB.
detect_platform() {
  FI_PLATFORM=none; FI_DEVICE="no supported GPU"; FI_CUDA_ARCH=0; FI_VRAM_MB=0; FI_UNIFIED_GB=0
  if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
    FI_PLATFORM=metal
    # sysctl can fail under sandbox/TCC; never let empty memsize blow up $(( )).
    local brand mem_bytes mem_label
    brand="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
    mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
    if [ -n "$mem_bytes" ] && [ "$(fi_uint "$mem_bytes")" != 0 ]; then
      FI_UNIFIED_GB=$(( $(fi_uint "$mem_bytes") / 1073741824 ))
      mem_label="$FI_UNIFIED_GB GB unified memory"
    else
      mem_label="unified memory"
    fi
    FI_DEVICE="$brand, $mem_label"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    FI_PLATFORM=cuda
    local q name cap_raw mem_raw
    # GB10 / DGX Spark (unified memory): memory.total is "[N/A]" — never feed that to $(( )).
    q="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
    name="$(printf '%s' "$q" | cut -d, -f1 | sed 's/^ *//;s/ *$//')"
    cap_raw="$(printf '%s' "$q" | cut -d, -f2 | tr -d ' .')"
    mem_raw="$(printf '%s' "$q" | cut -d, -f3 | tr -d ' ')"
    FI_CUDA_ARCH="$(fi_uint "$(printf '%s' "$cap_raw" | tr -cd '0-9')")"
    if [ "$(fi_uint "$mem_raw")" != 0 ] && [ "$mem_raw" = "$(fi_uint "$mem_raw")" ]; then
      FI_VRAM_MB="$(fi_uint "$mem_raw")"
      FI_DEVICE="$name, $(( FI_VRAM_MB / 1024 )) GB VRAM"
    else
      # Unified memory GPU (NVIDIA GB10 / DGX Spark, etc.): budget = system RAM.
      FI_VRAM_MB="$(system_mem_mb)"
      FI_UNIFIED_GB=$(( FI_VRAM_MB / 1024 ))
      if [ "$FI_UNIFIED_GB" -gt 0 ]; then
        FI_DEVICE="$name, ${FI_UNIFIED_GB} GB unified memory"
      else
        FI_DEVICE="$name, unified memory"
      fi
    fi
    [ "$FI_CUDA_ARCH" -gt 0 ] || die "nvidia-smi did not report a CUDA compute capability for '$name'."
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
    # SageAttention needs Ampere+ (sm_80). Turing (sm_75: Quadro RTX / RTX 20xx)
    # fails context creation with --sage-attn; force flash attention instead.
    # Blackwell GB10 is sm_121 — SageAttention is fine.
    if [ "${FI_CUDA_ARCH:-0}" -ge 80 ]; then
      FI_ARGS+=(--sage-attn)
    else
      FI_ARGS+=(--flash-attn "diffusion=on")
    fi
    if [ "${FI_UNIFIED_GB:-0}" -gt 0 ]; then
      # CUDA UMA (GB10 / DGX Spark): same trap as Metal — never --offload-to-cpu.
      FI_ARGS+=(--eager-load)
    elif [ "$FI_VRAM_MB" -lt 12000 ] || [ "${FI_OFFLOAD:-0}" = 1 ]; then
      # int8 (6.2 GB) + TE (4.3 GB) only fit together on 12 GB+ discrete cards.
      FI_ARGS+=(--offload-to-cpu)
    elif [ "$FI_VRAM_MB" -ge 16000 ]; then
      # 16 GB+ discrete (e.g. Quadro RTX 8000): keep the full working set resident.
      FI_ARGS+=(--eager-load)
    fi
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
      # Full-frame VAE decode OOMs at 1024² and falls back to 512² tiles at 50% overlap (~30 s).
      # Wide tiles with small overlap decode the same image in ~20 s.
      FI_ARGS+=(--vae-tiling --vae-tile-size 1024x512 --vae-tile-overlap 0.125)
      # Sampling is ~30 s/step at 1024² on a base M4 GPU; 6 steps saves ~60 s per image.
      FI_STEPS="${FI_STEPS:-6}"
    fi
    # Auto probe often leaves flash-attn off on Metal after a weak micro-bench;
    # force it for the diffusion model (TE stays on CPU above).
    FI_ARGS+=(--flash-attn "diffusion=on")
  fi
  FI_STEPS="${FI_STEPS:-8}"
}

require_install() {
  detect_platform
  [ -x "$FI_HOME/bin/sd-cli" ] || die "engine not built yet. Run ./install.sh"
  [ -f "$FI_HOME/models/z_image_turbo_int8_convrot.safetensors" ] || die "models missing. Run ./install.sh --models"
  local llm; llm="$(select_llm "$FI_HOME/models")"
  [ -f "$llm" ] || die "text encoder missing ($(basename "$llm")). Run ./install.sh --models"
}
