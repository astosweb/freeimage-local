# Benchmarks

All measurements on one machine, November 2026:

- **GPU:** NVIDIA RTX 3060 Laptop, 6 GB VRAM, 60–65 W
- **RAM:** 16 GB · **OS:** Ubuntu Server 26.04, no desktop · **CUDA:** 12.4
- **Engine:** stable-diffusion.cpp `sd-server`, pinned commit in `install.sh`
- **Image:** 1024×1024, CFG 1.0, model already loaded

## 1. Sampling speed by weight format

Time spent sampling only, across different prompts.

| Setup | Sampling, 8 steps | Note |
|---|---|---|
| GGUF Q8_0 | 24–27 s | |
| GGUF Q5_K_M | 27–29 s | Fits VRAM better, still slower than Q8 |
| int8_convrot | 15–16.5 s | ~40% faster than Q8 |
| **int8_convrot + SageAttention** | **14.4–16.3 s** | Small extra gain, no visible change |
| int8 + Sage, 6 steps | 10.5–11.5 s | Slight softening |
| int8 + Sage, 4 steps | ~8 s | Visible loss of detail |

## 2. Quality: does int8 cost anything?

10 everyday requests, each turned into an English prompt, generated at seed 42 and scored by a vision model
(prompt adherence 0–3, image quality 0–3, share of images with artifacts). Overall = mean(adherence, quality) × (1 − 0.3 × artifacts), out of 100.
Seconds include writing the prompt with an LLM, so they're higher than pure sampling time.

| Setup | s / image | Adherence | Quality | Artifacts | Overall |
|---|---|---|---|---|---|
| GGUF Q8, 8 steps | 34.8 | 2.73 | 2.07 | 29% | 73 |
| **int8 + Sage, 8 steps** | **22.8** | 2.72 | 2.08 | 28% | **73** |
| int8 + Sage, 8 steps, res_multistep | 22.7 | 2.74 | 2.06 | 28% | 73 |
| int8 + Sage, 6 steps | 18.0 | 2.70 | 2.04 | 29% | 72 |
| int8 + Sage, 4 steps | 13.8 | 2.72 | 2.01 | 29% | 72 |

The automatic score barely moves with step count. Looking at full-resolution crops, 6 steps softens skin and fur slightly and
4 steps does so clearly. That's why the default is 8 and "Fast" is 6.

## 3. Engine: stable-diffusion.cpp vs ComfyUI

Same GGUF files, same prompts, end to end (prompt writing included).

| Engine · model | s / image | Overall |
|---|---|---|
| stable-diffusion.cpp · Q8_0 | 34.8 | 73 |
| stable-diffusion.cpp · Q5_K_M | 36.0 | 74 |
| ComfyUI · Q5_K_M | 39.0 | 70 |
| ComfyUI · Q8_0 | 45.1 | 68 |

## 4. Production

freeimage.dev runs this setup around the clock. Average generation time over its first 71 images (model warm,
text encoding and VAE decode included):

| Steps | Size | Images | Average |
|---|---|---|---|
| 6 | 1024×1024 | 54 | 12.6 s |
| 8 | 1344×768, 896×1152, 1152×896 | 4 | 15.3 s |
| 10 | 1024×1024 and others | 13 | 18.7 s |

## Things that didn't help

| Tried | Result |
|---|---|
| easycache / dbcache / cache-dit | No effect at 8 steps |
| `--backend te=cpu` | Text encoding 0.85 s → 3.4 s, sampling unchanged |
| `--disable-prefetch` | ~20% slower |
| `--flash-attn off` | 2× slower |
| CPU turbo off (`intel_pstate no_turbo`) | No change in heat or speed |

## Heat

The laptop GPU throttles itself at 87–89 °C (`nvidia-smi` reports "SW Thermal Slowdown"). Cold, it's 10–25% faster.
A stand that lifts the back of the laptop helps more than any flag.

## Ideas not tried yet

- The int8 file (6.2 GB) doesn't fully fit in 6 GB and streams in pieces; GPU use sits around 85%. A version that fits entirely could gain ~15%. On 16 GB+ cards the scripts already pass `--eager-load` so weights stay resident.
- Few-step distill LoRAs (2–4 steps) and SVDQuant int4.
- Generating at 768² and upscaling.

## High-VRAM / Turing notes

On Quadro RTX / RTX 20-series (`sm_75`) and other pre-Ampere CUDA GPUs:

- Build with `CMAKE_CUDA_ARCHITECTURES=75` (installer sets this from `nvidia-smi`).
- Do **not** use `--sage-attn` (needs `sm_80+`); use forced diffusion flash attention.
- With ≥16 GB VRAM, skip `--offload-to-cpu` and use `--eager-load` so int8 DiT + Q8 text encoder stay on GPU.

## Apple Silicon notes

Metal has no SageAttention. On ≤18 GB unified memory, freeimage-local:

- downloads `Qwen3-4B-Q4_K_M.gguf` instead of Q8 (saves ~1.8 GB),
- runs text encoding on CPU and keeps the DiT on `MTL0`,
- enables `--mmap` and diffusion flash-attention,
- never uses `--offload-to-cpu` (harmful on unified memory).

That stops the text encoder from spilling to disk and the DiT from graph-cutting every step. It will still be slower than an RTX 3060 — share `./fi bench` if you measure a Mac.

## Add your machine

```bash
./fi bench
```

Open an issue with the output and your GPU, driver and OS. Results from RTX 40/50-series cards and Apple Silicon are especially welcome.
