# Troubleshooting

Start with `./fi doctor`. It shows what was detected and which flags the engine gets.

**`missing 'cmake'` / `missing 'nvcc'`**
Install the build tools listed in the README. On NVIDIA, `nvcc` comes with the CUDA toolkit, not with the driver.

**`RTX 50-series (sm_120) needs CUDA 12.8 or newer`**
Update the CUDA toolkit. The driver alone isn't enough.

**CUDA build fails with "unsupported GNU version"**
Your gcc is newer than your CUDA release supports. Install an older one (for example `sudo apt install g++-13`).
The installer picks the newest `g++-14/13/12` it finds automatically.

**A download stopped halfway**
Run `./install.sh --models` again. Downloads resume where they left off and are checked against SHA-256.

**Hugging Face rate limit / 401 / 403 on model download**
Pass a token: `HF_TOKEN=hf_xxx ./install.sh --models`. Create one at https://huggingface.co/settings/tokens.

**Out of memory on NVIDIA**
Force weight offloading: `FI_OFFLOAD=1 ./fi serve`. It's already on automatically for cards under 12 GB.

**Engine fails to start on Turing / Quadro RTX (`sm_75`) with SageAttention**
SageAttention needs Ampere or newer (`sm_80+`). On Turing the scripts use `--flash-attn diffusion=on` instead. Run `./fi doctor` — `attn` should say flash attention, not SageAttention. Rebuild is not required for this flag change; restart `./fi serve`.

**CUDA 13 build fails targeting an old GPU**
CUDA 13 only offline-compiles Turing and newer (`sm_75+`). Maxwell / Pascal / Volta need CUDA 12.x.

**It's slower than the numbers in the README**
- Make sure you're timing `fi serve` or `fi bench`, not single `fi "…"` runs, which include loading the model.
- Laptop? Check temperature with `nvidia-smi`. Around 87 °C the GPU throttles itself.
- Close other apps that use the GPU (browsers with hardware acceleration, games).
- Turing cards won't match the RTX 3060 + SageAttention numbers; flash attention is the correct path.

**Apple Silicon is much slower than a 6 GB NVIDIA card**
Expected: CUDA + SageAttention is the fast path; Metal has no SageAttention. On a 16 GB Mac the old default also spilled the text encoder to disk and graph-cut every step (~25 s/it). `./fi doctor` should show Metal flags like `--mmap`, `--backend diffusion=MTL0,te=cpu,…`, and `Qwen3-4B-Q4_K_M.gguf`. If you still have only the Q8 encoder, run `./install.sh --models`. Do **not** set `FI_OFFLOAD=1` on Mac — unified memory makes CPU offload slower and hungrier.

**Port already in use**
`FI_SD_PORT=7870 FI_UI_PORT=7871 ./fi serve`

**Ubuntu / headless server — `./fi serve` “shows nothing”**
There is no desktop browser on the server. The UI listens on `0.0.0.0:7861`; open it from your laptop:

- Network URL printed at startup (`http://<server-ip>:7861`), after opening the port in `ufw` / the cloud security group, or
- SSH tunnel (no firewall change): `ssh -L 7861:127.0.0.1:7861 user@server` then open `http://127.0.0.1:7861` locally.

`localhost` on the server is not your laptop. The diffusion engine stays on `127.0.0.1:7860` on purpose; only the UI port needs to be reachable.

**Logs**
`.fi/build.log` (build), `.fi/server.log` (engine), `.fi/last.log` (last single image).
