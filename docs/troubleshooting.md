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

**Out of memory on NVIDIA**
Force weight offloading: `FI_OFFLOAD=1 ./fi serve`. It's already on automatically for cards under 12 GB.

**It's slower than the numbers in the README**
- Make sure you're timing `fi serve` or `fi bench`, not single `fi "…"` runs, which include loading the model.
- Laptop? Check temperature with `nvidia-smi`. Around 87 °C the GPU throttles itself.
- Close other apps that use the GPU (browsers with hardware acceleration, games).

**Port already in use**
`FI_SD_PORT=7870 FI_UI_PORT=7871 ./fi serve`

**Ubuntu / headless server — `./fi serve` “shows nothing”**
There is no desktop browser on the server. The UI listens on `0.0.0.0:7861`; open it from your laptop:

- Network URL printed at startup (`http://<server-ip>:7861`), after opening the port in `ufw` / the cloud security group, or
- SSH tunnel (no firewall change): `ssh -L 7861:127.0.0.1:7861 user@server` then open `http://127.0.0.1:7861` locally.

`localhost` on the server is not your laptop. The diffusion engine stays on `127.0.0.1:7860` on purpose; only the UI port needs to be reachable.

**Logs**
`.fi/build.log` (build), `.fi/server.log` (engine), `.fi/last.log` (last single image).
