# FastFrames GPU runbook

Everything a rented GPU needs to finish FastFrames: which weights, where they go, what to run, and what to send back. The CPU-side
work is already merged or in review; this file covers only the runs that need a card.

## Hardware

| Run | Minimum | Why |
|---|---|---|
| S clip (81 frames, 832x480) | 24 GB (the 4090 peaked at 20.0 GB, all 30 blocks resident) | Weights 2.8 GB F16, cache about 11.6 GiB F32 at 81 frames (S runs F32 slots: its inner width is 1536) |
| Decoder GPU parity, exact lane | any CUDA card | Synthetic decoder, small |
| Streamed vs resident test | any CUDA card | Tiny model |
| Renorm device-sync test (#1209) | any CUDA card | Tiny refiner; asserts zero device-to-host syncs |
| L clip (81 frames) | 80 GB (A100 or H100 80 GB) | Generator 28.6 GB BF16, cache about 27 GB F16 at 81 frames; peak about 60 GB |

L does not fit a 24 GB card at 81 frames. With block streaming on a 24 GB card, L runs only about 14 latent frames (about 53 pixel
frames), because the cache is not streamed. That is a fallback, not the target.

Disk: about 50 GB for the seven files, the build and the clips. Host RAM: 64 GB is enough for the loaders.

## Weights

Place every file under the models root, in the folders the catalog expects (`FastFrames/fastframes-s`, `FastFrames/fastframes-l`,
`text_encoders`). The generator and its two companions must sit in the same folder.

| File | Source | Bytes | sha256 |
|---|---|---|---|
| `FastFrames/fastframes-s/framewise-2step.pt` | `zhuhz22/Causal-Forcing`, `causal-forcing++/framewise-2step.pt` | 5,676,220,819 | `8f899a016dffb9539788fac82c6200e43ef6e23b6e61ec6b885bb507a760ce42` |
| `FastFrames/fastframes-s/kv_refiner.safetensors` | `apple/FastFrames-S`, `fastframes-s/kv_refiner.safetensors` | 692,418,448 | `b4f36d00effd78a60435526dcc2b1a0e6bbed0f09621ee8ab4019460769a7bbb` |
| `FastFrames/fastframes-s/pixel_decoder.safetensors` | `apple/FastFrames-S`, `fastframes-s/pixel_decoder.safetensors` | 255,494,944 | `815682581f6376d2cdacc60568facf1bd979112091e4157fb6c00ae946b89b32` |
| `FastFrames/fastframes-l/dit.safetensors` | `apple/FastFrames-L` (revision `d18ec5f054bb03f6b8e561a63d52af3142b5b81d`), `fastframes-l/dit.safetensors` | 28,577,096,680 | `cd0ec8980516663d623186fafe4b2a3ca9398e7ed29ccce1c65581640bef15de` |
| `FastFrames/fastframes-l/kv_refiner.safetensors` | same repo, `fastframes-l/kv_refiner.safetensors` | 1,780,342,784 | `841047b9226cf770ca19f99ef718edcb57db29c29ee58a64f9ae5dc1d6fbdf74` |
| `FastFrames/fastframes-l/pixel_decoder.safetensors` | same repo, `fastframes-l/pixel_decoder.safetensors` | 286,952,688 | `75e6c99c358a3d8b87476add2c732b0853aef76279bae4c807a6157e95555f55` |
| `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `Comfy-Org/Wan_2.1_ComfyUI_repackaged`, `split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6,735,906,897 | `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` |

Check each file's sha256 before the run (`tools/fastframes/run_pod.sh verify` does this). Do not load a `.pt` file any way except the
engine's own reader: the generator is a pickle.

## Code

Branch: check out the newest `main`. Every FastFrames change this runbook needs is merged:
- decoder/refiner view fix, unpatchify, renorm on the device: #1161, #1173, #1209;
- decoder GPU parity (exact and default lanes): #1214;
- admission and post-placement VRAM check: #1175, #1219;
- block streaming: #1178;
- this runbook and the run script: #1220.

## Runs

Run every step below on the newest `main`, so each step runs on the code under test.

`tools/fastframes/run_pod.sh` runs the steps. It needs `MODELS` (the models root) and `REPO` (the checkout). It exports
`HARTSYINFERENCE_MODELS` from `MODELS`, so the CLI reads the verified weights and does not download them again. Run it with `bash`:

    MODELS=/path/to/Models REPO=/path/to/HartsyInference bash tools/fastframes/run_pod.sh build
    MODELS=/path/to/Models REPO=/path/to/HartsyInference bash tools/fastframes/run_pod.sh verify

1. Build the CLI: `run_pod.sh build` (net10.0; the CLI runs as a `.dll`, so no `.exe` is needed on Linux).
2. Verify the weights: `run_pod.sh verify` (byte counts and sha256 of all seven files).
3. Decoder parity, both lanes: `run_pod.sh decoder`. The exact lane must be under 1e-5. Record its measured value; it goes into
   `FastFramesDecoderGpuParityTests.cs`'s doc comment.
4. Streamed vs resident: `run_pod.sh streamed`.
5. Renorm device-sync test (#1209): `run_pod.sh renorm`. The log must show the test as passed, not skipped: it is a CUDA-only fact, so a
   machine without a device skips it and `dotnet test` still exits 0. A skip is not a result; record it as a skip.
6. S clip: `run_pod.sh clip-s`. Reference from the 4090 run: 93 s total, steps 0.85 to 1.1 s, peak VRAM 20.0 GB, all 30 blocks resident.
7. L clip: `run_pod.sh clip-l`. It refuses unless the card reports at least 79 GB. The catalog flag is `CliDrivable=true` for `fastframes-l` (set once an L clip ran through the CLI on an H100). If the CLI refuses,
   that is a finding, not something to work around on a merged branch.

Each run writes its log and an `nvidia-smi` sample (every 2 s) under `OUT`. `run_pod.sh all` runs steps 1 to 7 in order and stops at the first
failure; on a card under 79 GB it skips L with a message.

## Return

For each run: the test output or the CLI log, the per-step timings, the peak VRAM from the `nvidia-smi` log, the exact-lane relative
difference, and the frames (81 PNG files per clip). Send the frames to the parent, who makes the contact sheet and sends it to the user.

## Stop conditions

- A sha256 mismatch: stop, do not run.
- Out of memory on S at 24 GB: record the peak and stop; it is a result, not a retry.
- The exact lane above 1e-5: record the value and stop before the clip runs; do not change the bound.
