# Session handoff — Krea2 flash attention via cuDNN

**Paste this to the next session. It is self-contained.** Read the linked memories too (they're loaded automatically):
`krea2-flash-attention-cudnn`, `cudnn-fused-sdpa-recipe`, `local-engine-deploy-to-swarm`, `image-genperf-host-glue-wins`, `cuda13-libs-ld-path`.

---

## ✅ STATUS 2026-07-05: cuDNN fused SDPA (§3, "A") is DONE, shipped, live, coherent — but see the pivot.

Implemented pure-C# cuDNN backend-graph SDPA (`CudnnApi.cs`, `CudnnSdpa.cs`, resolver `"cudnn"` case,
`CudaBackend.TryCudnnSdpa` gated on `HARTSY_SDPA_CUDNN=1`). Engine `alpha.43.124-local`, deployed + running in Swarm.
Per-call **62.7→5.5ms (11×)**, SDPA 52%→~12% of GPU time, workspace 0. Validated (numpy + `CudnnSdpaTests`, coherent
astronaut @1024²). **The recipe that mattered:** the cuDNN **unified SOFTMAX backend op** (≥9.21) — the naive
decomposed-softmax graph does NOT hit the fused engine. Full recipe in memory `cudnn-fused-sdpa-recipe`.

**THE PIVOT — this did NOT beat Comfy, and here's why:** Krea2 wall only moved **36.7→33.0s (1.1×)**, gap 5.6→5.1×.
The profile proves **Krea2 is HOST-BOUND, not GPU-bound**: all GPU ops sum to ~15s/gen (serialized) but wall is 33s
→ ~18s is host launch overhead / non-overlapped H2D that killing SDPA GPU time can't touch. **SDPA is no longer the
bottleneck.** Post-cuDNN profile top ops (2 gens): `Linear` 12.4s (4714 calls), `SDPA` 3.2s, `GatedResidual` 3.1s,
`RmsNorm` 2.7s, `Conv2D` 2.6s, `Permute0213` 2.2s, `H2D_MISS_BIG` 1.7s. **NEXT SESSION = go straight to "B" (§3.6):**
CUDA-graph capture of the denoise step + fuse the DiT block glue (GatedResidual/RmsNorm/AffineBroadcast/Permute =
the still-pending Krea2 block-GPU-residency) + kill the H2D_MISS_BIG weight re-uploads. cuDNN SDPA stays env-gated
(not default-on) until B surfaces the banked GPU win. Everything below is the original A handoff, kept for reference.

---

## 1. The mission

HartsyInference is a pure C#/.NET inference engine, run through SwarmUI via the HartsyInference backend extension.
We are closing the image-gen speed gap vs ComfyUI, focused on **Krea2-Turbo** (12.9B fp8 MMDiT, 8-step, the model
everyone wants to test right now). **Goal: make Krea2-Turbo faster than ComfyUI** (Comfy 6.5s; Hartsy is 35s now,
was 70.6s).

## 2. What last session did (all live in engine `alpha.43.123-local`)

Fixed + validated 3 models (coherent images, deployed) by moving per-block **host-CPU RoPE** to the GPU (+ Chroma
mask-once). See `image-genperf-host-glue-wins`:
- **ERNIE-Image 296→53s (5.6×)**, **Chroma1-HD 550→119s (4.6×)**, **Krea2-Turbo 70.6→36.7s (1.9×)**.
Then on Krea2:
- **Profiled** it (SDPA=**52%** of GPU time, materialized ~2GB score matrix; Linear 18% already native-fp8; RoPE
  0.2%). SDPA is the whole game.
- **F16 SDPA** (`Krea2Attention.cs`, `allowF16:true` — Krea2 QK-norms Q/K so it's safe): only **1.05×** (casts
  offset it, matrix still materialized). Kept.
- **Decision (user):** beat Comfy via **cuDNN fused attention** (library P/Invoke, like the cuBLAS the engine
  already uses — the route PyTorch/Comfy take). The from-scratch WMMA flash kernel is documented-slower than cuBLAS.
- **Installed cuDNN 9.24.0.43 for CUDA 13** at `~/.local/lib/cuda13/libcudnn*.so.9`, proven loadable +
  `cudnnCreate` works (Python ctypes, `cudnnGetVersion`=92400).

## 3. THIS SESSION'S TASK — implement cuDNN fused SDPA

Wire cuDNN's fused scaled-dot-product-attention as a new fast path in the engine's SDPA, gated behind
`HARTSY_SDPA_CUDNN=1`, falling back to the current materialized path. Target: Krea2 SDPA (62.7ms/call) → few ms →
Krea2 wall 35s → **beat 6.5s**.

**Do it in this order (prototype in Python first — the engine build/pack/deploy cycle is ~3 min, too slow to debug
cuDNN descriptor errors against):**

1. **Get the cuDNN headers back** (they were lost to a full disk). Re-download the tarball's `include/` only:
   `curl -sL https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.24.0.43_cuda13-archive.tar.xz | tar -xJ --wildcards '*/include/*'`
   You need `cudnn_backend.h`/`cudnn_graph.h` for the `CUDNN_BACKEND_*_DESCRIPTOR` types and `CUDNN_ATTR_*` enum values.
   **⚠️ disk is ~100% full (~5GB free)** — MOVE don't copy large files; prune `~/.nuget/packages`, old
   `~/.local/share/hartsy-local-nuget/*` (keep newest), `src/*/{bin,obj}` first if needed.

2. **Prototype the SDPA graph in Python ctypes** against `~/.local/lib/cuda13/libcudnn.so.9`. Build the op-graph:
   `S = Q@Kᵀ` (matmul) → `S *= attn_scale` (pointwise) → `P = softmax(S)` (reduction max → sub → exp → reduction sum
   → div) → `O = P@V` (matmul). Then: operation-graph descriptor → heuristic engine → engine-config → execution-plan
   → variant-pack (Q/K/V/O device ptrs + workspace) → `cudnnBackendExecute`. **Verify** the output matches a numpy
   reference (and/or the engine's materialized path) within tol, then **time it** vs 62.7ms/call at Krea2 shape
   (B=1, H=24, S≈4608, D=128, F16 or TF32). If it's not clearly faster, stop and reconsider before touching C#.

3. **Translate the working graph to C#:**
   - Add a `"cudnn"` case to `CudaLibraryResolver.ResolveLibrary` → `libcudnn.so.9` (mirror the existing `cublas` case).
   - `src/HartsyInference.Cuda/CudnnApi.cs`: `[LibraryImport("cudnn")]` for `cudnnCreate/Destroy/SetStream` +
     `cudnnBackendCreateDescriptor/SetAttribute/GetAttribute/Finalize/Execute`.
   - `src/HartsyInference.Cuda/CudnnSdpa.cs`: build+cache the plan (keyed by shape/dtype), execute on the compute stream.

4. **Wire into `CudaBackend.ScaledDotProductAttention`** (`src/HartsyInference.Cuda/CudaBackend.cs:2187`): at the top
   of the `mask is null && query.DType==F32` branch (~line 2211), if `EnvFlag("HARTSY_SDPA_CUDNN")` and shape is
   supported (D∈{64,128}, MHA), route to `CudnnSdpa`. Keep everything else as fallback. Q/K/V are `[B,H,S,D]` here.

5. **Deploy + benchmark Krea2** (§4/§5 below). Verify the image is coherent. If it beats Comfy, make the cuDNN path
   the default for RMS-normed-Q/K archs; keep the env kill-switch.

6. **Then do "B"** (only after A lands): CUDA-graph capture of the denoise step + fuse RMSNorm/AdaLN/SwiGLU. These
   attack the ~30% launch-overhead tail. `CudaGraph` capture is proven (memory), not yet wired into a pipeline.

## 4. Deploy workflow (engine change → running Swarm) — see `local-engine-deploy-to-swarm`

```bash
# 0. env for ALL Swarm launches (else libcublas.so.13 not found):
export LD_LIBRARY_PATH=/home/hartsy/.local/lib/cuda13:$LD_LIBRARY_PATH
cd /home/hartsy/Desktop/HartsyInference
# 1. compile-check the changed project
dotnet build src/HartsyInference.Cuda/HartsyInference.Cuda.csproj -c Release -f net8.0 --nologo -v quiet
# 2. pack a NEW -local version (bump NNN each time; last was 43.123)
dotnet pack HartsyInference.sln -c Release -p:VersionSuffix=alpha.43.124-local -o ~/.local/share/hartsy-local-nuget --nologo -v quiet
# 3. bump the extension pin
SW="/home/hartsy/Desktop/Swarm/SwarmUI.not too old"
sed -i 's/alpha.43.123-local/alpha.43.124-local/' "$SW/src/Extensions/SwarmUI-HartsyInference-Backend/SwarmUI-HartsyInference.csproj"
# 4. shutdown Swarm, wait for port, FORCE extension rebuild (the gotcha), relaunch
SID=$(curl -s -X POST http://192.168.10.188:7801/API/GetNewSession -H "Content-Type: application/json" -d '{}' | python3 -c "import sys,json;print(json.load(sys.stdin)['session_id'])")
curl -s -X POST http://192.168.10.188:7801/API/ShutdownServer -H "Content-Type: application/json" -d "{\"session_id\":\"$SID\"}"
until ! ss -tlnp 2>/dev/null | grep -q ":7801 "; do sleep 3; done
rm -rf "$SW/src/bin/extensions/SwarmExtensionSwarmUI-HartsyInference-Backend" "$SW/src/Extensions/SwarmUI-HartsyInference-Backend/obj"
cd "$SW"; nohup ./launch-linux-dev.sh > /tmp/swarm.log 2>&1 & disown
# 5. wait for: "[HartsyInference] Backend #2 live on CUDA (NVIDIA GeForce RTX 4090, SM 8.9)" in /tmp/swarm.log
# 6. verify fresh DLL timestamp: ls -la "$SW/src/bin/extensions/SwarmExtension*/HartsyInference.Diffusion.dll"
```
**Why the `rm -rf`:** `ExtensionsManager` skips the extension rebuild unless the built DLL is missing (it keys on the
extension's git HEAD, NOT the csproj) — so a pin bump alone reuses the OLD engine. Deleting the built folder forces it.

## 5. How to test + record benchmarks

**Harness (durable copy):** `benchmarks/swarm_image_bench/bench_t2i.py` + `models.json`.
```bash
cd /home/hartsy/Desktop/HartsyInference/benchmarks/swarm_image_bench
python3 bench_t2i.py --backend hartsy --config models.json --out /tmp/krea2.json --only Krea2-Turbo
# → prints cold + 3 warm timings + peak VRAM. Random seed each run (defeats Swarm's result cache).
```
- **Warm median** is the headline number. **Comfy baseline Krea2-Turbo = 6.5s** (RTX 4090). Gap = hartsy/comfy.
- **ALWAYS verify coherence** (don't trust timing alone — a broken kernel can be fast + garbage): find the newest
  `"$SW/Output/local/raw/<date>/*Krea2*.png"` and Read/view it — must be a clean astronaut-on-horse, not noise.
- **Profile** (find where time goes): relaunch Swarm with `export HARTSY_PROFILE=1 HARTSY_PROFILE_SYNC=1` (BOTH — SYNC
  alone doesn't accumulate), run 2 gens, `ShutdownServer` to dump `/tmp/hartsy_profile.txt` (op | calls | total_ms,
  sorted). Relaunch WITHOUT the flags for real timing (profiler serializes ops → slow).
- **Record** results in `benchmarks/results/image_comfy-vs-hartsy_2026-07-05.md` (the "Optimization results" table).

## 6. Critical environment facts (verify before trusting any number)

- **Swarm API binds ONLY to `http://192.168.10.188:7801`** (not 127.0.0.1).
- **GPU_ID 0 = RTX 4090 (SM 8.9)**, 1 = 3060 — **reversed from nvidia-smi**. Confirm the init line says `RTX 4090`.
- **Backends:** Hartsy = backend #2 (enabled). Comfy = #1 (disabled). Toggle via `/API/ToggleBackend` `{backend_id,
  enabled}`, one at a time, for A/B. To bench Comfy: enable #1 (GPU_ID 0 = 4090), disable #2.
- **Randomize seed** every gen or you time Swarm's ~0.2s result cache, not a real gen.
- **Disk ~100% full** (~5GB free) — prune before big downloads/copies; MOVE not copy.
- Current clean state: Swarm UP, `alpha.43.123-local`, profiler OFF, all 3 model fixes + F16 SDPA live.

## 7. Definition of done

`HARTSY_SDPA_CUDNN=1` routes Krea2 attention through cuDNN; Krea2-Turbo warm gen **< 6.5s** with a coherent image;
cuDNN path made default for safe (RMS-normed-Q/K) archs with an env kill-switch; results recorded; then start "B".
