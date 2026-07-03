# Wan2.2-S2V CFG-darkening parity harness

Localizes the S2V-14B fp8 CFG bug (output darkens ~linearly in `cfg−1`; even text-only CFG at cfg 5
collapses to black while cfg=1 is healthy ⇒ the cond−uncond velocity difference is mis-scaled vs
ComfyUI). The C# transformer/pipeline dump per-stage tensors when `WAN_DEBUG_DIR` is set
(zero-cost off); `dump_s2v_reference.py` replays the SAME inputs through a faithful CPU-fp32 port of
ComfyUI `WanModel_S2V.forward_orig` on the real fp8 checkpoint; `diff_s2v_layers.py` reports
per-stage relL2 + mean/std ratios and names the first divergent stage.

Instrumented files (all dumps gated on `WAN_DEBUG_DIR`, tag per CFG branch set by the pipeline):
- `src/HartsyInference.Diffusion/Models/Denoisers/WanVideoDebugDump.cs` — tag support (`SetTag`/`WAN_DEBUG_TAG`), `DumpValues`
- `src/HartsyInference.Diffusion/Models/Denoisers/WanS2VTransformer.cs` — latent_in, text_embeds, post_patchify, post_condmask, ref_tokens, joined_tokens, timesteps, temb, timestep_proj (per-group), text_proj, block_{0,20,39}, pre_unpatchify, velocity_out
- `src/HartsyInference.Diffusion/Models/Denoisers/DiTBlocks/WanS2VAudioInjector.cs` — audio_delta_inj{0,11} (injector residuals)
- `src/HartsyInference.Video/Pipelines/WanS2VPipeline.cs` — audio_features, audio_global, audio_local (per branch), ref_latent, cfg_combined; sets the `cond`/`uncond` tag around each branch forward

## GPU run recipe (step 1: C# dumps — needs the 4090)

Tiny config, ONE step (multi-step runs overwrite the per-step .bin files):

```bash
cd /home/hartsy/Desktop/HartsyInference
dotnet build tests/HartsyInference.Video.Tests --framework net10.0   # once, if not already built
rm -rf /tmp/s2v_dbg && mkdir -p /tmp/s2v_dbg
LD_LIBRARY_PATH=$HOME/.local/lib/cuda13 CUDA_VISIBLE_DEVICES=0 \
WAN_DEBUG_DIR=/tmp/s2v_dbg WAN_NO_CACHE_CASTS=1 \
WAN_W=320 WAN_H=192 WAN_FRAMES=17 WAN_STEPS=1 WAN_CFG=5 \
dotnet test tests/HartsyInference.Video.Tests --framework net10.0 --no-build \
  --filter "FullyQualifiedName~WanS2V_Gpu_E2E" -v n
```

Notes:
- Do NOT set `HARTSY_S2V_TEXT_CFG` — the parity target is the reference flow (uncond = zeroed audio).
- Keep the reference image on (test default; `WAN_S2V_NO_REF=1` would change dump shapes — see gotchas).
- Dumps do mid-forward D2H reads (stream-sync per dump) — the instrumented step is slower; that's fine for 1 step.
- The pipeline overrides the tag per branch, so `WAN_DEBUG_TAG` is not needed.
- Result: `/tmp/s2v_dbg/layers/{cond,uncond}_*.bin` + `ref_latent.bin` + `cfg_combined.bin` + `shapes.txt`.

## Reference + diff (step 2: CPU, ~2-5 min per branch)

```bash
cd /home/hartsy/Desktop/HartsyInference/tests/python-reference/s2v_reference
venv/bin/python dump_s2v_reference.py --dump-dir /tmp/s2v_dbg --tag cond
venv/bin/python dump_s2v_reference.py --dump-dir /tmp/s2v_dbg --tag uncond
venv/bin/python diff_s2v_layers.py /tmp/s2v_dbg --tag cond
venv/bin/python diff_s2v_layers.py /tmp/s2v_dbg --tag uncond
```

The CFG mis-scaling shows up as the guidance direction, not either branch alone — after both diffs pass
(or drift equally), compare the cond−uncond velocity between C# and reference:

```bash
venv/bin/python - <<'EOF'
import numpy as np
d = "/tmp/s2v_dbg"
cs = np.fromfile(f"{d}/layers/cond_velocity_out.bin", np.float32) - np.fromfile(f"{d}/layers/uncond_velocity_out.bin", np.float32)
rf = np.fromfile(f"{d}/ref/layers/cond_velocity_out.bin", np.float32) - np.fromfile(f"{d}/ref/layers/uncond_velocity_out.bin", np.float32)
print(f"guidance dir: relL2={np.linalg.norm(cs-rf)/np.linalg.norm(rf):.3e} "
      f"mean cs={cs.mean():.6f} ref={rf.mean():.6f} (DC! cfg multiplies this) "
      f"std ratio={cs.std()/rf.std():.4f}")
EOF
```

## Interpreting the numbers

- The C# side computes fp8/F16 GEMMs while the reference dequantizes to fp32 — a smooth relL2 baseline
  drift (~1e-3…1e-2 growing through 40 blocks) is expected. Look for the first *step change* in relL2
  (or a mean/std ratio far from 1.0), not merely the 1e-2 flag; re-run the diff with `--threshold 5e-2`
  to silence pure-precision noise if needed.
- Magnitude bugs are invisible to correlation-style metrics — that's what the mean/std ratio columns are for.
- A healthy-per-branch diff with a bad guidance-direction DC (mean far off in the snippet above) points at
  a global velocity offset that CFG amplifies rather than a per-stage math bug.

## Port notes / gotchas

- **Checkpoint**: `Models/Stable-Diffusion/Wan/wan2.2_s2v_14B_fp8_scaled.safetensors` (raw ComfyUI key
  names, e.g. `time_embedding.0.weight`). fp8 dequant = `tensor.float() * <name-minus-.weight>.scale_weight`;
  the `.scale_input` companions are unused (they only feed ComfyUI's fp8 fast-path input quantization).
  Weights stream lazily — the 14B model is never materialized in fp32 at once (62 GB RAM box), which is
  why the port is functional (`F.linear` + lazy fetch) instead of `nn.Linear` modules.
- **cond_encoder**: ComfyUI's node always feeds a zeros control latent (`process_in(process_out(0)) == 0`),
  so `conv3d(zeros)` adds the cond_encoder bias to every main token. Both the C# port and this reference
  do that by default; `--no-control` isolates it.
- **Per-frame timesteps**: ComfyUI always expands `t` per latent frame; C# only builds groups when a
  reference latent is present (identical rows → identical math). WITHOUT a reference image the C# `temb`
  dump is `[1,dim]` vs the reference `[gt,dim]` — the diff reports a size mismatch on temb/timestep_proj
  only; every other stage still compares. Prefer running with the reference image.
- **Shapes**: C# token tensors are `[S,dim]` (B=1 dropped) vs `[1,S,dim]` here; audio dumps `[T,n,dim]` vs
  `[1,T,n,dim]`. The diff compares flat — element counts match.
- **Known blind spot**: the reference consumes the exact text embeds the C# branch fed the transformer, so
  a conditioning-prep divergence vs ComfyUI (e.g. text padding/batching differences upstream of the
  transformer) is NOT caught here — only transformer-internal math is.
- venv: CPU torch 2.12.1 + safetensors + numpy (`python3 -m venv venv && venv/bin/pip install
  --index-url https://download.pytorch.org/whl/cpu torch && venv/bin/pip install safetensors numpy`).
  Fallback if recreating offline: the ComfyUI venv at
  `/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI/venv` has torch 2.11 + safetensors.
