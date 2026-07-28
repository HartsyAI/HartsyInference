# VPS / Big-GPU Test Handoff

Everything below was **built and compiles on the dev box (3060/12GB) but could not be fully run there**
(too big to fit, or needs a reference to parity-check against). This is the "spin up a bigger GPU and
verify it all" guide: what to run, in what order, what a PASS looks like, and exactly where the fix lives
if it fails — so GPU time is testing/debugging, not code archaeology.

Two independent workstreams: **(A) the Qwen3.5/3.6 LLM+VLM family** and **(B) Mage-Flow image gen**.
Do A first (cheaper, mostly already green) to warm up the harness, then B (the build-blind image model).

---

## A. Qwen3.5 / 3.6 family (LLM + VLM + MoE)

One command runs every tier with PASS/FAIL:

```bash
bash scripts/gpu-verify-qwen35.sh                 # all tiers (~40GB of GGUF, cached)
bash scripts/gpu-verify-qwen35.sh moe-text kat    # just the untested-on-dev-box MoE tiers
```

| Tier | Model (~size) | Verified on dev box? | On GPU: |
|---|---|---|---|
| `dense` | Qwen3.5-0.8B (0.5GB) | ✅ green | regression anchor — must stay green |
| `vl` | Qwen3.5-0.8B + mmproj (0.7GB) | ✅ green (OCR+VQA) | regression anchor |
| `moe-text` | Qwen3.5-35B-A3B `qwen35moe` (~11GB IQ2) | ❌ **OOM'd dev box** | **PRIORITY 1** — first real MoE run |
| `moe-vl` | + mmproj | ❌ | MoE + vision together |
| `vl9b` | Qwythos-9B-v2 (~6GB) | ❌ | larger VL tower |
| `kat` | KAT-Coder-V2.5-Dev `qwen35moe` (~20GB) | ❌ | MoE on a real coder |

**PASS** = coherent, on-topic output (the harness greps for a keyword). **If `moe-text` fails** (garbage),
the suspects, in order, all live in `src/HartsyInference.LLM/Ssm/Qwen35Model.cs`:
1. Stacked-expert byte-range split (`BuildMoeLayerWeights` / `SplitExpertsInto`) — layout/stride.
2. Shared-expert gate wiring (`ffn_gate_inp_shexp` → `mlp.shared_expert_gate`, rank-1 → [1,hidden] reshape).
3. Router config (`MoeConfig`: softmax + `NormTopKProb=true`, no group routing) vs llama.cpp `qwen35moe.cpp`.
4. MTP-block skip (`block_count − nextn.predict_layers`).
Cross-check against `github.com/ggml-org/llama.cpp · src/models/qwen35moe.cpp`. No GPU needed to write the fix.

After `moe-text` passes → bump the `qwen35moe` catalog entry `Status` from `vp` to `ok` and note it in
`docs/Checklists/MODEL_STATUS_LLM.md`.

**Deliberately NOT built (don't test — nothing to run):** Qwen3-VL *deepstack* (no GGUF conversion carries the
merger tensors — verified 0.8B+4B) and text-side spatial M-RoPE (matches the engine's verified Qwen2.5-VL
scalar-position approach). See `docs/Design/TRENDING_MODELS_PHASED_PLAN.md §1C`.

---

## B. Mage-Flow (image gen) — **build-blind, first real run happens here**

Microsoft Mage-Flow: a 4B dual-stream NR-MMDiT (reuses the engine's `QwenImageTransformer` via the
`QwenImageConfig.MageFlow` preset) + Qwen3-VL-4B text encoder + a **bespoke one-step `MageVAE`** decoder. All
of it **compiles and is registered** (`hartsy image -m mage-flow` resolves), but **none of it has run** — the
whole VAE is a from-scratch port with no local reference. Expect the first gen to be wrong and to need
parity-debugging. This is the main GPU task.

### B.1 Get a runnable checkpoint

The official `microsoft/Mage-Flow` transformer is **bf16, 8.2GB** — fits a 24GB+ card directly. On smaller
cards, quantize the DiT to fp8 or GGUF Q4 first (the recipe loads either **quant-native** — GGUF stays
quantized for the per-GEMM dequant path). The Qwen3-VL-4B TE (`SideModels.Qwen3VL_4B`) is already fp8
(~2.2GB); the MageVAE is ~0.35GB.

```bash
# T2I (downloads DiT + TE + VAE on first run):
hartsy image "a red fox in snow, photorealistic" -m mage-flow --steps 30 --cfg 5 -o ./out
# Turbo variant (4 steps, guidance-free) — point --model-path at a *-turbo* checkpoint:
hartsy image "..." --model-path /path/mage-flow-turbo.safetensors --steps 4 --cfg 1 -o ./out
```

### B.2 Parity checklist (in likely-failure order)

The DiT + RoPE + scheduler are reused from the *verified* Qwen-Image stack, so if the image is structured-but-
wrong the bug is almost certainly in the **MageVAE decoder** or the **conditioning**. All decoder parity traps
are annotated in-code (`src/HartsyInference.Diffusion/Models/Vae/Mage/MageVaeDecoder.cs`):

1. **MageVAE decode** (highest risk). Isolate it: dump the reference latent from the Python model (or a known
   latent), run `MageVaeDecoder.Decode` on it, compare to the reference RGB. Check, in order:
   - NeRF DCT coord table `BuildDctCoords` — `freqs = linspace(0,8,8)` (endpoint-inclusive step 8/7, NOT arange).
   - `Fold` / `BuildNerfInput` reshape & sub-pixel ordering (row-major `sp = ky*16+kx`).
   - `.byte()` truncation on output (`ToRgbBytes` — truncate, not round).
   - CoD `AttnBlock` uses **global** attention here vs the reference's patch-32 attention — exact for latents
     ≤32 (≤512px), an approximation above; if large images are wrong but 512px is right, that's this.
   - Channel-dim LayerNorm2d (affine=false) in `MageDiCoBlock.ChannelLayerNormAffineFalse`.
2. **Chat template / text conditioning** — `MageFlowRecipePipeline` reuses the Qwen-Image/Krea2 system prompt +
   prefix-drop; Mage-Flow has its own `chat_template.json`. If images are coherent but ignore the prompt,
   transcribe Mage-Flow's actual template here.
3. **Noise RNG** — `MageFlowPipeline.GaussianLatent` uses a host Box-Muller (System.Random), which will NOT
   match torch's RNG → different (not wrong) images per seed. For reference-image parity, swap in the same
   noise the reference used (dump it) before comparing.
4. **Scheduler** — flow-match Euler static shift 6.0; verify the velocity sign/direction against
   `FlowMatchEulerDiscreteScheduler` output if the image is noisy/inverted.
5. **DiT quant** — if using an fp8/GGUF DiT and output is garbage, first confirm bf16 works (rules out the
   quant path vs the port).

### B.3 Edit (Mage-Flow-Edit-Turbo)

`SupportsEditing=true`; the DiT reuses `refGrids` frame-axis RoPE. Needs the MageVAE **encoder** + a ref-latent
concat + Qwen3-VL vision conditioning (built alongside this handoff — see the recipe/pipeline edit path). Test
with a source image once T2I decode is confirmed correct (debugging edit on an unverified decoder = two
unknowns at once).

---

## B.4 SwarmUI (test Mage-Flow through Swarm)

Mage-Flow is wired into the SwarmUI extension (`/SwarmUI/src/Extensions/SwarmUI-HartsyInference`), **2 files**:
- `Generation/ModelSupport.cs` — `["mage-flow"] = new("mage-flow", Kind.Image)` in `_families`.
- `Generation/ModelClassRegistrations.cs` — `RegisterMageFlow()`: a `mage-flow` compat class + two model classes
  (`mage-flow-t2i`, `mage-flow-edit-turbo`) detected by `img_in.weight` + `.attn.add_q_proj.` keys. Steps/CFG
  defaults and edit support come from the **engine recipe** (`ImageFeatures.Img2Img`); the init image is the edit
  reference. Both my files compile clean.

⚠️ **BLOCKER (pre-existing, not Mage-Flow):** the extension currently fails to build against the refactored engine
— `Backends/HartsyInferenceBackend.cs` calls `new InferenceEngine(a, b)` / `InferenceEngine.Create(a, b)` which no
longer exist after the E-* engine refactor. This is the pending "SwarmUI extension refactor." **Sync the extension
backend to the current `InferenceEngine` API first**; then Mage-Flow appears in Swarm automatically (drop a
Mage-Flow checkpoint in the models folder → it classifies into `mage-flow` → generates; init image → edit).

## C. Environment / sizing quick reference

| Component | Size | Notes |
|---|---|---|
| Qwen3.5-35B-A3B IQ2 | ~11GB | needs ~16GB VRAM or CPU-offload |
| KAT-Coder Q4 | ~20GB | needs 24GB+ |
| Mage-Flow DiT bf16 | 8.2GB | fp8 ~4GB / GGUF Q4 ~2.5GB (quant-native) |
| Qwen3-VL-4B TE | 2.2GB (fp8) | shared with Krea2 |
| MageVAE | 0.35GB | runs F32 |

`--low-vram-quant` (LLM) keeps GGUF compressed; block-streaming auto-engages for large DiTs. Set `-b cuda`
explicitly if `auto` mispicks.

## D. What to report back
For any failure: model + exact command + the output (or first-token/first-pixels). Every fix is writable
without a GPU — the code is all in `Qwen35Model` (LLM) or `Mage*` (image). Green results → flip the relevant
catalog `Status` to `ok` and record the session in the matching `docs/Checklists/MODEL_STATUS_*.md`.
