# ERNIE-5.5 per-layer reference comparison

Offline check of the engine's ERNIE-5.5 residual stream against an independent plain-torch reference, run on the real
checkpoint on a rented pod. Nothing here runs on the engine's inference path. The engine only gains an env-gated dump
(`ERNIE_DUMP_LAYERS`, `src/HartsyInference.LLM/Ernie5/Ernie5LayerDump.cs`).

| File | Role |
|---|---|
| `reference.py` | CPU torch forward, one layer at a time. Writes the stream after every decoder layer (`layer_LL_all.npy`, `layer_LL_last.npy`). |
| `compare.py` | Loads the reference `.npy` and the engine `.f32` dumps. Reports per layer: max abs diff, relative error, cosine, last-token cosine. Flags the first layer below cosine 0.999. |

Sources the reference follows (fetched for this change, not from the engine):
- `baidu/vLLM-ERNIE` at `b8414da6b703579814c2009c5a63bac101916bbe`, `src/vllm_ernie/models/ernie5_5/` (`nvidia/model.py`,
  `nvidia/attention.py`, `nvidia/moe.py`, `mhc.py`, `mhc_kernels.py`, `nvidia/quant_config.py`, `config.py`).
- `vllm-project/vllm` at tag `v0.32.0` (the version `vllm-ernie` pins): `kernels/mhc/torch.py`, `kernels/mhc/tilelang_kernels.py`
  (hc head), `models/deepseek_v4/attention.py`, `compressor.py`, `common/rope.py`, `layers/rotary_embedding/*` (YaRN),
  `layers/activation.py` (SiTU).
- Checkpoint: `baidu/ERNIE-5.5-Flash-Base` at `028544b2ffceea189b2d8768ed2b9d09b06f2343`. The key names are the engine's
  `Ernie5WeightNames`, which the engine already maps against the real index.

## What the reference covers

- Embedding broadcast to `hc_mult` streams; mHC pre (ERNIE's `1/(sqrt(mean)+eps)` RMS factor), mHC post, sinkhorn.
- DSA layers (`compress_ratios <= 0`): non-absorbed MLA, the lightning indexer (top-k over keys before the sliding window),
  the attention sink (denominator only), VHA, the sigmoid gate, `o_proj`.
- HCA layers (`compress_ratios > 0`) while the sequence is shorter than the ratio: q/kv projections and norms, per-head query
  RMS, RoPE at `compress_rope_theta` with the real config's YaRN frequencies, the sink, VHA (BF16 fold), inverse RoPE, the
  grouped `wo_a`, the gate, `wo_b`.
- Dense SiTU MLP and latent MoE (split-feature routing, learned per-expert scale, latent norm, `fc1`/`fc2`, shared expert).
- FP8 128x128 block dequantization to F32, one tensor at a time. Experts are read only when a token selects them.
- Final hc head read-out, final RMS norm, and optionally the LM head with multimax (`--logits`).

## Not covered (the reference says so or refuses)

- **HCA compressor.** Pooling, gating and the APE row index are not in the sources. `reference.py` raises
  `NotImplementedError` once a block of `compress_ratio` tokens completes, i.e. at token 128 for the real checkpoint. Any run
  of 127 tokens or fewer is unaffected, and the pod run feeds 5 tokens.
- **Indexer ReLU and scoring.** The DeepGEMM score kernels are not in the sources. The reference uses the standard form
  `sum_h w_h * relu(q_h . k)`. The indexer only selects keys once a sequence is longer than `sliding_window` (128), so the
  5-token pod run does not exercise it.
- **MTP layer** (`num_nextn_predict_layers = 1`). Not run. The reference stops at `num_hidden_layers`.
- **Shared-expert combine order.** The reference adds the shared expert after the latent `fc2` of the routed branch. That is
  read from the `routed_output_transform` naming, not from a line of the source.
- **Indexer Hadamard and FP8 quantization.** Skipped. An orthonormal transform on both q and k leaves q.k unchanged, and the
  scores are kept in F32.
- **Engine precision.** The reference is F32, apart from the BF16 VHA fold (see below). The engine's CUDA GEMMs default to
  TF32 and its Vulkan GEMMs round inputs to F16. The pod run pins the engine's `reference` knob profile, as
  `Ernie5RealCheckpointTests` does.

## The VHA fold is rounded to BF16

vLLM computes `m = I + V U^T` with `torch.addmm` in the parameter dtype (`VHAPostMix.fold`). A bf16 model folds in BF16, so
the reference rounds `m` to BF16 too. The engine's `FoldVha` does the same (`roundToBf16: true`). Without the rounding, the
synthetic layer-0 difference was about 0.5 percent relative. With it, the difference is about 1e-5 (dry run below).

## Dry run (done, CPU, no pod)

Real-dimension synthetic checkpoint (`Ernie5SyntheticCheckpoint.RealDimensionOverrides`, FP8 storage, 3 shards, layers HCA /
DSA / HCA, 8 routed experts, 4096-token vocab) written by `Ernie5ReferenceFixtureTests` (gated on `ERNIE_WRITE_SYNTHETIC_DIR`).
Tokens `1,2,3,4,5`.

```
layer       max_abs       rel_err        cosine      last_cos      last_rel
    0      0.122925   5.89739e-06    1.00000000    1.00000000   7.95069e-06
    1      0.199219   8.00168e-06    1.00000000    1.00000000   1.07834e-05
    2      0.380859    1.3858e-05    1.00000000    1.00000000   1.71563e-05
all 3 layers have cosine >= 0.999
```

The reference refuses at 130 tokens (`NotImplementedError: HCA compressor ...`), as designed. The dry run proves that the
pipeline runs end to end and that the engine and the reference agree on this synthetic model. It does not prove the real
weights match.

## Pod commands

Assumes the pod setup in `docs/Checklists/ERNIE55_POD_RUNBOOK.md` sections 0-4 (license confirmed, volume mounted,
checkpoint downloaded and verified). Use the same branch on the pod. Set `RUN` to a new folder on the volume.

```bash
export CKPT=/workspace/ERNIE-5.5-Flash-Base          # checkpoint: config.json, model.safetensors.index.json, 128 shards
export TOKENS=1,2,3,4,5                              # the same ids the engine test feeds (ERNIE_REAL_TOKENS default)
export RUN=/workspace/ernie-ref-$(date +%Y%m%d-%H%M)
mkdir -p "$RUN"

# 1. Reference (CPU, F32). Writes $RUN/reference/layer_LL_all.npy, layer_LL_last.npy, final_hidden.npy, meta.json.
python tools/ernie5-reference/reference.py --ckpt "$CKPT" --tokens "$TOKENS" --out "$RUN/reference" --logits 2>&1 | tee "$RUN/reference.log"

# 2. Engine dump on the same tokens (CUDA, the runbook's test with the dump on). Writes $RUN/engine/layerLL_pos0_t5_hc4_h4096.f32.
ERNIE_DUMP_LAYERS="$RUN/engine" ERNIE_REAL_DIR="$CKPT" ERNIE_REAL_DEVICE=cuda ERNIE_REAL_TOKENS="$TOKENS" \
  dotnet test tests/HartsyInference.Engine.Tests --filter "FullyQualifiedName~Ernie5RealCheckpointTests" \
  --logger "console;verbosity=normal" 2>&1 | tee "$RUN/engine.log"

# 3. Compare. Exit 1 on the first layer below cosine 0.999 (or a missing layer).
python tools/ernie5-reference/compare.py --ref "$RUN/reference" --engine "$RUN/engine" --threshold 0.999 --strict \
  2>&1 | tee "$RUN/compare.txt"
```

Check the engine log before trusting step 3: `Real_PrefillAndDecode_ProduceFiniteLogits` must pass with no `SKIPPED` line, and the
dump folder must hold `layerLL_pos0_t5_*.f32` for every layer. A run without these is not a validation.

`ERNIE_REAL_DEVICE=cpu` is valid too, for a run without a GPU. The engine's CPU path is F32 throughout, so it should sit
closer to the reference than CUDA does; use it to separate engine precision from a real divergence (not measured on the real weights).

### Expected behaviour on the real checkpoint

- Step 1 refuses only if the run is longer than 127 tokens. For 5 tokens it runs layers 0 to 42. A layer that uses an
  unsupported rope type raises `NotImplementedError` with the reason.
- Step 2 is the existing runbook test, with `ERNIE_DUMP_LAYERS` set; the runbook's VRAM and timing expectations apply.
- Step 3 reports one row per layer. A first flagged layer is the place to look; the layer types are in
  `config.json` (`compress_ratios`, `-2` for DSA, `128` for HCA).

### Time and memory (estimates, not measured)

- Reference: reads each layer's weights from the volume once, so about one pass over the 288 GB checkpoint. On a network volume
  that is tens of minutes to a couple of hours. The per-token Python loops are short for 5 tokens. Estimate only.
- Reference host memory: the largest tensors are the embedding and the LM head, each about 3.3 GB as F32. Attention and
  expert tensors are dequantized one at a time. Expect well under 10 GB. Estimate only; record the peak with
  `/usr/bin/time -v python tools/ernie5-reference/reference.py ...` if the number matters.
- Engine dump: one prefill and one decode, as in the runbook.

## Local dry run (no pod)

```bash
# write the synthetic checkpoint (needs a built Engine.Tests)
ERNIE_WRITE_SYNTHETIC_DIR=/tmp/ernie_synth dotnet test tests/HartsyInference.Engine.Tests --filter "FullyQualifiedName~Ernie5ReferenceFixtureTests"
python tools/ernie5-reference/reference.py --ckpt /tmp/ernie_synth --tokens 1,2,3,4,5 --out /tmp/ref_synth --logits
ERNIE_DUMP_LAYERS=/tmp/eng_synth ERNIE_REAL_DIR=/tmp/ernie_synth ERNIE_REAL_DEVICE=cpu ERNIE_REAL_TOKENS=1,2,3,4,5 \
  dotnet test tests/HartsyInference.Engine.Tests --filter "FullyQualifiedName~Ernie5RealCheckpointTests"
python tools/ernie5-reference/compare.py --ref /tmp/ref_synth --engine /tmp/eng_synth --strict
```

(On Windows, use a folder under the scratchpad and `cygpath -m` paths for Python.) The synthetic checkpoint is about 0.9 GB.
