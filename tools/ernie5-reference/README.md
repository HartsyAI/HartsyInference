# ERNIE-5.5 per-layer reference comparison

Offline check of the engine's ERNIE-5.5 residual stream against an independent plain-torch reference, run on the real
checkpoint on a rented pod. Nothing here runs on the engine's inference path. The engine only gains an env-gated dump
(`ERNIE_DUMP_LAYERS`, `src/HartsyInference.LLM/Ernie5/Ernie5LayerDump.cs`).

| File | Role |
|---|---|
| `reference.py` | CPU torch forward, one layer at a time. Writes the stream after every decoder layer (`layer_LL_all.npy`, `layer_LL_last.npy`). |
| `compare.py` | Loads the reference `.npy` and the engine `.f32` dumps, stitching the engine's prefill and decode-step files by position. Reports per layer: max abs diff, relative error, cosine, the prefill block's cosine, and the worst position's cosine. Flags the first layer below cosine 0.999 overall or at any position. `--strict` also fails on a layer-set difference, a gap or overlap in the engine's positions, an engine dump shorter than the reference, a fed-token mismatch (`tokens.json` against `meta.json`) or a last-argmax mismatch; `--expect-layers N` requires exactly layers 0..N-1 on both sides and `--expect-tokens N` a reference over N positions. `--tie-margin M` is a diagnostic that leaves indexer and MoE-router near-tie positions out of the per-position check, and `--explain` lists each position's first drop with the smallest top-k gaps at or before it (see Long-input parity). |
| `test_compare.py` | `python -I tools/ernie5-reference/test_compare.py`: the strict layer-set checks, decode stitching (pass, gap, overlap, short engine), the token and argmax checks, and a single bad position being flagged and named. |
| `../parity/compressed_kv.py` | Shared with any model that has the DeepSeek-V4 compressor: pooling with the APE score bias, the RMS norm, each latent's RoPE position and which queries see it. Tests: `python -I tools/parity/test_compressed_kv.py`. |
| `../parity/quant_sim.py` | Shared Hadamard rotation and FP8 E4M3 round trip with UE8M0 scales per 128 values, for indexer queries and keys. Tests: `python -I tools/parity/test_quant_sim.py`. |

Sources the reference follows (fetched for this change, not from the engine):
- `baidu/vLLM-ERNIE` at `b8414da6b703579814c2009c5a63bac101916bbe`, `src/vllm_ernie/models/ernie5_5/` (`nvidia/model.py`,
  `nvidia/attention.py`, `nvidia/moe.py`, `mhc.py`, `mhc_kernels.py`, `nvidia/quant_config.py`, `config.py`).
- `vllm-project/vllm` at tag `v0.32.0` (the version `vllm-ernie` pins): `kernels/mhc/torch.py`, `kernels/mhc/tilelang_kernels.py`
  (hc head), `models/deepseek_v4/attention.py`, `compressor.py`, `common/rope.py`, `layers/rotary_embedding/*` (YaRN),
  `layers/activation.py` (SiTU).
- `deepseek-ai/DeepSeek-V4-Flash` at `60d8d70770c6776ff598c94bb586a859a38244f1`, `inference/model.py` (`Compressor`,
  `get_compress_topk_idxs`, `rotate_activation`) and `inference/kernel.py` (`act_quant`): the plain-torch form of the compressor that
  vLLM's `DeepseekCompressor` runs in Triton, which vLLM-ERNIE's `Ernie5_5Compressor` subclasses unchanged.
- Checkpoint: `baidu/ERNIE-5.5-Flash-Base` at `028544b2ffceea189b2d8768ed2b9d09b06f2343`. The key names are the engine's
  `Ernie5WeightNames`, which the engine already maps against the real index.

## What the reference covers

- Embedding broadcast to `hc_mult` streams; mHC pre (ERNIE's `1/(sqrt(mean)+eps)` RMS factor), mHC post, sinkhorn.
- DSA layers (`compress_ratios <= 0`): non-absorbed MLA, the lightning indexer (top-k over keys before the sliding window, with
  the queries and keys Hadamard-rotated and rounded through FP8 as vLLM-ERNIE stores them; `--indexer-quant f32` turns that off),
  the attention sink (denominator only), VHA, the sigmoid gate, `o_proj`.
- HCA layers (`compress_ratios > 0`) at any length: q/kv projections and norms, per-head query RMS, RoPE at
  `compress_rope_theta` with the real config's YaRN frequencies, the window rows, the compressor (one latent per completed group
  of `compress_ratio` tokens: softmax gate with the APE bias, RMS norm, RoPE at the group's first position, visible from the
  group's last token on), the sink, VHA (BF16 fold), inverse RoPE, the grouped `wo_a`, the gate, `wo_b`.
- Dense SiTU MLP and latent MoE (split-feature routing, learned per-expert scale, latent norm, `fc1`/`fc2`, shared expert).
- FP8 128x128 block dequantization to F32, one tensor at a time. Experts are read only when a token selects them.
- Final hc head read-out, final RMS norm, and optionally the LM head with multimax (`--logits`).

## Not covered (the reference says so or refuses)

- **Compressor storage precision.** DeepSeek's plain-torch source casts each pooled latent to bf16 before the norm and rounds the
  non-rope dims through FP8 after RoPE (`act_quant`, simulating the FP8 cache). The reference skips both and keeps latents in F32,
  as the engine's host path does. Overlapping windows (DeepSeek-V4's ratio-4 layers) are refused; ERNIE has none.
- **Indexer ReLU and scoring.** The DeepGEMM score kernels are not in the sources. The reference uses the standard form
  `sum_h w_h * relu(q_h . k)`. The indexer scores the keys before the sliding window (128), but keeps all of them until there are more
  than `index_topk` (2048), so its selection changes the result only past 2,176 tokens.
- **Indexer FP8 in the reference is from vLLM's call sites, not its kernels.** `per_token_group_quant_fp8(q, 128, use_ue8m0=True)` and
  `SparseAttnIndexer(..., 128, "ue8m0")` give the format. `hadacore_transform` is orthonormal (vllm `tests/kernels/quantization/test_hadacore.py`
  checks it against the Sylvester matrix over sqrt(n)); its bf16 output and the kernel's FP8 rounding mode are not simulated. A near
  tie between the last kept and the first dropped key can still pick differently from a GPU kernel (see Long-input parity). Without
  the FP8 rounding, one of the 9 selecting queries in the small dry run below (position 31) picked a different key than the engine.
- **MTP layer** (`num_nextn_predict_layers = 1`). Not run. The reference stops at `num_hidden_layers`.
- **Shared-expert combine order.** The reference adds the shared expert after the latent `fc2` of the routed branch. That is
  read from the `routed_output_transform` naming, not from a line of the source.
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

The dry run proves that the pipeline runs end to end and that the engine and the reference agree on this synthetic model. It does
not prove the real weights match. The long-input dry runs are under Local dry run below.

## Pod commands

Assumes the pod setup in `docs/Checklists/ERNIE55_POD_RUNBOOK.md` sections 0-4 (license confirmed, volume mounted,
checkpoint downloaded and verified). Use the same branch on the pod. Set `RUN` to a new folder on the volume.

```bash
set -o pipefail                                      # a failing step fails the pipeline instead of tee's exit 0
export CKPT=/workspace/ERNIE-5.5-Flash-Base          # checkpoint: config.json, model.safetensors.index.json, 128 shards
export TOKENS=1,2,3,4,5                              # the same ids the engine test feeds (ERNIE_REAL_TOKENS default)
export RUN=/workspace/ernie-ref-$(date +%Y%m%d-%H%M)
mkdir -p "$RUN"

# 1. Reference (CPU, F32). Writes $RUN/reference/layer_LL_all.npy, layer_LL_last.npy, final_hidden.npy, meta.json.
python tools/ernie5-reference/reference.py --ckpt "$CKPT" --tokens "$TOKENS" --out "$RUN/reference" --logits 2>&1 | tee "$RUN/reference.log"

# 2. Engine dump on the same tokens (CUDA, the runbook's test with the dump on). Writes $RUN/engine/layerLL_pos0_t5_hc4_h4096.f32.
#    Filter on the one test: the class's Text-service test also runs ForwardStream on another 5-token prompt and would overwrite
#    the pos0_t5 dumps.
ERNIE_DUMP_LAYERS="$RUN/engine" ERNIE_REAL_DIR="$CKPT" ERNIE_REAL_DEVICE=cuda ERNIE_REAL_TOKENS="$TOKENS" \
  dotnet test tests/HartsyInference.Engine.Tests --filter "FullyQualifiedName~Ernie5RealCheckpointTests.Real_PrefillAndDecode_ProduceFiniteLogits" \
  --logger "console;verbosity=normal" 2>&1 | tee "$RUN/engine.log"

# 3. Compare. Exit 1 on the first layer below cosine 0.999, or when either folder is not exactly layers 0..42.
python tools/ernie5-reference/compare.py --ref "$RUN/reference" --engine "$RUN/engine" --threshold 0.999 --strict \
  --expect-layers 43 2>&1 | tee "$RUN/compare.txt"
```

Check the engine log before trusting step 3: `Real_PrefillAndDecode_ProduceFiniteLogits` must pass with no `SKIPPED` line, and the
dump folder must hold `layerLL_pos0_t5_*.f32` for every layer. A run without these is not a validation.

`ERNIE_REAL_DEVICE=cpu` is valid too, for a run without a GPU. The engine's CPU path is F32 throughout, so it should sit
closer to the reference than CUDA does; use it to separate engine precision from a real divergence (not measured on the real weights).

### Decode parity

The reference is a causal forward over every position, so feeding it the prompt plus the decoded tokens gives the stream the engine
must reach at each decode step through its KV caches. The harness feeds fixed decode tokens (`ERNIE_REAL_DECODE_FEED`), writes one dump
file per forward call (`pos0_t5`, then `pos5_t1`, `pos6_t1`, ...) and `tokens.json` (every fed token and the argmax after the last).
`compare.py` stitches the files by position, checks the tokens against the reference's `meta.json`, applies the threshold to every
position, and with `--logits` checks the last argmax.

```bash
export FEED=2551,7479,2811,1230,2810,4357,1298,36326   # any ids work; these are the 2026-11-16 run's (see below)
ERNIE_DUMP_LAYERS="$RUN/engine-decode" ERNIE_REAL_DIR="$CKPT" ERNIE_REAL_DEVICE=cuda ERNIE_REAL_RESIDENT=1 ERNIE_REAL_TOKENS="$TOKENS" \
  ERNIE_REAL_DECODE_FEED="$FEED" dotnet test tests/HartsyInference.Engine.Tests \
  --filter "FullyQualifiedName~Ernie5RealCheckpointTests.Real_PrefillAndDecode_ProduceFiniteLogits" \
  --logger "console;verbosity=normal" 2>&1 | tee "$RUN/engine-decode.log"
python tools/ernie5-reference/reference.py --ckpt "$CKPT" --tokens "$TOKENS,$FEED" --out "$RUN/reference-decode" --logits 2>&1 \
  | tee "$RUN/reference-decode.log"
python tools/ernie5-reference/compare.py --ref "$RUN/reference-decode" --engine "$RUN/engine-decode" --threshold 0.999 --strict \
  --expect-layers 43 --expect-tokens 13 2>&1 | tee "$RUN/compare-decode.txt"
```

Use a fresh `ERNIE_DUMP_LAYERS` folder per run: the dump overwrites files of the same name but never clears the folder, so a shorter run
into a used folder leaves the longer run's decode files behind (`compare.py` refuses most such mixes, but not every one).

Every position must reach the threshold, not only the whole stream. That applies to the prefill-only command above too, which before
checked only the whole-stream cosine. One position holds only `hc*hidden` values; `--position-threshold` sets a looser bound for single
positions without loosening the whole-stream check.

The fed ids need not be the engine's greedy choices: the reference runs on the same ids, so the check holds for any sequence. The ids
above are a 16-step greedy run's outputs. That run's own first decode input was the prefill argmax, 1298, which the feed leaves out, so
every fed id sits one position early against the greedy path and the engine's argmax agrees with 6 of the 8.

What it covers: decode through the HCA and DSA KV caches, RoPE at offsets, the one-token MoE path, and the head's argmax. With 13
positions it does not reach the HCA compressor, whose first 128-token group completes at the 128th token, or the DSA indexer's key
selection, which drops keys only past 2,176 tokens (`sliding_window` 128 plus `index_topk` 2048); the long-input run below does. MTP is
not covered.

Real checkpoint, 2026-11-16 (RTX 4090, `580cd2a7d`, prompt `1,2,3,4,5` plus the feed above, 13 positions): all 43 layers reach
cosine >= 0.999 at every position (printed 1.00000000 everywhere; relative error 1.6e-6 to 9.1e-6, max abs 2.8e-2 at layer 42), and the argmax
after the last position is 1232 on both sides. The engine took 150 s to load, 33 s for the prefill and 4.1 to 7.1 s per decode step; the
CPU reference took 238 s.

Local dry run on the real-dimension synthetic checkpoint (CPU engine, 5 prompt tokens plus feed `6,7,8`): all 3 layers have cosine
1.0000 at every position (relative error 4.9e-6 to 2.0e-5), and the argmax after the last position is 781 on both sides. With one
wrong token in the reference (`...,6,9,8`), the comparison fails: every layer is flagged from position 6, and the token and argmax
checks both fail.

### Long-input parity

The same decode check with a prompt long enough for both long-context paths: 2,300 prompt tokens complete 17 compressor groups in
every HCA layer and put every DSA query past position 2,175 into top-k selection; the 8 fed tokens then cross the group boundary at
position 2,303 during decode. Any ids work, as above.

```bash
export LONG=$(python -c "print(','.join(str(1000 + (37 * i) % 90000) for i in range(2300)))")
export FEED=2551,7479,2811,1230,2810,4357,1298,36326
ERNIE_DUMP_LAYERS="$RUN/engine-long" ERNIE_REAL_DIR="$CKPT" ERNIE_REAL_DEVICE=cuda ERNIE_REAL_RESIDENT=1 ERNIE_REAL_TOKENS="$LONG" \
  ERNIE_REAL_DECODE_FEED="$FEED" dotnet test tests/HartsyInference.Engine.Tests \
  --filter "FullyQualifiedName~Ernie5RealCheckpointTests.Real_PrefillAndDecode_ProduceFiniteLogits" \
  --logger "console;verbosity=normal" 2>&1 | tee "$RUN/engine-long.log"
python tools/ernie5-reference/reference.py --ckpt "$CKPT" --tokens "$LONG,$FEED" --out "$RUN/reference-long" --logits 2>&1 \
  | tee "$RUN/reference-long.log"
python tools/ernie5-reference/compare.py --ref "$RUN/reference-long" --engine "$RUN/engine-long" --threshold 0.999 --strict \
  --expect-layers 43 --expect-tokens 2308 2>&1 | tee "$RUN/compare-long.txt"
```

The strict run is the check. If it flags only a few single positions, rerun the comparison with `--explain`: it lists each position
that falls below the per-position threshold at the first layer it does, with that position's smallest top-k gaps at or before that
layer. Two top-k choices can flip under rounding alone: the DSA indexer's keys (`index_gaps.json`) and the MoE router's experts
(`router_gaps.json`, every MoE token), each gap measured as a fraction of that choice's candidate-score spread. A first drop whose
smallest gap is among the smallest few percent of its layer is a near tie; one with no small gap is a real difference.

`--tie-margin M` then leaves the positions with a gap below M out of the per-position check from that layer on, prints them per layer,
keeps them in the whole-stream cosine, and is an error when more than 5 percent of either file's positions would be left out
(`--max-tie-fraction`). Only the tie positions themselves are left out; from the next layer on, a later position reads their changed
state, so it can still be flagged, and the note then names the nearest earlier near tie from an earlier layer (`after near-tie at N`).

The reference reads nearly every routed expert at this length, so it makes about one full pass over the checkpoint; its indexer scores
are one `[2308, 64, 2308]` F32 tensor per DSA layer (about 1.4 GB).

### Expected behaviour on the real checkpoint

- Step 1 runs layers 0 to 42 at any length. A layer that uses an unsupported rope type raises `NotImplementedError` with the
  reason.
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
  dotnet test tests/HartsyInference.Engine.Tests --filter "FullyQualifiedName~Ernie5RealCheckpointTests.Real_PrefillAndDecode_ProduceFiniteLogits"
python tools/ernie5-reference/compare.py --ref /tmp/ref_synth --engine /tmp/eng_synth --strict
```

(On Windows, use a folder under the scratchpad and `cygpath -m` paths for Python.) The synthetic checkpoint is about 0.9 GB.

`ERNIE_WRITE_SYNTHETIC_OVERRIDES` adds config fields on top of the real dimensions (`key=json` pairs split by `;`). A small ratio,
window and top-k reach the compressor and the indexer's selection in a few dozen tokens:

```bash
ERNIE_WRITE_SYNTHETIC_DIR=/tmp/ernie_small ERNIE_WRITE_SYNTHETIC_OVERRIDES="compress_ratios=[8, -2, 8];sliding_window=8;index_topk=16" \
  dotnet test tests/HartsyInference.Engine.Tests --filter "FullyQualifiedName~Ernie5ReferenceFixtureTests"
```

Long-input dry runs (CPU engine, 2026-11-16):

- Ratio 8, window 8, top-k 16; 21 prompt tokens plus 12 fed tokens (33 positions, so 4 groups complete, one of them during decode,
  and DSA selection runs at positions 24 to 32): all 3 layers reach cosine 0.99999995 or more at every position, and the argmax
  after the last position is 3372 on both sides. With `--indexer-quant f32` the DSA layer is flagged at position 31 (cosine 0.9951),
  where its 16th and 17th indexer candidates are 8 percent apart.
- Real dimensions (ratio 128, window 128, top-k 64); 250 prompt tokens plus 12 fed tokens (262 positions: one group completes in the
  prefill and one at position 255 during decode, and DSA selection runs from position 192): the HCA layer before the DSA layer
  (layer 0) reaches cosine 1.00000000 at every position. Strict, the DSA layer is flagged at 2 of its 70 selecting positions: at
  225 and 248 the engine kept a different key, and the cosine of that layer's whole contribution there (attention and MoE) is 0.9965
  and 0.9949. Those two queries' last kept and first dropped keys are 2.6e-5 and 1.8e-5 of the candidate-score spread apart, the
  two smallest gaps of all 70. With `--tie-margin 1e-4` they and one more near tie (259) are left out, and every other position
  reaches 0.9998 or more. The argmax after the last position is 1965 on both sides. The CPU engine took 10 minutes, the reference 17 s.
- Real dimensions with window 8 and top-k 120, so each selecting query keeps nearly all of its candidates, as the real checkpoint's
  2048 of about 2,180 does; 132 prompt tokens plus 8 fed tokens (140 positions, selection from position 128): strict, all 3 layers
  reach cosine 0.99999996 or more at every position, and the argmax after the last position is 3625 on both sides. None of the 12
  selecting queries is a near tie at 1e-4 (the smallest gap is 1.1e-3 of the spread). The CPU engine took 3 minutes.
