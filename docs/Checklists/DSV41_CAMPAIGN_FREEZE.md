# DeepSeek-V4.1-Flash: campaign freeze

Frozen 2026-10-09, before any rented run. A gate changes only by a dated entry under [Changes](#changes), with the
measured evidence that forced the change. Nothing here is loosened to make a run pass. The runner
(`tests/dsv41-certification.sh`) enforces the acceptance rules. The values below are read from the test sources named in
each table.

## Pin

- Checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash` at revision `dba1be0a40aa45a94ad051997016db3960a90277`
  (`tests/python-reference/deepseek_v41/fetch_upstream.py`).
- Shards: 48 files totalling 510,296,708,312 bytes, measured on the home RAID on 2026-10-09.
- Oracle: the unmodified upstream `model.py` in float32, through `dump_real_layers.py`, plus the ports envelope.

## Layer gates

From `tests/HartsyInference.LLM.Tests/DeepSeekV41/DeepSeekV41RealWeightsTests.cs`:

| Gate | Value | Applies to | Provenance |
|---|---|---|---|
| Exact hidden relative L2 | at most 2e-3 | exact mode, every layer | The source says these gates were fixed before the first comparison (layer 0). Measured on layer 0: 4.3e-6 (#302). |
| Exact logits cosine | at least 0.9999 | exact mode | Same as above. |
| Ports cosine floor | at least 0.95 | ports mode | Same as above. |
| Exact block-output relative L2 | at most 2e-3 | exact mode, every block output and tapped sublayer | Source: "gated separately". Measured on layer 0: 3.1e-6 (#302). |
| Exact logits relative L2 | at most 2e-3 | exact mode | Source: added after the first run as a regression guard. Measured on layer 0: 2.8e-6. |
| Exact top-10 overlap | at least 9 | exact mode | Source: added after the first run as a regression guard (measured overlap 10). |
| Ports relative L2 ceiling | at most 0.2 | ports mode, one layer; deeper ports runs only check finite values | Envelope, about 10 times the measured 2e-2. |
| Quantized block and hidden relative L2 | at most 5e-2 | quantized exact mode at depth | Source: fixed before the first deep run, after a 4-layer run showed the cause. |
| Quantized logits cosine | at least 0.999 | quantized exact mode at depth | Same as above. |
| Quantized top-10 overlap | at least 8 | quantized exact mode at depth | Same as above. |
| Near-tie logit gap | 0.05 | argmax disagreements | Set with the structural amendment below. |
| Near-tie routing margin | 1e-5 | routing near-ties | Set with the structural amendment below. |
| Slice dequantization | exact, maximum absolute difference 0 | `RealTensorSlices_DequantizeLikeTheReference` | Bit-exact against the reference dump. |
| Full-depth run | finite logits | `FullDepth_PrefillGivesFiniteLogitsAndGreedyTokens`, which needs `DSV41_FULL_RUN=1` | The 40-layer structural run (#312) is the campaign's full-depth evidence. |

**Structural amendment (disclosed).** The structural gate was amended after the first 1300-token run failed the cache gate.
The cause was routing near-ties: the 6th and 7th biased gate scores of a token sat within float noise, so two correct
implementations chose different experts. The check is strict up to the first layer that holds a near-tie. At that layer,
every token without a near-tie must still match strictly, and the tie tokens are reported. Deeper layers and the final
outputs then use the flip-aware gates above. This is an amendment made after a failure. It is recorded here as one, not
as a gate that was set in advance.

## DSpark gates

From `DeepSeekV41DSparkTests.cs` and `DeepSeekV41DSparkChainTests.cs`:

| Gate | Value | Applies to | Provenance |
|---|---|---|---|
| Stage relative L2 | at most 2e-3 | every draft stage: attention, feed-forward and output | The test header states it. Measured on the 160-token oracle: at most 8e-6 (#326). |
| Draft logits relative L2 | at most 2e-3 | draft logits | Same as above. |
| Confidence maximum absolute difference | at most 2e-3 | draft confidences | Same as above. |
| Target tap relative L2 | at most 2e-3 | prefill and decode `main_hidden` taps | The chain test states the gates were fixed before the first comparison. |
| Draft ids | exact match | the draft chain | Exact match against the oracle. Measured: exact past the window (#326). |

## Workloads

- **Layer oracle.** Default ids `0,671,6102,294,8760,344`: BOS plus the default prompt. 40 layers, structural mode,
  16 teacher-forced decode steps (#312).
- **DSpark oracle.** `dump_real_dspark.py --max-prompt-tokens 160`.
- **Vision oracle.** Grids 28x28 and 17x23, seed 20260930, the defaults of `dump_real_vision.py`.
- **Not in the runner.** The 1300-token structural run that found the near-tie is not a runner class. Add it as a dated
  workload here before it counts as a gate.
- **Not written at all.** The plan's Phase 13 workloads (100k to 1M context, multi-image, tools, reasoning, concurrency,
  cancellation, rank failure, restart, leak detection) do not exist yet. This certification does not claim them.

## Acceptance rules

1. A class is green only when at least one test ran, none failed, none were skipped, and the output contains no line
   reading SKIPPED. The runner enforces this. A test that returns early without running is a failure.
2. Every run sets `HARTSY_REQUIRE_REAL_WEIGHTS=1`, so a missing asset fails its class.
3. A blocked lane is not green. Missing hardware, unbuilt features and simulated ranks are reported, not counted as passes.
4. Numerical correctness, quantization quality, performance and hardware coverage are reported separately.
5. Throughput, time to first token, memory and transfer figures are measured and reported. This file freezes no target.
   A target may be added only after a measured baseline is recorded below.

## Not in the runner

- `DeepSeekV41CheckpointTests`: reads `HARTSY_DSV41_HEADER_REPLICA`, and no replica is defined yet.
- `ShardSetHeaderOnlyTests` (ModelAssets): not specific to V4.1.
- The plan's lanes for PRs 20 to 24b (multi-GPU, two-node, offload, Vulkan on AMD) are BLOCKED in the runner.

## Known open items at freeze

- The target's decode tap at position 160 on the 160-token DSpark run has relative L2 1.24e-3. It passes the 2e-3 gate.
  The cause is not explained. A routing near-tie is the leading candidate, and it has not been checked at that position.
- V4.1 GPU kernels and bf16 numerics are not verified. The CPU oracle does not cover them.

## Measured baseline (before this freeze)

| Quantity | Value | Source |
|---|---|---|
| Layer 0, exact mode: block output / hidden / logits relative L2 | 3.1e-6 / 4.3e-6 / 2.8e-6 | #302 |
| 40-layer structural run, prefill and 16 teacher-forced steps | argmax equal at every position | #312 |
| 160-token DSpark, stage relative L2 | at most 8e-6; draft ids exact past the window | #326 |

## Changes

- 2026-10-09: frozen, first version.
