# Attention contract

The semantics every attention kernel, planner row and model-side layer in this engine implements. It is normative:
`AttentionHostReference` (`src/HartsyInference.Core/Backends/Attention/`, invoked explicitly by the caller) is its
executable form, and a kernel that disagrees with it beyond the tolerance of the precision it was permitted is wrong,
whatever any model's output looks like. Code comments point here instead of restating it.

The [support and admission contract](ATTENTION_SUPPORT.md) distinguishes mandatory common cases, optional
backend capabilities and current implementation gaps. Normative semantics do not assert that every feature is
represented by the current API or already verified on every backend.

## 1. Operands and shapes

One call computes, for every batch `b`, query head `h` and query row `i`:

```
o[b,h,i,:] = Σ_j p[b,h,i,j] · v[b, g(h), j, :]
```

| Symbol | Meaning |
|---|---|
| `B` | batch |
| `Hq`, `Hkv` | query heads and key/value heads; `Hq % Hkv == 0`, otherwise the call is invalid |
| `g(h)` | the key/value head query head `h` reads: `h / (Hq / Hkv)` |
| `Sq`, `Skv` | query rows and key rows present in the operands (allocated extent, not necessarily valid length) |
| `Dqk`, `Dv` | head dims of Q/K and of V/O; they differ for MLA and are equal everywhere else |

Each of Q, K, V and O is a strided view of a tensor: a base offset and four strides `(batch, head, row, dim)` in
**elements** of the view's dtype. Head-major `[B,H,S,D]`, token-major `[B,S,H,D]` and a slice of a fused QKV
projection are all just strides. A call is invalid when any addressed element falls outside its tensor's allocation,
when the dim stride is not 1 for a row a kernel reads vectorized (the row's capability states its alignment), or when
O overlaps Q, K or V. In-place attention is not supported.

## 2. Positions and lengths

Positions are **logical**: the position of query row `i` is `qOrigin + i` and of key row `j` is `kOrigin + j`. A
sequence's valid key length `kvLen[b] ≤ Skv` and valid query length `qLen[b] ≤ Sq` come with the call; rows beyond them
are not read (keys) or not written (queries). An allocation larger than the valid length — a KV cache's capacity, a
prefill bucket — never changes the result.

Causal alignment is bottom-right in logical positions: query row `i` may see key `j` only when
`kOrigin + j ≤ qOrigin + i`. For the common cache layout (`kOrigin = 0`, `qOrigin = kvLen − qLen`) that is the decode
and chunked-prefill rule; for `Sq = Skv`, `qOrigin = kOrigin` it is ordinary square causal attention. Alignment is
computed from the valid lengths, never from `Skv`, a capacity or a bucket size.

KV storage is either **dense** (one contiguous `Skv` per sequence plus `kvLen`) or **paged** (a page table, a page
size and `kvLen`); the two are distinct representations, not one with a flag.

A paged call (`AttentionCall.KvPages`) passes one layer's page buffers as its key and value operands, `[1, Hkv, pages ×
pageSize, D]`, and an I32 page table: row `r` of the sequence is buffer row `table[r / pageSize] × pageSize + r mod
pageSize`. It holds one sequence and states `kvLen` on the host. The owner replaces the table tensor whenever the table
changes and states its highest page, so the call is checked against the buffers without reading the table back. Only a
row that declares `PagedKv` reads such a call. Every other row excludes it, a backend without such a row refuses it,
and the host reference refuses it, because each would take the page buffer for a contiguous cache. A caller asks
`IBackend.AdmitsPagedAttention` first and, when the answer is no, gathers the rows into a contiguous buffer and issues
a dense call: the gather is the caller's, recorded as `PrefixGather`, never a backend's silent choice.

A call may give **a length per sequence** (`AttentionCall.SequenceLengths`) in place of one shared `kvLen`: sequence
`b` attends the first `lengths[b]` rows of its keys, with a length below zero read as zero and one above `Skv` as `Skv`,
and the rows past that are never read. It is then one call per sequence. Under the causal rule the `Sq` query rows are
the sequence's last positions, so row `j` sits at `lengths[b] − Sq + j`; a row below position zero attends nothing.
Without the causal rule every row attends all `lengths[b]` keys. Such a call leaves the shared positions unset and
carries no other position-dependent or additive term: a window, a block size, a bias, a softcap, sinks and ALiBi are
refused, and so are paged keys and values. A call may also name an **LSE output** (`AttentionCall.Lse`, F32
`[B, Hq, Sq]`, §3). Only a row that declares `SequenceLengths`, or `Lse`, takes a call that uses it; every other row
excludes the call, and a GPU backend without such a row refuses it. The host reference runs both, and the CPU backend
takes such a call through it. On CUDA and on Vulkan one row declares both, `flash.decode.lengths`: F32 head-major Q and
the `[B, Sq, Hq × D]` output over a head-major cache. CUDA takes an F32, F16 or BF16 cache, head dim 64 or 128 and at
most 16 query heads per key head; Vulkan an F32 cache, a head dim of one to four subgroup widths and at most 8. Each
takes only calls with sequence lengths, so an LSE on a call with one shared length is still refused on a GPU. Both
kernels cut the keys into chunks sized from the longest sequence of the call, so the order a sequence's keys are summed
in, and with it the last bits of its result, depends on the other sequences' lengths. Each result stays within the
kernel's tolerance of the host reference; it is not bit-equal to the same sequence run alone.

The **speculative-decode merge** is not an `AttentionCall`. It is a backend primitive, `IBackend.DecodeTreeMerge`,
defined by `DecodeMergeReference`, and `IBackend.HasDecodeMerge` says whether a backend runs it; one that does not
refuses the call. A verify step of `S` rows (at most 32) first attends the committed prefix alone, as a decode with
sequence lengths, without the causal rule, that names an LSE. The merge then folds the step's own keys and values into
that result: row `j` attends the step rows its mask entry names, one bit a row, and the two softmaxes are joined through
the prefix LSE, which is negative infinity for a row whose prefix was empty. A row with no key at all is zero. Every
operand is F32: the prefix output and the merged output `[B, S, Hq × D]` (they may be one tensor), the prefix LSE
`[B, Hq, S]`, Q `[B, Hq, S, D]`, and the step's K and V `[B, Hkv, S, D]`. The CPU backend runs the reference; CUDA and
Vulkan each have one kernel. It records no `Execution` evidence and has no planner row, because no kernel is chosen.

A position held on the device (graph decode) that exceeds the storage it indexes is clamped so no kernel reads out
of bounds, and the kernel sets a device status word. The next host synchronization point that owns the call (the end
of a step or of a graph replay) reads it and raises. An out-of-range position is never a successful result.

## 3. Scores

For a visible pair the score is computed in this order; the order is part of the contract:

1. `s = α · (q · k)` with `α` the call's scale. Padding a head dim (40 → 64) never changes `α`.
2. Softcap, when set: `s = c · tanh(s / c)`.
3. Additive terms: `s += bias[b,h,i,j]` (finite, broadcast over any axis with stride 0), and ALiBi: `slope[h] · (kpos − qpos)`
   for a causal call, `−slope[h] · |kpos − qpos|` for a bidirectional one.
4. Hard exclusions remove the pair from the softmax. They are applied after the softcap, so an excluded score can never
   become finite.
5. Sinks, when set, add one extra logit `sink[h]` to the row's softmax denominator with no value.

`p = exp(s − m) / Σ exp(s − m)` over the row's visible pairs and sinks, with `m` the row maximum. LSE, when requested,
is `m + ln Σ exp(s − m)` in natural-log units, sinks included.

A **fully excluded row** (no visible key and no sink) produces `o = 0` and `LSE = −∞`, on every backend.
A `−∞` sink is no sink; a `+∞` sink makes the row's output and LSE NaN.

**NaN propagates.** An output element or LSE whose computation reads a NaN — in Q, K, V, a bias, a sink or a scale — is
NaN. No kernel drops a NaN from a maximum or rounds one into a finite value; a quantized kernel may spread it further
(through a shared scale or mean), never less.

## 4. Masks

A call carries any combination of these; they compose (logical AND of visibility, sum of finite terms):

| Component | Kind | Meaning |
|---|---|---|
| `Causal` | exclusion | §2's rule |
| `Window(left, right)` | exclusion | key visible only when `qpos − left ≤ kpos ≤ qpos + right`; both ends inclusive; `right = 0` with `Causal` is the usual sliding window |
| `KeyValidLength` | exclusion | keys at `j ≥ kvLen[b]` are not visible |
| `SegmentIds` | exclusion | a key is visible only to queries of the same segment (block-diagonal, prefix-LM) |
| `ExclusionTensor` | exclusion | an explicit boolean tensor, broadcast like a bias |
| `BlockSize` | granularity | with `n > 0`, `Causal` and `Window` count blocks of `n` tokens instead of tokens (block-causal) |
| `AdditiveBias` | finite term | a tensor with broadcast strides per axis (per batch, per head, per query row, per key) |

A backend that cannot apply a component a call carries refuses the call; it never drops the component.

A value never turns into an exclusion: `log(1e-4)` and `−1e30` are finite biases and keep their probability mass
exactly as written; only an exclusion component or `−∞` stored in an `AdditiveBias` excludes. Converting a model's
finite sentinel into an exclusion is a numerical change of that model and is declared as one.

## 5. Numerical policy

A call states which approximations it **permits**; a kernel may use an approximation only when permitted. The
permissions are independent — none of them implies another, and they do not form a ladder (BF16 has more range and
less precision than F16; INT8 error depends on scales, grouping and clipping).

| Permission | Values |
|---|---|
| Operand storage | per operand: F32, BF16, F16, INT8 (with a stated scale granularity), FP8 E4M3 |
| QK and PV multiply format | F32, TF32, BF16, F16, INT8, FP8 |
| Accumulation and softmax | F32 required unless a lower format is permitted explicitly |
| Saturating conversion | permits clipping an out-of-range value to the format's maximum. It prevents an overflow and introduces clipping error; it is an approximation, never a correctness fix |
| Quantization | INT8/FP8 for Q/K, for P, for V, each separately, with the smoothing it may use |
| Determinism | bitwise run-to-run on one device and build: required or not |

A call also declares the **bounds** that make a narrowing safe: an element bound per operand and a bound on `|s|`,
derived from the model's geometry, never asserted from a feature's presence. After a per-head RMS norm with gain `γ`
over `D` elements, `|qᵢ| ≤ √D · max|γ|` and `‖q‖₂ ≤ √D · max|γ|`, so `|α q·k| ≤ α · D · max|γ_q| · max|γ_k|`; eps
lowers it, a later scale multiplies it, RoPE preserves it. A bounded score does not make Q or K safe to narrow — their
elements must fit the format themselves — and a bounded V does not bound the PV accumulator, whose bound is
`Skv · max|v|` per row. A permission whose bound is unknown is not granted.

Presets are named sets of permissions; a request preset can only remove permissions from what the call and the model
allow, never add them:

| Preset | Grants |
|---|---|
| `exact` | F32 storage and multiply, TF32 off, no saturation, no quantization, F32 accumulation and softmax. A documented strict computation, still rounded F32 arithmetic: two `exact` kernels need not agree bitwise |
| `bf16` | BF16 operand storage and multiply where the element bounds fit, F32 accumulation |
| `f16` | F16 operand storage and multiply where the element bounds fit, saturation permitted, F32 accumulation |
| `int8` | `f16`, plus INT8 Q/K with per-row or finer scales and K smoothing, plus the P/V format the model's gate selected |

A call may also refuse a multiply format (`AttentionNumerics.Refused`, per layer `AttentionLayerConfig.RefusedFormats`): every
row that declares that format is excluded, whatever the call grants or asserts. An operand or the output in a refused format
excludes every row, since a row that widens it still reads it narrow. `AttentionNumerics.WideRange` is the F32-or-BF16 demand
of a model whose values overflow F16.

K smoothing (subtracting one vector `μ` from every key) shifts each query row's scores by `cᵢ = α qᵢ · μ`. Plain
softmax, additive bias and exclusions are invariant to that shift; a softcap is not (`tanh(x − c)` is not a shift of
`tanh(x)`), a sink logit must be shifted by the same `cᵢ`, the returned LSE must add `cᵢ` back, and two segments
smoothed with different vectors need per-segment corrections. A kernel that smooths implements those corrections or
refuses the call. A stale smoothing vector is invariant in real arithmetic and still changes quantized bytes.

INT8 attention v2 is specified by its emulator, `Int8AttentionV2Reference` (beside the host reference): the stats pass,
quantization and online recurrence its kernel rows reproduce, bit for bit on the Q/K/V bytes, scales, K means and INT32 QK
products when the emulator is fed the kernel's own scales, and up to summation order and the hardware exp2 on the output.
It applies the corrections above: a softcap gets each row's offset back before the tanh, sinks and the LSE take it, and
each key segment carries its own.

## 6. Planning and fallback

A backend declares one capability row per kernel. The planner picks, among the rows whose declared support and
granted permissions cover the call, the one with the lowest cost, counting the **lowering steps** a row needs (head
widening, staging copies for unsupported strides, mask expansion, casts, per-batch iteration) in both memory and time.
The chain of candidates ends at a memory-bounded GPU row or at `AttentionUnsupportedException` naming the unmet
permission or budget. It never ends at the CPU for a call whose operands live on the GPU: crossing devices is the
caller's explicit choice.

Failures are classed:

| Class | Examples | Effect |
|---|---|---|
| Unsupported | no row covers the call | raised at planning, before any work |
| Recoverable | a plan build refused, an engine configuration rejected, a workspace too small | the next row runs; no state changed |
| Sticky | an illegal address, a launch failure, anything raised after output writes began | the context is poisoned and nothing retries |

An admitted row must not raise `AttentionUnsupportedException`: it declines by returning false. If every admitted
row declines, the planner raises `AttentionRecoverableException`. Any exception a row raises is sticky and propagates
unchanged rather than being converted into a decline. A GPU backend runs the host reference only through a declared
row: one named in the plan and the execution evidence, with `HostReference` lowering, that warns once per gap. Vulkan declares one for `FlashAttention` (C.1);
CUDA declares none. Otherwise host execution is the caller's explicit choice through `AttentionHostReference.Run`, and
a backend without rows refuses `IBackend.Attention`.

A failure while a stream is capturing is never recorded as a permanent property of a head dim or a kernel.

## 7. Ownership, capture and caching

- The runtime knobs that steer attention are read once per step into a policy snapshot. A captured graph records the
  snapshot's version; a knob change invalidates the graph, so replay never runs a kernel the current policy forbids.
- Nothing on a capture call allocates, synchronizes, reads free memory or builds a plan. Budgets are fixed per step
  signature; workspaces come from stream-ordered allocation or scratch grown in the warm-up calls; plans are built in
  warm-up.
- Captured graphs and in-flight executions hold leases on the plans, workspaces and cached K/V they reference. Eviction
  waits for the last lease. Concurrent executions use distinct scratch or serialize explicitly.
- A cache key carries a content version the owner bumps on any in-place mutation, the model and LoRA identity,
  positions, dtype and quantization configuration. A key of tensor reference and shape alone is stale the moment the
  tensor is rewritten.
- Appending to a quantized cached prefix keeps per-segment statistics with §5's corrections, or requantizes the
  segment; a scale computed before the append is never silently reused.

## 8. Conformance

The oracle computes §1–§5 in double precision from the exact values the kernel saw (inputs rounded to their storage
format first). A row passes when it stayed inside its granted permissions and agrees within the tolerance of the
precision it actually used, each relative bound paired with an absolute floor of `tol · max|reference|`:

| Precision used | relL2 |
|---|---|
| F32 | 2e-5 |
| TF32 | 2e-3 |
| F16 storage, F32 accumulation | 3e-3 |
| BF16 storage | 8e-3 |
| BF16 compute | 1e-2 |
| INT8 Q/K | 1e-2, mean abs ≤ 2e-3 |

Negative controls keep the oracle honest: a per-head mask must change the output, a saturating cast must clamp, keys
beyond `Sq` must matter, a fully excluded row must give zero and `−∞`, a smoothed call with a softcap must match the
unsmoothed one.

## 9. Where kernels differ today

The planner reproduces these until the row that owns each is rewritten.

- CPU dense SIMD softmax still produces NaN for a fully excluded row. PR 02's executed CUDA materialized and
  cuDNN 9.21 controls return zero; that cohort-specific evidence does not verify every cuDNN release.
- A sliding window without `Causal` keeps every later key on CUDA and excludes them on Vulkan.
- CUDA's token-major permute fallback refuses key-only masks, masks with `B ≠ 1` and `B ≠ 1` operands; its cuDNN rows and its
  head-major entry accept them.
- CUDA's cuDNN rows read Q, K, V and the output as strided views (head dim contiguous, offset and strides multiples of 8
  elements, F16 or BF16; the output must cover its whole tensor) and are the only rows that take a strided or batched
  token-major call; no row stages a copy, so a view none of them takes is refused at planning. F32 operands are cast whole
  and so must be whole tensors. A layer takes the token-major entry for a batch of per-head operands only where its config
  sets `TokenMajorBatches` and the backend plans a row for the call; Vulkan's token-major rows read one sequence and plan none.
- CUDA's equal-head token-major entry has no SageAttention row: a call its head-major entry runs through Sage (F16, head
  dim 64 or 128, unmasked, both sequences at `numerics.sageF16MinSkv`) runs on cuDNN unless `numerics.ltx2SageTokenmajor`
  permutes it into the head-major entry. A layer whose config sets `HeadMajorForSage` keeps such calls head-major.
- CUDA's dense cuDNN rows read grouped K/V natively, and the generic layer preserves grouped heads through its head-major
  transform. Two rows repeat K/V and re-enter dense planning with the original mask, scale and numerical permissions:
  `widen.preferred`, ahead of cuDNN, for a grouped call whose equal-head form INT8 v2, Sage or the F16 flash kernel takes
  (the routes those calls had when the caller widened), and `widen.dense`, after cuDNN, when no native row completes. The
  grouped token-major row yields the same calls to its permute row, which uses that dense planner. Execution evidence
  reports `WidenKv` and the repeats only when a widening row runs. Whether widening for Sage beats native grouped cuDNN
  is unmeasured; calibrated routing decides it.
- CUDA's INT8 attention v2 rows (S.17 and M.4; with `numerics.int8AttnV2F16Pv`, S.18 and M.5) run only with
  `numerics.int8AttnV2`: unmasked, equal heads, head dim 128, F16 or F32 views with 16-byte aligned rows and an output
  covering its tensor, both sequences at `numerics.int8AttnV2MinTokens`. They take F32 whatever `allowF16` says, the
  grouped token-major row yields to them through `widen.preferred`, and a layer keeps a call token-major for
  M.4 only when the backend says M.4 takes it now (a capture its workspace does not fit goes to Sage). K is quantized per key: a scale
  shared by several keys lets one outlier key move the winner of a near-argmax row. u8 P rounds each key against its
  128-key tile's maximum, so a key below 1/510 of it is dropped, and a tile whose whole mass sits below that step loses
  it; the F16 PV keeps it. V scales default to one per channel per tile (`numerics.int8AttnV2VTile`), at or below the
  per-sequence error on every synthetic and captured call measured.
- CUDA runs F32 attention GEMMs in TF32 unless `numerics.sdpaNoTf32` is set (read when the backend is constructed; the
  `reference` profile sets it; `numerics.noTf32` does not reach them), and `FlashAttention`'s F32 prefill narrows to F16
  without a permission.
- CUDA's materialized F32→F16 cast does not saturate; cuDNN's does.
- CUDA chooses between the materialized and tiled paths, the tile height, and cuDNN or FA2 prefill from free VRAM. The tile holds about a
  quarter of free VRAM in F32 scores and never fewer than 128 MB (unless `numerics.sdpaTile` or the query length asks for less, or the 128 MB
  does not fit, in which case the tile is halved until it does and later calls start from that height, doubling it back after a run of calls that fit at it), because `cuMemGetInfo` reads far too low on a card a process has filled past about
  8 GiB (Windows), and a one-row tile is minutes per layer at 32 thousand tokens.
- CUDA's declared materialized and per-query-tile budget remains at most 2^32 score elements (B·H·Sq·Skv). The
  attention-specific softmax now uses wide row offsets; other preparation/GEMM/index limits have not been relaxed.
  Past that an unmasked F16 or BF16 call runs the query-tiled core on exact F32 widenings, an
  unmasked or key-masked F32 call the tiled F32 rows, and any other call is refused at planning, naming the budget.
- Vulkan's scalar/tiled mask shaders carry batch/head/query bias strides; the naive path uses the same layout to
  offset each block and expands key-only masks. Source indexing is not a complete executed mask-conformance campaign.
- Vulkan `FlashAttention` runs its declared host-reference row (C.1) when neither its scalar admission nor its
  restricted wide-sink row admits the call: softcap, causal sinks, ALiBi, or a head dim past the flash tile. Gemma-2
  (decoder and text encoder), GPT-OSS and ALiBi models take it. The wide row covers unmasked F32 sinks with optional
  softcap, no ALiBi, up to D1024. CUDA refuses Flash above D1024.
- **General Sol-Attn** is not an `AttentionCall`. It is a backend primitive, `IBackend.SolAttention`, defined by
  `SolAttentionReference`, and `IBackend.HasSolAttention` says whether a backend runs it. The CPU backend runs the
  reference; CUDA and Vulkan each run it as five kernels, seven with the coarse gate, in F32, with every option the
  reference has. Which key blocks a query block reads is the same as the reference's on CUDA, and on Vulkan wherever
  the driver keeps subnormals; the output is compared by tolerance. The Vulkan routing stages take their divides and
  their square root through `divRn` and `sqrtRn` (`Shaders/div_rn.glsl`, `sqrt_rn.glsl`), because a device's own
  need not be correctly rounded and on an RTX 3060 are not: about a third of sampled quotients and a quarter of sampled
  roots differ from the rounded result. An operand that is itself subnormal is read as zero by that device, helpers
  included, and no routing test covers a subnormal sum or spread, so the match with the reference stays conditional
  on that.
  The Kitchen's `token_aug` option is an open gap: its CUDA kernel implements it and its eager definition ignores it,
  so the only thing a port could be checked against is that kernel's output. Neither the reference nor a backend here
  has it. No model calls the primitive yet, and it has no timing.
