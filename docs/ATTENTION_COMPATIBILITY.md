# Attention compatibility ledger

This ledger owns the observable compatibility surface that still translates legacy attention configuration into the
call contract. The [attention contract](ATTENTION_CONTRACT.md) owns semantics and numerical permissions; the
[support contract](ATTENTION_SUPPORT.md) owns the common/optional backend domain. A listed adapter is compatibility
debt, not proof that its Boolean is a geometry-derived numerical grant.

## Legacy numerical adapters

| Adapter | Current behavior | Compatibility promise | Planned disposition |
|---|---|---|---|
| `IBackend.ScaledDotProductAttention(..., bool allowF16)` | Builds a dense `AttentionCall` through `AttentionNumerics.FromLegacy`. | Public consumers may retain the overload. `true` preserves the existing route-pinned F16 preference. | Keep as a thin adapter through the package audit; new model code supplies an explicit policy. |
| `IBackend.ScaledDotProductAttentionTokenMajor(..., bool allowF16)` (equal heads) | Builds a token-major call through the same adapter. | Public signature and existing token-major routing remain stable. | Keep as a thin adapter through the package audit. |
| `IBackend.ScaledDotProductAttentionTokenMajor(..., bool allowF16)` (GQA) | Builds a grouped token-major call through the same adapter. | Public signature and grouped-head behavior remain stable. | Keep as a thin adapter through the package audit. |
| `AttentionCall.FromScaledDotProductAttention` and `FromTokenMajor` | Preserve the legacy Boolean in `LegacyAssertedBounded`. | Existing backend entry-point wrappers continue to construct the same call shape and route. | Retain only as the implementation of the public overloads above. |
| `AttentionLayerConfig.AllowF16` | Preserves each package's current route-pinned assertion. | Existing serialized/model construction configuration remains valid. | Replace per package with bounds-backed numerical policy during the wiring audits; do not remove before its consumer is migrated. |
| `JointAttentionConfig.AllowF16` and `AllowF16Batched` | Select the existing single/batched `AttentionLayer` policy. | Batched compatibility choice remains independent from the single-item choice. | Replace with explicit policies together with the owning package's joint-attention audit. |
| CUDA/Vulkan Boolean overloads | Forward the public Boolean through the shared legacy adapter. | Direct backend consumers retain current API behavior. | Keep until the public compatibility review explicitly deprecates an overload. |

`AttentionNumerics.LegacyAssertedBounded` deliberately records provenance separately from `QueryKey`, `ProbValue`,
their bounds and saturation. It must not be used by a new `Auto` route as evidence that narrowing is safe. Every
remaining use is visible in source through `AttentionNumerics.FromLegacy` or the two configuration properties above.

## Behavioral and configuration rules

- The shared `AttentionCallValidation` boundary runs before output writes on every public attention entry: CPU
  `Attention`, `PlanAttention` and `ScaledDotProductAttention` (the CPU `FlashAttention`, `FlashAttentionDev` and
  token-major entries are interface defaults that build a call and enter `Attention`), and every CUDA and Vulkan
  legacy entry through its row runner. It checks finite scale and softcap, non-negative windows, block size and host
  query offset, operand geometry, storage bounds and strides, output aliasing, bias broadcast, `[Hq]` F32 sinks and
  ALiBi slopes, bias dtype and explicitly refused formats.
- Exception classes: every malformed-call check runs first and throws `ArgumentException` (or a subclass). Only a
  well-formed call can reach `AttentionUnsupportedException`, for a non-F32 bias or a refused stored format. A legacy
  entry's own contract check runs before the shared boundary and keeps its historical classes: an operand dtype the
  kernel cannot read is `NotSupportedException`, which `AttentionUnsupportedException` derives from, so callers that
  catch `NotSupportedException` see one family on every backend.
- "Structurally valid" means the host reference's broadcast rule: a bias may broadcast its key axis. The legacy CUDA
  and Vulkan kernels require the key axis to be `Skv`; their rows decline the rest after this boundary, so planning
  and execution still agree.
- Route admission constraints stay in the rows and legacy contracts, not this boundary: the CUDA flash rule that a
  causal query lies inside the valid prefix (`qOffset + Sq <= kvLen`, `kvLen >= 1`), F16 K/V head-dim limits and
  32-bit launch bounds.
- Validation is intentionally independent of optional capability admission. A structurally valid optional call may
  still produce `AttentionUnsupportedException` before output writes.
- Planning uses the same preflight as execution, so a plan cannot describe a malformed call that execution would
  reject. Backend runtime predicates may only narrow a capability after that boundary.
- Knobs can disable a permitted route, but cannot make a refused format admissible. A forced-format knob is evidence
  for a diagnostic profile, never a substitute for policy and bounds.

## Intentional tightening

Moving validation to one boundary rejects some calls that previously ran. Each is either unsafe or already rejected
by another backend:

| Check | Previously | Now | Evidence that no caller depends on it |
|---|---|---|---|
| Output aliases Q, K or V | Rejected only on CUDA view calls; CPU, Vulkan and the CUDA legacy dense, token-major and flash entries ran it. | Rejected on every backend. | A scan of all 45 four-argument legacy attention call sites under `src/` found no output argument textually equal to Q, K or V; the Core, LLM, Diffusion, Vision and World CPU lanes pass. Aliasing through an intermediate variable is not covered by the scan; a caller that did it would now fail loudly with `ArgumentException` rather than run. |
| Negative operand strides | Accepted by the host reference, whose extent check assumed positive strides and could read outside the tensor. | Rejected. | Legacy entries build contiguous operands, so only direct `AttentionCall` views can carry one. |
| Sinks or ALiBi slopes not exactly `[Hq]` F32 | CUDA rejected (`ArgumentException`); the host reference accepted extra values and treated a non-F32 tensor as unsupported. | `ArgumentException` everywhere. | The in-tree producers store exactly `Hq` F32 values: the GPT-OSS and Flint per-layer `self_attn.sinks` (converted to F32 on load, never sharded because tensor parallelism refuses sinks), the HCA and DSA sinks (checked to `h` on load), and the GPT-style and BERT ALiBi slopes, built with exactly `NumHeads` values (tensor parallelism refuses ALiBi too). |
| Negative softcap, window, block size or host query offset | Rejected by the CUDA flash and CPU flash contracts only. | Rejected everywhere. | Values are invalid by the [attention contract](ATTENTION_CONTRACT.md). |

## Behavior changes outside validation

These calls still validate; what happens after validation changed.

| Change | Previously | Now | Who can observe it |
|---|---|---|---|
| CUDA F16 flash kernel (`numerics.flashF16`, head dim 256) launch failure | Logged once, disabled the kernel for the session and ran the call on cuDNN. | The exception propagates and the request fails. `false` from the row means only a pre-launch shape or alignment decline. | A session whose `flash_attn_f16` launch fails. A failed launch leaves the CUDA context in an unknown state, so the retry was not reliable; `numerics.flashF16=0` selects cuDNN. |
| Every admitted row declined | `AttentionUnsupportedException` (a `NotSupportedException`). | `AttentionRecoverableException` (an `InvalidOperationException`). No admitted row still raises `AttentionUnsupportedException`. | Code that catches `NotSupportedException` to fall back. No handler in `src/` sits on an attention path. |
| `IBackend.Attention`, `PlanAttention` and the `FlashAttention` default on a backend without rows | Ran the host reference; planning returned a host-reference plan. | Validate, then refuse with `AttentionUnsupportedException`; planning validates and returns a plan with no row. `AttentionHostReference.Run` is the explicit host path. | An out-of-tree `IBackend` that relied on the defaults. CPU, CUDA and Vulkan override `Attention` and `PlanAttention`; the CPU serves `FlashAttention` through its `Attention` override. |
| CUDA Flash above head dim 1024 | Ran the host reference (row F.1). | Refused. | No in-tree model has a head dim that wide. |
| Additive bias on CUDA's F16 materialized rows | The F32 bias was clamped to the F16 range, cast to F16 and added in F16, so a row masked on every key with a finite sentinel such as `-1e30` averaged its highest-scoring keys. | The F32 bias is added unclamped in F32 (`attention_softmax_f16`), as the host reference does: that row averages every key uniformly. Every other biased row now receives the bias at F32 precision instead of F16. | A caller that masks a whole row with a finite sentinel and reads that row. |

## Removal evidence

Before an adapter may be removed or changed, the owning package audit records the caller, model/configuration path,
explicit replacement policy, affected route evidence and real-generation correctness result. Public adapters are
preserved until a documented compatibility decision; removing an internal use does not authorize a public break.
