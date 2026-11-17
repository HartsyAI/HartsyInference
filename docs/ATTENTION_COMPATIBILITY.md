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

- The shared `AttentionCallValidation` boundary runs before CPU, CUDA or Vulkan planning/execution can write output.
  It checks structural geometry, storage bounds, output aliasing, finite scale, broadcast bias, per-head terms and
  explicit refused operand formats.
- Validation is intentionally independent of optional capability admission. A structurally valid optional call may
  still produce `AttentionUnsupportedException` before output writes; a malformed call is `ArgumentException`.
- Planning uses the same preflight as execution, so a plan cannot describe a malformed call that execution would
  reject. Backend runtime predicates may only narrow a capability after that boundary.
- Knobs can disable a permitted route, but cannot make a refused format admissible. A forced-format knob is evidence
  for a diagnostic profile, never a substitute for policy and bounds.

## Removal evidence

Before an adapter may be removed or changed, the owning package audit records the caller, model/configuration path,
explicit replacement policy, affected route evidence and real-generation correctness result. Public adapters are
preserved until a documented compatibility decision; removing an internal use does not authorize a public break.
