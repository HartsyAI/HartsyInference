# Add a model

For a newly released model, run the end-to-end [add-model skill](skills/add-model/SKILL.md): research, reuse map, official-Python and ComfyUI baselines, full feature support, real-weight verification through SwarmUI, and the engine and extension PRs. Install it for Claude with `mkdir -p ~/.claude/skills/add-model && cp docs/Agents/skills/add-model/SKILL.md ~/.claude/skills/add-model/`. The steps below are the engine rules it relies on.

1. Read the relevant architecture reference and modality status; search TROUBLESHOOTING for the family/operation.
2. Map shapes, dtypes, masks, prediction type and ownership at every boundary. Inspect checkpoint keys/metadata before designing conversion; repack once at load time.
3. Use the sibling package and IBackend/shared operations. Engine owns registration, catalog, loading and consumer dispatch; video must use the planning contract in [engine patterns](ENGINE_PATTERNS.md#video-planning-contract).
4. Declare the recipe's complete `Defaults` (and a video family's `Constraints` frame grid) from SwarmUI's documented defaults, commenting any deviation; the registry validator and `RecipeLedgerTests` pick the recipe up automatically. See [recipe contract](ENGINE_PATTERNS.md#recipe-contract).
5. If builds share the architecture but not the contract (base/edit, turbo/distilled), declare a variant catalog; see [model variants](ENGINE_PATTERNS.md#model-variants). Map each SwarmUI class id and `modelspec.architecture` value before reaching for filename tokens.
6. Validate components with saved reference noise/embeddings, then real weights, generated output and the intended consumer. Matching seeds does not match RNG implementations.
7. Record exact checkpoint, reference version, inputs, dtype/backend, tolerances and evidence in the canonical status/parity docs.

Use the safe-subset pickle parser; never execute checkpoint pickle. Preserve configuration/tokenizer metadata and report quantization loss. Choose precision per family from validated behavior, not a universal Q8/F16 prescription.

Keep synthetic tests labeled SyntheticSmoke. Real-weight/GPU/network tests keep their resource categories and gates even after verification; a clean skip is not a pass. Follow [test filtering](../CODE_STYLE.md).

Bisect numerical differences before changing tolerances. Typical FP32 diagnostic scales are elementwise 1e-7, normalization 1e-6, GEMM 1e-5, attention 1e-4 and full denoiser 1e-3; these are starting points, not universal acceptance criteria. Finite tensors alone do not establish useful output.

Common traps: legacy diffusers attention_head_dim can mean head count; absent Shape dimensions return zero; gated activations split the last dimension; CLIP final normalization and timestep ordering matter; BF16/F16 storage must not be read as float pointers.
