# Add a model

Use this file when a model is new to the engine. For a newly released model, run the end-to-end
[add-model skill](skills/add-model/SKILL.md). It covers research, the reuse map, official-Python and ComfyUI baselines,
full feature support, real-weight verification through SwarmUI, and the engine and extension PRs. To install it for
Claude, run `mkdir -p ~/.claude/skills/add-model && cp docs/Agents/skills/add-model/SKILL.md ~/.claude/skills/add-model/`.
The steps below are the engine rules that the skill depends on.

Work through these steps in order. Each one produces evidence the next one uses.

1. Read the architecture reference and modality status for the model. Search [troubleshooting](../Checklists/TROUBLESHOOTING.md)
   for its family and operation.
2. Map shapes, dtypes, masks, prediction type and ownership at every boundary. Inspect the checkpoint keys and metadata
   before you design the conversion, and repack once at load time.
3. Build on the sibling package and the shared `IBackend` operations. The engine owns registration, the catalog, loading
   and consumer dispatch. Video models must follow the planning contract in [engine patterns](ENGINE_PATTERNS.md#video-planning-contract).
4. Declare the recipe's complete `Defaults` (and, for a video family, its `Constraints` frame grid), taken from SwarmUI's
   documented defaults. Comment any deviation. The registry validator and `RecipeLedgerTests` pick the recipe up
   automatically. See the [recipe contract](ENGINE_PATTERNS.md#recipe-contract).
5. If builds share an architecture but not a contract (base and edit, or turbo and distilled), declare a variant catalog.
   See [model variants](ENGINE_PATTERNS.md#model-variants). Map each SwarmUI class id and `modelspec.architecture` value
   before you reach for filename tokens.
6. Validate components in this order: saved reference noise and embeddings, then real weights, then generated output,
   then the intended consumer. Matching seeds does not mean matching RNG implementations.
7. Record the exact checkpoint, reference version, inputs, dtype and backend, tolerances and evidence in the canonical
   status and parity documents.

Parse checkpoints with the safe-subset pickle parser. Never execute checkpoint pickle. Keep configuration and tokenizer
metadata, and report any quantization loss. Choose precision per family from validated behaviour, not from a fixed
Q8 or F16 default.

Label synthetic tests `SyntheticSmoke`. Keep the resource categories and gates on real-weight, GPU and network tests even
after they pass. A skipped test is not a passing test. Follow [test filtering](../CODE_STYLE.md#testing).

Bisect a numerical difference before you change a tolerance. Typical FP32 diagnostic scales are 1e-7 for elementwise
operations, 1e-6 for normalization, 1e-5 for GEMM, 1e-4 for attention and 1e-3 for the full denoiser. These are starting
points, not acceptance criteria. Finite tensors alone do not show that the output is useful.

Common traps:

- In legacy diffusers configs, `attention_head_dim` can mean a head count.
- An absent Shape dimension returns zero.
- Gated activations split the last dimension.
- CLIP final normalization and timestep ordering both affect output.
- BF16 or F16 storage must not be read as float pointers.
