# Research Agent

> Deep-dive a topic and produce a research doc that an implementer can build from without re-deriving it.
> Assumes you've read `AGENTS.md` + `docs/CODE_STYLE.md`. For broad multi-source web research, use the
> `deep-research` skill; for a model architecture, ground everything in the reference implementation's code.

## What "good research" means here

The deliverable is **exact numbers + a validation plan**, not prose. Every architecture claim needs a
concrete source (repo file/line), and every risky component needs a way to check the C# port against a
reference within a tolerance.

```text
✅ "SD1.5 timestep embed: flip_sin_to_cos=True → [cos,sin], divisor (half_dim-1); ref
    diffusers/models/embeddings.py get_timestep_embedding()."
❌ "SD1.5 uses a standard sinusoidal timestep embedding."   (an implementer still has to go read the code)
```

```text
✅ validation plan: dump the reference's per-layer mean/std/min/max + a saved noise tensor; the C# port
   feeds the SAME noise and matches layer-by-layer within the FP32 tolerance ladder (ADD_MODEL.md).
❌ "verify it matches by running both with seed 42."   (C# Box-Muller ≠ PyTorch RNG — share noise, not seeds)
```

## Where to look

- Live corpus: `docs/Research/{SIMD_INTRINSICS_DOTNET,CUDA_AND_PTX,CONV2D_CUDA,CUDA_PERFORMANCE,
  VULKAN_COMPUTE_API,SPIRV_COMPUTE_SHADERS,VULKAN_MEMORY_MANAGEMENT,SAFETENSORS_FORMAT,GGUF_FORMAT,
  QUANTIZATION_DIFFUSION}.md`.
- Authorities to reconcile against: `docs/Checklists/PARITY_VERIFICATION.md` (what's already proven correct),
  `MODEL_STATUS.md` (status index), `TROUBLESHOOTING.md` (bugs a prior port already hit — read before
  proposing a plan, so you don't re-discover a solved trap).

## Output shape

```markdown
# [Topic] — Research Notes
> Status: Complete | Last Updated: YYYY-MM-DD | Needed before: [component]

## Summary            — 1-2 paragraphs
## Key numbers        — exact channels/layers/shapes/scale factors/eps values code needs
## Data layouts       — byte layouts, tensor shapes, weight-key names, memory formats
## Algorithm          — pseudocode where it isn't obvious
## References         — repo file/line for each claim; where implementations DISAGREE, say which and why
## Validation plan    — reference to dump, tensors to compare, tolerance per stage
## Open questions     — anything unresolved, clearly marked (never guess into the body)
```

Mark a doc `Status: Complete` **only** when every section is filled and the open questions are resolved or
explicitly deferred. Precise beats vague: "48 blocks, hidden 3072, RoPE θ=150000, factor 32" — not "a large
transformer with rotary embeddings."
