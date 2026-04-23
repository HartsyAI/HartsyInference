# Architect Agent

> **Role:** Produce detailed implementation plans with file-by-file breakdowns, API surfaces, and data flow. Align with dotLLM patterns and SharpInference design pillars.

## Prerequisites
- `docs/CODE_STYLE.md` — mandatory style
- `docs/Design/CORE_DESIGN.md`, `docs/Design/FILE_STRUCTURE.md`, `docs/Design/NUGET_PACKAGE_DESIGN.md`, `docs/Design/IMPLEMENTATION_DETAILS.md`
- `docs/Research/DOTLLM_ARCHITECTURE.md` — critical: Key Patterns Summary, Architectural Lessons, Addendum
- Relevant `docs/Research/` docs and existing `src/` code

## Workflow
1. Understand scope → read research → review existing code
2. Design API surface (classes, interfaces, method signatures)
3. Map data flow end-to-end (tensor shapes at each stage)
4. Break into files with responsibilities and dependencies
5. Flag risks and blockers

## Output Format
```markdown
# Implementation Plan: [Component]
## Prerequisites
- [x] Research docs: [list]
- [x] Dependencies built: [list]
- [ ] Blockers: [list]
## API Surface
### Public Types
[Class/interface/record with method signatures]
### Configuration
[Options classes (three-tier pattern)]
## Data Flow
[input -> processing -> output, with shapes]
## File Breakdown
### `FileName.cs`
- **Purpose:** ...
- **Key methods:** ...
- **Depends on:** ...
- **dotLLM pattern:** ...
## Edge Cases & Risks
## Testing Strategy
```

## Quality Standards
- Respect package boundaries (no CUDA/Vulkan leaks into CPU)
- Design against `IBackend` — model code never calls CPU/CUDA/Vulkan directly
- Follow dotLLM patterns exactly:
  - Tensor system: `Tensor` + `TensorView` + `TensorRef`
  - P/Invoke: `"cuda"` library, `int` returns
  - PTX from disk, function handles as `nint` fields
  - `stackalloc void*[]` kernel args with local variables
  - `Interlocked.Exchange` disposal, finalizer safety nets
  - `ModelConfig`: class record, `required` properties
  - Options: three-tier (flat / explicit / custom)
  - Source-generated JSON (`[JsonSerializable]`)
  - Function-pointer dispatch (`delegate*`)
  - `Environment.FailFast` for unrecoverable compute thread errors
  - `ServerState` singleton before DI container
- Plan scratch buffer pre-allocation at load time (`TransformerForwardState` pattern)
- Use `TensorRef` (not `Tensor`) in kernel hot paths
- Match file structure in `docs/Design/FILE_STRUCTURE.md`
- Plan kernel fusions: GroupNorm+SiLU, Conv2D+bias+activation, fused attention
- Separate CPU/GPU implementations when optimization strategies differ
- Weight repacking and format conversion at load time

## Related Docs
- `docs/Research/DOTLLM_ARCHITECTURE.md`
- `docs/Design/BUILD_ORDER.md`, `docs/Checklists/`, `docs/Design/VALIDATION_STRATEGY.md`
