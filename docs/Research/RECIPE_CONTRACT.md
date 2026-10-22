# Recipe contract

How a model's generation defaults are owned, resolved and exposed. Engine-wide rules live in
[engine patterns](../Agents/ENGINE_PATTERNS.md#recipe-contract); this file keeps the reasoning, the ownership split
and the per-family defaults ledger with its sources.

## Why

Model defaults used to live in nine places that disagreed: `VideoDefaults.Standard` / `ImageDefaults.Standard`
(a silent generic fallback any recipe inherited by omission), `IRecipePipeline.VariantDefaults`, recipe statics read
directly by pipelines, per-pipeline `request.X ?? literal` fallbacks, Diffusion's `GenerationDefaults` table, a
Z-Image variant table in Diffusion, `ModelIdentityCatalog.StandardResolution`, `MusicRequest`'s baked-in values, and
hosts that re-declared all of it. Two consequences shipped: Wan VACE and Wan S2V ran 25 frames because their recipes
declared no frame count and the generic fallback filled it, while their pipelines meant 81; and hosts forced 25
frames / 24-25 fps on every family because nothing told them what the model wanted.

## Concepts

| Concept | What it is | Mutable |
|---|---|---|
| Recipe definition | A family's recommended values (`ImageDefaults`, `VideoDefaults`), what it can generate (`VideoConstraints`), its features and variants | No |
| Overrides | The values a user or host explicitly set on the request; `null` means "not set" | — |
| Resolved plan | Recipe + variant + overrides, validated and normalized: what the pipeline runs (`VideoPlan.EffectiveSettings` today) | No |

A recipe never changes because of a request; a request never changes the recipe. A user who asks Wan for 49 frames
gets 49 in the plan while the recipe still says its default.

```
ModelSpec ─► variant resolution ─► recipe definition ─► planner(definition, variant, overrides) ─► resolved plan ─► pipeline
```

## Ownership

| Concern | Owner |
|---|---|
| Recommended steps, guidance, size, frames, frame rate | Recipe `Defaults` / `DefaultsFor(variant)` |
| Limits a request must satisfy (frame grid, caps) | Recipe `Constraints` |
| Architecture constants (VAE compression, latent scale, sampling constants) | Model config, not the recipe |
| Explicit user values | Host request |
| Effective values | Resolved plan |
| Output metadata (frames, generation fps) | Generation result |
| UI display and toggles | Host |
| Installed / downloaded state | Host asset discovery, never the recipe |

A constraint is not a fallback: refusing a 700-pixel width or capping Lance at 121 frames is the model's capability.
The rule that replaces "delete every default": **no pipeline and no host may decide an omitted user-configurable
generation parameter.**

## Rules

- **Complete by type.** `Defaults` and (video) `Constraints` are abstract on the recipe interfaces, and every
  member a pipeline needs is `required` and non-nullable, so omitting one is a compile error. Optional members
  (`Sampler`, `Scheduler`, `SigmaShift`, `FlowShift`, `AudioFlowShift`) are null only where the family owns that
  choice internally.
- **Valid by registration.** `RecipeRegistryValidator` runs when a registry loads and on `Register`: every
  family-level and variant defaults set has positive steps, finite positive guidance, sizes that are positive
  multiples of 8, a positive frame rate and a frame count on the family's grid; every recipe resolves by its own name
  and no earlier recipe shadows it. A broken built-in registry throws at load instead of serving invented values.
- **No family, no defaults.** An unregistered family raises `RecipeNotRegisteredException`; nothing plans or
  estimates memory against a stand-in.
- **Frame grids.** A video family declares `FrameGrid(step, offset, normalization, max)`. Normalization is part of
  the model's contract (LTX and H3 snap, as SwarmUI documents); a family that declares `Reject` refuses an off-grid
  user value instead of silently changing it.
- **Three frame rates.** Generation fps (plan), container fps (mux) and preview fps are different values. Hosts mux
  at the result's fps, never at a literal.
- **Deviations are commented.** Defaults follow SwarmUI's documented per-model defaults (`docs/Video Model Support.md`,
  `docs/Audio Model Support.md`, `docs/Model Support.md` in SwarmUI) and core's ComfyUI fallbacks
  (`WorkflowGeneratorModelSupport.cs`, `WorkflowGenerator.cs`). Any value that differs, or that comes from upstream
  because SwarmUI has none, carries a one-line comment saying why.

## Defaults ledger

Values the recipes declare today against the reference they should match. A row is frozen only after its source has
been re-read; until then it is an audit finding, not a specification.

| Family | Today (steps / cfg / size / frames @ fps) | SwarmUI reference | Status |
|---|---|---|---|
| wan, wan-21-1_3b, wan-21-14b | 50 / 5.0 / 832×480 / 33 @ 25 | 81 frames, 24 fps default (2.1 base trained at 16), 832×480, cfg 6 | unverified |
| wan-22-5b | as wan | 24 fps native, cfg 5 | unverified |
| wan-vace | 50 / 5.0 / 704×480 / 25 @ 25 | as Wan; pipeline intends 81 | unverified |
| wan-s2v | 50 / 5.0 / 704×480 / 25 @ 25 | none in SwarmUI; pipeline intends 81 | unverified |
| wan-animate | 20 / 1.0 / 704×480 / 77 @ 30 | none in SwarmUI; upstream `wan_animate_14B.py` | unverified |
| wan-animate-2 | 40 / 3.0 / 704×480 / 81 @ 24 (distilled 10 / 1.0) | none in SwarmUI | unverified |
| ltx-video | 50 / 3.0 / 704×480 / 97 @ 25 | 97 frames, 24 fps, 768×512, cfg 3, 8n+1 rounded | unverified |
| ltx-video-2, ltx-2.5-distilled | 20 / 4.0 / 1280×736 / 121 @ 24 (distilled 8 / 1) | core fallback 97 frames, 24 fps; euler / normal | unverified |
| hunyuan-video | 20 / 6.0 (embedded guidance) / 512×320 / 25 @ 24 | 73 frames, 24 fps fixed, 960×544 | unverified |
| kandinsky5-video | 50 / 5.0 / 512×512 / 25 @ 24 | none in SwarmUI; pipeline fallback was 121 | unverified |
| lance-video | 30 / 4.0 / 512×512 / 25 @ 25 | none in SwarmUI; pipeline fallback was 81, cap 121 | unverified |
| minimax-h3 | 30 / 1.0 / 1344×768 / 124 @ 24 | 124 frames (17k+5, rounded up), 24 fps, cfg 1, `res_multistep` / `simple` | unverified |
| ace-step 1.5 (audio) | loader table | 120 s (max 600), turbo cfg 1 / 8 steps / shift 3 | unverified |
| minimax-music-3 (audio) | loader table | 120 s max, cfg 1.7, 30 steps | unverified |
| yue2 (audio) | loader table | 300 s max, cfg 1, 20 steps | unverified |

Rows marked "today" with 25 fps or 704×480 that SwarmUI does not list are the old generic fallback, kept so the
change that made defaults total moved no output.

## Open

- Planner with per-parameter provenance (recipe default, user override, normalization, lock) on every plan and
  result; image and audio plans of the same shape.
- Variant resolution refuses an unresolved variant instead of taking the catalog default, and filename guesses for
  families whose variants differ in defaults; Flux.1, Flux.2, Z-Image and LTX variant defaults move out of pipelines.
- Pipelines take the resolved plan only; pipeline `??` fallbacks, `GenerationDefaults` and `VariantDefaults` go.
- Audio recipes (music, speech, transcription identity and defaults).
- A side-effect-free descriptor (`IInferenceEngine.Recipes`) and HTTP endpoints for hosts.
