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
  multiples of 8, a positive frame rate and a frame count on the family's grid, and every recipe resolves by its own
  name. A broken built-in registry throws at load (a `TypeInitializationException` whose inner exception lists every
  problem) instead of serving invented values. The shadowing check covers the built-in list only: `Register` puts a
  recipe in front on purpose, to override one. The multiple of 8 is a floor every VAE shares, not a family's real
  alignment; that belongs in its constraints.
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
| minimax-music-3 (audio) | `MiniMaxMusic3MusicRecipe`: 10 s, 30 steps, cfg 1.7 | 120 s max, cfg 1.7, 30 steps; frame grid caps at 360 s | unverified |
| yue2 (audio) | `Yue2MusicRecipe`: 10 s, 32 ODE steps; cfg owned by the planning mode | 300 s max, cfg 1, 20 steps; protocol caps at 900 s | unverified |

Rows marked "today" with 25 fps or 704×480 that SwarmUI does not list are the old generic fallback, kept so the
change that made defaults total moved no output.

## Audio

Music families follow the same contract through `IMusicRecipe` and `MusicRecipeRegistry`, which mirror the video
types. `MusicDefaults` holds the values a request's unset fields take; `MusicConstraints` holds the longest clip a
family renders. `MusicService` applies the recipe before loading or generation, so `MusicRequest` carries only
values a caller set. Every field is nullable, and a `null` in a recipe means the loader or pipeline owns that choice.

Two cases deviate from "the recipe owns it":

- ACE-Step's 1.5 and v1 pipelines take different values for steps and guidance, and a placed checkpoint only shows
  which one it is after its header is read. The service therefore applies only the family-level values to a placed
  checkpoint, and the 1.5 loader applies the variant's steps, shift and guidance once it knows the checkpoint is 1.5.
- The output sample rate is not a recipe value: the loaded model reports it (48 kHz on ACE-Step 1.5, 44.1 kHz on v1),
  and the runner is the authority.

Audio speech and transcription recipes are not yet typed; their defaults stay in their loaders.

## Open

- What `SnapNearest` and `SnapUp` do when the snapped count would pass a grid's `Max` (refuse, or clamp down onto
  the grid), decided where the planner first applies the grid.
- Planner with per-parameter provenance (recipe default, user override, normalization, lock) on every plan and
  result; image and audio plans of the same shape.
- Variant resolution refuses an unresolved variant instead of taking the catalog default, and filename guesses for
  families whose variants differ in defaults; Flux.1, Flux.2, Z-Image and LTX variant defaults move out of pipelines.
- Pipelines take the resolved plan only; pipeline `??` fallbacks, `GenerationDefaults` and `VariantDefaults` go.
- Audio speech and transcription recipes (music is typed; see Audio above).
- A side-effect-free descriptor (`IInferenceEngine.Recipes`) and HTTP endpoints for hosts.
