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

## Planning and provenance

Video planning resolves every request against a `VideoRecipeDeclaration` (the family's defaults, features and
constraints for the resolved variant) and returns a `VideoPlan` whose `Recipe` names the family, defaults revision,
variant, the evidence that chose it, and any checkpoint profile. `Provenance` records each effective setting's value
and source:

| Source | Meaning |
|---|---|
| `RecipeDefault` / `ProfileDefault` | unset by the caller; the recipe, or an exact-checkpoint profile, supplied it |
| `UserOverride` | set by the caller and run as set |
| `Normalized` | set by the caller and changed by planning or by a pipeline that pins it (a frame rate taken from an input clip); the note names the requested value |
| `Derived` | computed: a random seed, or a frame rate the pipeline pinned from an input clip when the caller set none |

The plan's identity and provenance travel onto `VideoGenerationResult` and `ImageResult` and are logged once per
generation, so a wrong value in an output can be traced to the recipe, the host, the user or the pipeline.

Two limits, by design for now:
- **Image provenance compares the caller's request with the resolved request.** A size a pipeline later rounds
  (to its latent multiple) still reports the resolved value; the result's own `Width`/`Height` are what ran.
- **The streaming path (`GenerateFramesAsync`) yields frames only.** It logs `[VideoPlan]` but carries no result
  object, so a host that streams reads provenance from `IVideoPlanningService.PlanAsync`.

Frame counts are normalized in planning, not in pipelines. A count off a family's grid snaps as the grid declares;
**a snap that would pass the grid's cap lands on the cap**, which is on the grid and the closest count the family can
generate (Lance's former `Math.Min(121, …)`). A family that declares `Reject` refuses the plan naming the two nearest
valid counts.

**Revisions.** Each defaults record carries a `Revision`. A change to any value bumps it, so output generated under
the old and new values stays distinguishable in its metadata.

## Descriptors and the query surface

Hosts ask the engine what a model's recipe declares; they never keep their own tables. `IInferenceEngine.Recipes`
(`IRecipeCatalog`) answers without loading weights or constructing a pipeline:

| Query | Answer |
|---|---|
| `Describe(ModelSpec)` | The family and variant the spec resolves to, through the same routing a generation uses |
| `DescribeFamily(modality, familyId, variantId?)` | A family's catalog default variant, or the named variant; an unknown variant throws and names the valid ids |
| `All(modality?)` | One family-level descriptor per built-in recipe, in registry order |

Every descriptor carries `Recipe` (family, defaults revision, variant), `Variants` (the catalog's ids) and `ProfileBound`
(true for MiniMax-H3, whose exact-hash checkpoint profiles overlay the defaults at plan time; the plan and the descriptor
read the same check). An `ImageModelDescriptor` adds `Defaults`, `Features`, `InputLimits` and `PromptWeighting`; a
`VideoModelDescriptor` adds `Defaults`, `Constraints`, `Features` and `Sampling`. The descriptor is the wire type:
`[JsonPolymorphic]` with a `$type` of `image` or `video`.

Routes: `GET /v1/native/recipes?modality=`, `GET /v1/native/recipes/{modality}/{familyId}?variant=` and
`POST /v1/native/recipes/describe`. The last takes the same `model` and `modelPath` as the video plan route and returns
422 `recipe.family.unregistered` for a family with no recipe.

## Defaults ledger

Values the recipes declare today against the reference they should match. A row is frozen only after its source has
been re-read; until then it is an audit finding, not a specification.

| Family | Today (steps / cfg / size / frames @ fps) | SwarmUI reference | Status |
|---|---|---|---|
| wan | 50 / 5.0 / 832×480 / 81 @ 24 | core: 81 frames, 24 fps for every Wan; size kept native (core class is 960² or 640²) | verified 2026-10-22 |
| wan-21-1_3b, wan-21-14b | 50 / 6.0 / 832×480 / 81 @ 24 | core `wan-21-*` branch: cfg 6, 81 frames, 24 fps; docs: official cfg 6. | verified 2026-10-22 |
| wan-21-14b, variant `i2v-22-14b` | 50 / 3.5 / 832×480 / 81 @ 24 | core `wan-2_2-image2video-14b` branch: cfg 3.5, 24 fps (49 frames image-to-video); told apart by structure: hidden 5120, 36-channel patch embedding, no `k_img` / `img_emb` | verified 2026-10-22 |
| wan-22-5b, ti2v-5b | 50 / 5.0 / 832×480 / 81 @ 22 | core `wan-22-5b` branch: cfg 5, 22 fps; 81 frames text-to-video (49 image-to-video). Docs say 24 fps native | verified 2026-10-22 |
| wan-vace | 50 / 6.0 / 704×480 / 81 @ 24 | classed under `wan-21-*`, so core runs the Wan 2.1 branch; 704×480 is the old fallback (core class 960²) | verified 2026-10-22 |
| wan-s2v | 50 / 5.0 / 704×480 / 25 @ 25 | none in SwarmUI; pipeline intends 81 | unverified |
| wan-animate | 20 / 1.0 / 704×480 / 77 @ 30 | none in SwarmUI; upstream `wan_animate_14B.py` | unverified |
| wan-animate-2 | 40 / 3.0 / 704×480 / 81 @ 24 (distilled 10 / 1.0) | none in SwarmUI | unverified |
| ltx-video | 50 / 3.0 / 768×512 / 97 @ 24 | 97 frames, 24 fps, 768×512, cfg 3, 8n+1 rounded | verified 2026-10-22 |
| ltx-video-2, ltx-2.5-distilled | 20 / 3.0 / 1280×736 / 97 @ 24 (distilled 8 / 3.0) | core: cfg 3 and 97 frames for every LTX-2, distilled included (docs: ~4 dev, 1 distilled); size kept native (core class 960²) | verified 2026-10-22 |
| hunyuan-video | 20 / 6.0 (embedded guidance) / 720×720 / 73 @ 24 | core class 720×720, 73 frames, 24 fps | verified 2026-10-22 |
| kandinsky5-video | 50 / 1.0 / 640×640 / 73 @ 24 | core Kandinsky 5 branch: cfg 1 (docs: 5 regular, 1 distilled), class 640², 73 frames | verified 2026-10-22 |
| lance-video | 30 / 4.0 / 512×512 / 25 @ 25 | none in SwarmUI core; the bridge registers 832×480 as a host-declared size; pipeline cap 121 | unverified |
| minimax-h3 | 30 / 1.0 / 1344×768 / 124 @ 24 | 124 frames (17k+5, rounded up), 24 fps, cfg 1, `res_multistep` / `simple`; size kept native (core class 960²) | verified 2026-10-22 |
| ace-step 1.5 (audio) | 120 s; turbo 8 / cfg 1 / shift 3, sft 50 and base 32 at cfg 7 / shift 1 | 120 s (max 600), turbo cfg 1 / 8 steps / shift 3 | verified 2026-10-22 |
| minimax-music-3 (audio) | 120 s, 30 steps, flow cfg 1.7 | core: 120 s; docs: flow cfg 1.7 official, 30 steps. Core's `cfg_scale` 1.5 on `MiniMaxMusic3TextEncode` is the code LM's guidance, which `MiniMaxMusic3ArPipeline` already runs at 1.5; frame grid caps at 360 s | verified 2026-10-22 |
| yue2 (audio) | 300 s, 32 ODE steps; cfg owned by the planning mode | 300 s, cfg 1, steps "eg 20" (an example, so 32 stays); protocol caps at 900 s | verified 2026-10-22 |

Rows marked "today" with 25 fps or 704×480 that SwarmUI does not list are the old generic fallback, kept so the
change that made defaults total moved no output.

Image families whose defaults were moved to SwarmUI's documented values (`docs/Model Support.md` = MS,
`docs/Obscure Model Support.md` = OMS, read 2026-10-22). Image families not listed already matched, or SwarmUI gives
only an example value, so the upstream value stays.

| Family / variant | Was | Now | Source |
|---|---|---|---|
| chroma | 35 / 5.0 | 26 / 3.5 | MS Chroma: official 26 steps, CFG around 3.5 |
| flux1 schnell | 4 / 3.5 | 4 / 1.0 | MS Flux: CFG 1, Flux Guidance does not apply to Schnell |
| flux2 dev | 50 / 4.0 | 50 / 3.5 | MS Flux.2: Flux Guidance defaults to 3.5 |
| flux2 klein (4B, 9B) | 4 / 1.0 | 8 / 1.0 | MS Flux.2 Klein: distilled, 8 steps, CFG 1 |
| flux3 distilled | 15 / 1.0 | 16 / 1.0 | MS Flux 3 Image: 16 steps for 'sd' |
| hunyuan-image | 50 / 3.5 | 20 / 3.5 | MS HunyuanImage 2.1: steps around 20 |
| omnigen2 | 28 / 4.0 | 20 / 5.0 | OMS OmniGen2: steps ~20, CFG 5 to 7 |
| lens, lens turbo | 1024² | 1440² | OMS Lens: 1440 is the official default side |
| boogu base | 25 / 3.5 | 20 / 4.0 | MS Boogu: 20 steps, CFG around 4 |
| mage-flow base | 30 / 5.0 | 20 / 5.0 | OMS Mage Flow: 20-ish steps |
| z-image base | shift 6 | shift 3 | MS Z-Image: default sigma shift 3 |

Samplers, schedulers and shifts SwarmUI sets per model were checked the same way (2026-10-22). Two are taken: `er_sde`
for Anima Base and Flux 3 Image Base (core `WorkflowGenerator.cs`). The rest stay with the pipeline:

- **Refused names.** `flux2`, `ideogram4` and `AlignYourSteps` are not in `SigmaSchedule.Names`. The solver-owned
  families (Wan, LTX, LTX-2, MiniMax-H3, Ideogram 4, Lumina 2) refuse any sampler or scheduler. Declaring one of these
  would fail every unset request.
- **Schedules on image families.** Any declared schedule is refused with img2img or inpaint (`FlowMatchSampling.Resolve`),
  and `simple` over a family's own grid returns the same sigmas. So Flux.1, Flux 3, Anima and Wan keep `null`, and so
  do Chroma's and Radiance's `beta`.
- **Shifts.** Where the engine computes a resolution-dependent shift (Chroma, Qwen-Image, Flux.2, Krea 2, Boogu), a
  fixed SwarmUI shift would override it. SwarmUI core applies a fixed value only for Chroma (1) and Qwen-Image (3); the
  docs' Flux.2, Krea 2, Boogu and HunyuanImage values apply only when the user sets a shift. Where the value already
  equals the pipeline's (SD3 3, AuraFlow 1.73, HiDream 3, Lumina 2 6, MiniMax-H3 12), declaring it changes nothing.
- **Mode-owned values.** YuE2's CFG depends on its planning mode (1.01 with planning off, 1.0 otherwise), so its
  recipe leaves it to the pipeline. Boogu Turbo's own shifted linspace already equals `sgm_uniform`.

Lance image stays at 1024² (upstream's official size). The SwarmUI bridge registers its class at 768² and Lance
video at 832×480; those are host-declared sizes, which this contract replaces with `IInferenceEngine.Recipes`.

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

- ACE-Step steps, shift and guidance for a checkpoint placed on disk are still applied by the loader, which tells v1
  from 1.5 only after reading it. A structural ACE-Step variant catalog moves them into the recipe, as for Flux.
- Image sizes are still snapped inside the pipelines (multiples of 16, area clamps), after provenance is recorded;
  that moves into planning with the pipeline migration, so a snapped size reports as a normalization.
- Variant resolution refuses an unresolved variant instead of taking the catalog default, and filename guesses for
  families whose variants differ in defaults. Variant defaults now come from the catalog alone, and `VariantDefaults`
  is gone. Flux.2's filename guess remains the last resort for prefixed checkpoints, whose names its structural rule
  does not read; LTX-Video's filename clauses in `SupportsFor` remain.
- Pipelines take the resolved plan only; the remaining pipeline `??` fallbacks go (`GenerationDefaults` is gone).
- Audio speech and transcription recipes (music is typed; see Audio above).
