# Adding a Model

> Back to [Core Design](CORE_DESIGN.md) · See also [Engine Refactor Plan](ENGINE_REFACTOR_PLAN.md), [File Structure](FILE_STRUCTURE.md)

This is the step-by-step for wiring a new model into HartsyInference. It is written from the code as it
actually is — every path, type, and snippet below is real. Pick the path that matches what you are adding:

| You are adding | Go to |
|---|---|
| A new image or video **architecture family** | [A. Image / video family](#a-a-new-image--video-architecture-family) |
| A new **audio** model (TTS / STT / music / voice-conversion / FX) | [B. Audio model](#b-a-new-audio-model) |
| A new **LLM** | [C. LLM](#c-a-new-llm) |
| Anything — how to prove it works | [D. Testing it](#d-testing-it) |

Read [Gotchas](#gotchas) before you start. Most of the time lost on these lifts was lost there.

---

## A. A new image / video architecture family

### A0. First: is this an Engine task at all?

**The math is not in the Engine.** Pipelines, denoisers, text encoders, VAEs, schedulers, and checkpoint
converters live in `HartsyInference.Diffusion` (and `HartsyInference.Video` for video backbones):

| Concern | Package | Example |
|---|---|---|
| Denoiser / DiT / UNet | `HartsyInference.Diffusion/Models/Denoisers/` | `UNet`, `ZImageConfig` |
| Text encoders | `HartsyInference.Diffusion/Models/TextEncoders/` | `ClipTextEncoder`, `LlamaStyleEncoder` |
| VAE | `HartsyInference.Diffusion/Models/Vae/` | `VaeDecoder`, `VaeEncoder`, `VaeConfig` |
| The generate loop | `HartsyInference.Diffusion/Pipelines/` | `SdxlPipeline`, `ZImagePipeline` |
| Checkpoint → named weights | `HartsyInference.ModelAssets/CheckpointConverters/` | `SdxlCheckpointConverter` |
| Request records | `HartsyInference.Diffusion/Requests/` | `TextToImageRequest`, `ImageToImageRequest` |

`HartsyInference.Engine` **only constructs and drives**. If the architecture's pipeline class does not exist
yet, stop — that is a Diffusion-package task first (build the pipeline, validate it against the Python
reference per `VALIDATION_STRATEGY.md`), and only then come back here. Everything below assumes
`<Fam>Pipeline` already exists and produces correct pixels.

### A1. Add a catalog entry

**File:** `src/HartsyInference.Cli/Infra/ModelCatalog.cs`

**This is the key insight: the catalog `Id` IS the family slug the recipe matches on.** Recipes are *not*
keyed on the `ModelArchitecture` enum — that enum is a coarse tensor-signature sniff that only names eight
families, and it exists purely as a fallback for a raw `--model-path` with no catalog entry. See
`InferenceEngine.ResolveFamilyId`:

```csharp
private static string ResolveFamilyId(ModelSpec spec)
{
    if (spec.Catalog is not null)
        return spec.Catalog.Id;          // ← the catalog slug wins
    ModelArchitecture arch = PipelineFactory.DetectArchitecture(spec.LocalPath!);
    return arch switch { ModelArchitecture.Sdxl => "sdxl", /* …8 entries… */ _ => arch.ToString().ToLowerInvariant() };
}
```

So the string you put in `Id` here is the exact string your recipe's `Matches(familyId)` must accept.
Keep it lowercase-kebab (`"qwen-image"`, `"chroma-radiance"`, `"lance-image"`).

Short form, for a family whose checkpoint the user supplies:

```csharp
E("mynewfam", img, "My New Family 1.0", "single-stream DiT (T5-XXL)", vp),
```

Long form, when the model should be auto-fetchable — list every file that makes it runnable:

```csharp
new CatalogEntry
{
    Id = "krea2",
    Modality = img,
    DisplayName = "Krea 2 Turbo",
    Architecture = "Krea2 DiT (Qwen3-VL-4B + Qwen-Image VAE)",
    Status = ok,
    CliDrivable = false,
    Assets = new ModelAsset[]
    {
        new() { Repo = "Comfy-Org/Krea-2", RepoPath = "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
                TargetSubdir = "Stable-Diffusion/Krea2", Role = "transformer" },
        // …text encoder, vae…
    },
},
```

`Status` is one of `ModelStatus.Structural` (`st`), `ValidationPending` (`vp`), `Verified` (`ok`) — do not
claim `Verified` until real-weight parity is recorded in `docs/Checklists/PARITY_VERIFICATION.md`.
`CliDrivable = true` only once `hartsy image -m <id>` actually runs end to end.

### A2. Register side models

**File:** `src/HartsyInference.Engine/SideModels.cs`

Anything the checkpoint does *not* carry — text encoders, VAEs, tokenizer sidecars — becomes a
`ModelAsset` here, so families that share a component share the file on disk instead of re-downloading it.

```csharp
/// <summary>T5-XXL encoder-only fp8 (used by Flux.1 Dev, SD3 with T5, Chroma).</summary>
public static readonly ModelAsset T5XxlEnconly = new ModelAsset
{
    Repo = "mcmonkey/google_t5-v1_1-xxl_encoderonly",
    RepoPath = "t5xxl_fp8_e4m3fn.safetensors",
    TargetSubdir = "text_encoders",
    TargetName = "t5xxl_enconly.safetensors",   // canonical local name ≠ repo file name
    Role = "text encoder",
    Sha256 = "7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09"
};
```

- `TargetSubdir` is models-root-relative (`"text_encoders"`, `"VAE/QwenImage"`, `"Stable-Diffusion/Krea2"`).
- `TargetName` sets the **canonical on-disk name**, which is deliberately allowed to differ from the repo
  file name. Match the name ComfyUI/SwarmUI uses so the file is shared, not duplicated.
- `Sha256` is not optional in practice: a size-correct but byte-corrupt download loads as NaN and you will
  spend a day chasing a black image. Compute it once and pin it.

Resolve it from the recipe with the one-call helper (`Construct` is a synchronous seam, so block):

```csharp
string qwenPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen3_4B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
string vaePath  = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe,   onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
```

`EnsureSideModelAsync` takes a per-target lock, skips a file already present, stages atomically, and
SHA-256-verifies before the file appears at its canonical path.

### A3. Write the recipe

**File:** `src/HartsyInference.Engine/Recipes/Image/<Fam>Recipe.cs` (one public type per file)

Implement `IArchitectureRecipe`:

```csharp
public interface IArchitectureRecipe
{
    string Name { get; }
    bool Matches(string familyId);
    IRecipePipeline Construct(RecipeContext context);
    ImageFeatures Supports => ImageFeatures.None;   // opt-in, see A5
}
```

`RecipeContext` gives you `CheckpointPath`, `Backend`, optional `Components` (per-request VAE / text-encoder
overrides) and optional `Loras`.

**Reference example — `Recipes/Image/SdxlRecipe.cs`.** It is the architecture every other family is measured
against: load + convert the checkpoint, stage weights off the mmap, merge LoRA, build encoders/UNet/VAE, hand
the constructed pipeline to the recipe pipeline, dispose the loader in `finally`:

```csharp
(SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
    SdxlCheckpointConverter.LoadAndConvert(context.CheckpointPath);
MergedLoraStack? loraStack = null;
try
{
    Dictionary<string, Tensor> unetWeights = WeightStaging.ToOwnedF32(converted.UNet);
    // …clipL / clipG / vae…
    loraStack = LoraApplier.BuildAndApply(LoraResolver.Resolve(context.Loras), context.Backend,
        unetWeights: unetWeights, clipLWeights: clipLWeights, clipGWeights: clipGWeights);
    // …construct encoders, UNet, VaeDecoder, VaeEncoder, SdxlPipeline…
    return new SdxlRecipePipeline(pipeline, context.Backend, clipL, clipG, loraStack);
}
catch (Exception ex) { Logs.Error("[SdxlRecipe] Construction failed.", ex); loraStack?.Dispose(); throw; }
finally { loader.Dispose(); }   // safe: ToOwnedF32 copied every weight out of the mmap
```

**Live-encode-then-free example — `Recipes/Image/ZImageRecipe.cs`.** Use this shape when the checkpoint
carries only the transformer and the encoder/VAE arrive as side models: resolve the assets, load each with
its own `SafeTensorsLoader`, and dispose each loader on every failure path as soon as its weights are staged.
Z-Image also shows `<Fam>Config.FromWeights(...)` — deriving the model config from the tensors rather than
hard-coding it.

### A4. Write the recipe pipeline

**File:** `src/HartsyInference.Engine/Recipes/Image/<Fam>RecipePipeline.cs`

```csharp
public interface IRecipePipeline : IDisposable
{
    ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel);
}
```

The body is always the same four moves: map the native request, tokenize, bridge progress, call the
Diffusion pipeline. Map the request with the shared helper — do not hand-roll the seed clamp:

```csharp
string negative = request.NegativePrompt ?? "";
TextToImageRequest inner = RecipeRequestMapper.ToTextToImage(request, negative);
```

`RecipeRequestMapper` (`src/HartsyInference.Engine/Recipes/RecipeRequestMapper.cs`) is the single source of
truth for native→diffusion scalar mapping:

| Member | Contract |
|---|---|
| `MapSeed(long)` | Folds a 64-bit seed into the pipelines' 31-bit space; **negative → `null` = random**. |
| `MapSteps(int)` | Non-positive → `null`, so the pipeline's `GenerationDefaults` win. |
| `MapCfgScale(float)` | Non-positive → `null`, same reason. |
| `MapClipSkip(int)` | Non-positive → `null` (standard final CLIP layer). |
| `ToTextToImage(request, negative)` | The defaults-win core: prompt, negative, W/H, mapped Steps/Cfg/Seed. |

`TextToImageRequest` is a `record`, so a family that carries extra fields layers them on:

```csharp
return RecipeRequestMapper.ToTextToImage(request, negative) with
{
    Scheduler = request.Scheduler,
    ClipSkip = RecipeRequestMapper.MapClipSkip(request.ClipSkip),
    InitialNoise = plan.TakeVariationNoise(),
};
```

If your family deliberately passes `Steps`/`CfgScale` **verbatim** (SDXL, Boogu) or precomputes them from a
preset (Ideogram 4, Boogu), build the initializer by hand and use `MapSeed` only. That is intentional and
correct — do not "unify" it.

Progress bridging is uniform:

```csharp
Action<GenerationProgress> bridge = p =>
{
    cancel.ThrowIfCancellationRequested();
    progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
};
```

`Dispose()` must dispose everything the recipe handed you: the Diffusion pipeline, tokenizers, the merged
LoRA stack, and **any `SafeTensorsLoader` still held** — weights are mmap-backed and leak the mapping otherwise.

### A5. Register the recipe

**File:** `src/HartsyInference.Engine/Recipes/RecipeRegistry.cs` → `BuildDefaults()`

```csharp
private static List<IArchitectureRecipe> BuildDefaults() => new List<IArchitectureRecipe>
{
    new Image.SdxlRecipe(),
    new Image.ZImageRecipe(),
    // …
    new Image.MyNewFamRecipe(),
};
```

Resolution is first-match-wins over the list; `RecipeRegistry.Register(recipe)` inserts at the front for
tests and out-of-tree families. An unregistered family throws a `NotSupportedException` that names every
drivable family, so a missing registration is never a silent failure.

**Video** is the same shape one namespace over: implement `IVideoRecipe` / `IVideoRecipePipeline`
(`Generate(VideoRequest, …)` returning `IReadOnlyList<VideoFrame>`), put the files in
`Recipes/Video/`, and register in `VideoRecipeRegistry.BuildDefaults()`. Note that registry entries may be
parameterized — `new Video.WanVideoRecipe("wan")` plus one entry per Wan compat class id — when one recipe
class covers several catalog slugs.

### A6. Declare `Supports`

`IArchitectureRecipe.Supports` defaults to `ImageFeatures.None`, meaning **text-to-image only, and every
composition object on the request is rejected**. That is deliberate: `ImagesService` maps each set
composition object to an `ImageFeatures` bit and throws by name rather than silently ignoring it —

```csharp
ImageFeatures missing = requested & ~_engine.SupportedFeatures(spec);
if (missing != ImageFeatures.None)
    throw new NotSupportedException($"Model family '{InferenceEngine.FamilyIdFor(spec)}' does not support: {missing}.");
```

So only declare a bit once the wiring is genuinely in your recipe/pipeline. SDXL, the most complete family:

```csharp
public ImageFeatures Supports =>
    ImageFeatures.Lora | ImageFeatures.ControlNet | ImageFeatures.IpAdapter | ImageFeatures.Refiner
    | ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.VariationSeed;
```

Flags: `Lora`, `ControlNet`, `IpAdapter`, `Refiner`, `Img2Img`, `Inpaint`, `Regional`, `VariationSeed`.

### A7. Reuse, don't reimplement

Everything in `src/HartsyInference.Engine/Features/` already solves a problem you are about to hit:

| Helper | What it does |
|---|---|
| `RecipeRequestMapper` (in `Recipes/`) | Native→diffusion request mapping (A4). |
| `WeightStaging.ToOwnedF32` | Copies weights into owned F32 so the checkpoint mmap can be released. |
| `VaePrecisionHelper` | BF16 on Ampere+, F32 elsewhere — **never F16** (SDXL VAE overflows to NaN). |
| `LoaderVaeUtils` | Shared VAE construction / weight-prefix handling. |
| `LoraApplier` / `LoraResolver` | Resolve a `LoraStack` request into files and merge into weights. |
| `UnetCompositionPlan` | Resolves img2img / inpaint / regional / variation-seed into a ready plan. |
| `Img2ImgResolver`, `MaskResolver`, `VariationSeedResolver` | Individual pieces of the above. |
| `ControlNetResolver`, `IpAdapterResolver`, `ReduxResolver`, `RefinerResolver` | Per-feature asset resolution + caching. |
| `PromptConditioningResolver`, `PromptRegionParser`, `WeightedConditioning` | Weighted / regional prompt parsing. |
| `SamplingParamResolver`, `RequestExtras` | Sampler/scheduler names and `ImageRequest.Extra` reads. |
| `ModelFileLocator`, `TaesdResolver`, `EmbeddingResolver` | Path discovery for sidecar files. |

---

## B. A new audio model

**Directory:** `src/HartsyInference.Engine/Audio/` — one subfolder per sub-modality:
`Tts/`, `Stt/`, `Music/`, `Vc/`, `Fx/`.

Audio does **not** use recipes. It uses a **descriptor + runner** pattern:

1. **Descriptor** — a data object, not a class hierarchy. E.g. `Audio/Tts/TtsModelDescriptor.cs`:

   ```csharp
   internal sealed class TtsModelDescriptor
   {
       /// <summary>Maps the request's variant hint to a HuggingFace repo id.</summary>
       internal required Func<string, string> ResolveRepo { get; init; }

       /// <summary>Loads the model (downloading on first use) into a uniform runner.</summary>
       internal required Func<string, CancellationToken, Task<ITtsRunner>> LoadAsync { get; init; }
   }
   ```

2. **Runner** — the uniform execution interface for that sub-modality (`ITtsRunner`, `IMusicRunner`, …),
   so the service layer never knows which model it is driving.

3. **Register in the catalog** — `TtsCatalog` / `SttCatalog` / `MusicCatalog` / `VcCatalog` / `FxCatalog`.
   Per-model descriptors normally live next to the model in `Audio/<Mod>/Models/` and are exposed as a
   static `Descriptor` property, then wired into the registry dictionary:

   ```csharp
   private static Dictionary<string, TtsModelDescriptor> Registry => _registry ??= new(StringComparer.OrdinalIgnoreCase)
   {
       ["neutts"]   = NeuTtsModel.Descriptor,
       ["kyutaitts"]= KyutaiTtsModel.Descriptor,
       ["piper"]    = PiperModel.Descriptor,
       // …
   };
   ```

4. **Addressing** — `AudioModelSelector.Parse` splits the requested token into `id` and `variant` on the
   first `:` (`whisper:large-v3`, `acestep:turbo`); with no colon the whole token is passed through as the
   variant too, so a bare repo id still works. Your dictionary key is the `id`; your `ResolveRepo` receives
   the `variant`.

5. **Weights** — register file sets in `Audio/AudioWeightsCatalog.cs` as `ModelAsset`s and fetch through
   `ModelDownloader` (same per-target lock, SHA-256 verify, atomic move as image side models). That catalog
   also supports *alternate sources* keyed by save name — tried only when the primary download fails, so an
   install can prefer a small pre-converted repack but still succeed from the canonical repo. Never alias two
   variants onto one hash: every ACE-Step variant, for example, is a genuinely distinct checkpoint.

Also add a `ModelCatalog` entry (modality `Modality.Speech` / `Music` / `Transcribe`) so the model shows up
in `hartsy list` and `hartsy pull`.

---

## C. A new LLM

**Usually there is nothing to do.** `src/HartsyInference.Engine/Services/TextService.cs` is fully
config-driven: it peeks `general.architecture` from the GGUF metadata (`PeekArchitecture`), routes to the
SSM loader or the transformer loader, and picks up the chat template from the file. Drop a supported-family
`.gguf` on disk and `hartsy text --model-path model.gguf` runs it. A `ModelCatalog` entry (`Modality.Text`)
is worth adding so it is discoverable, but it is not required to run.

Work is only needed when the *architecture* is new — that is an `HartsyInference.LLM` task (add the model
class + GGUF key mapping there), not an Engine task.

**Limitation to know:** `TextService.LoadInto` requires a **local `.gguf` path**:

```csharp
if (string.IsNullOrEmpty(path))
    throw new HartsyInferenceException("Text model has no local path. Pass a .gguf file via the model spec …");
```

There is no safetensors preset path for LLMs in `TextService` — a safetensors LLM must be converted to GGUF
first. (Diffusion is the opposite: it is safetensors-first.) `TextService` also enforces a minimum
free-RAM-to-file-size ratio before loading, because the GGUF load dequantizes tensors into host memory.

---

## D. Testing it

**The CLI is the end-to-end harness.** A single `hartsy` invocation exercises the typed service, the recipe,
and the CLI dispatch path in one go — that is the point of it. Run `hartsy --help` for the live list; the
commands are configured in `src/HartsyInference.Cli/Program.cs`:

| Command | Purpose |
|---|---|
| `hartsy image "<prompt>"` | Diffusion generate |
| `hartsy text "<prompt>"` | LLM generate (streams tokens) |
| `hartsy video "<prompt>"` | Video generate (BMP frame sequence) |
| `hartsy music "<prompt>"` | Music generate (WAV) |
| `hartsy speak "<text>"` | TTS (WAV) |
| `hartsy transcribe <file.wav>` | STT |
| `hartsy vision <image>` | CLIP embedding / YOLO detection |
| `hartsy 3d <image>` | Mesh (GLB) |
| `hartsy world <image>` | Interactive world-model rollout |
| `hartsy list [modality]` | Catalog, `--verified` to filter |
| `hartsy models` | What is in the local cache |
| `hartsy pull <repo>` | Download / register a model |
| `hartsy preview <file>` | Show an image inline in the terminal |
| `hartsy` (no args) | Interactive REPL |

`hartsy image` flags (`src/HartsyInference.Cli/Commands/ImageCommand.cs`):

```
<prompt>                       positional
-m,  --model      <id>         catalog id (the family slug)
     --model-path <path>       explicit checkpoint; bypasses catalog download
-b,  --backend    <auto|cpu|cuda|vulkan>
-n,  --negative   <text>
     --width      <int>        default 1024
     --height     <int>        default 1024
     --steps      <int>        default 20
     --cfg        <float>      default 7.5
     --seed       <int>        default -1 (random)
-o,  --output     <path>
-q,  --quiet
```

Typical first run for a new family:

```bash
hartsy image "a fox in snow" --model-path /models/Stable-Diffusion/mynewfam.safetensors --steps 30 --seed 1234
hartsy image "a fox in snow" -m mynewfam --steps 30 --seed 1234     # once the catalog entry + assets land
```

Using `--model-path` alone exercises the `ModelArchitecture` sniff fallback; using `-m <id>` exercises the
catalog-slug path. **Test both** — they resolve the family id differently (see A1).

Then record the result: per-modality status docs indexed by `docs/Checklists/MODEL_STATUS.md`, and
real-weight parity in `docs/Checklists/PARITY_VERIFICATION.md`. Do not flip a catalog `Status` to `Verified`
without a parity entry.

---

## Gotchas

These are the ones that actually cost time on the lifts:

1. **The catalog `Id` is the family slug.** Recipes match on the catalog id, not the `ModelArchitecture`
   enum (which only names eight families and is only consulted when there is no catalog entry). A typo
   between `ModelCatalog.E("…")` and `Matches("…")` produces "no recipe lifted into the Engine yet" for a
   recipe that is right there in the registry.

2. **Side-model canonical names differ from repo file names.** That is what `ModelAsset.TargetName` is for
   (`t5xxl_fp8_e4m3fn.safetensors` in the repo, saved as `t5xxl_enconly.safetensors`). Match the name
   ComfyUI/SwarmUI already uses, or every user downloads a second copy of a 9 GB encoder.

3. **Recipes must dispose their loaders.** Weights are mmap-backed. `SafeTensorsLoader` must be disposed on
   *every* path — `finally` after `WeightStaging.ToOwnedF32` copied the weights out, and explicitly before
   each `throw` in a multi-loader `Construct` (see `ZImageRecipe`). A leaked loader holds the whole file
   mapped.

4. **`WeightStaging.ToOwnedF32` roughly doubles host RAM at load.** It upcasts to F32 and copies out of the
   mmap, so for a moment you hold the mapping plus a full F32 copy. Budget for it; stage and release
   component by component rather than converting everything up front on large checkpoints.

5. **`ImageRequest.CfgScale` defaults to `7.5f`, which is wrong for turbo/distilled families.** Z-Image
   Turbo wants `1.0`. Because the default is positive, `MapCfgScale` will *not* null it out and the
   pipeline's own `GenerationDefaults` will *not* win — the user gets 7.5 and a burned image. Either handle
   it explicitly in the recipe pipeline or document the flag (`--cfg 1.0`) for that family.

6. **`Supports` defaults to `None`.** A newly added family rejects every LoRA/ControlNet/img2img request by
   name until you declare the bit. This is intentional — silent ignoring was the failure mode this replaced —
   but it will look like a bug the first time you hit it.

7. **`Construct` is a synchronous seam.** Side-model resolution is async, so recipes block with
   `.GetAwaiter().GetResult()`. Do not "fix" this by making the interface async without changing the
   caching in `InferenceEngine.GetOrConstructRecipe`.

8. **Pipelines are cached per checkpoint path and reused across requests.** Anything you stash in the
   recipe pipeline is shared by every subsequent request for that model — keep per-request state on the
   stack, not in fields.
