# Engine prerequisites of a 3D host (Hartsy3D): spike

Scope: what the engine already offers a host that runs the 3D models (a SwarmUI extension is the first), what it lacks, the
additive API that closes the gap. Source snapshot: `origin/main` `524cd8ebb`
(2026-11-12), read against the handoff written at `3295deef4`. Nothing here is implemented; each section ends with the PR
that would do it. The host-side split is a rule: the engine owns everything that is not specific to one host (capabilities,
plans, contracts, preparation, post-processing, conversion, previews, asset status, progress, memory); a viewer, three.js or
host types never enter the engine.

## What is there

| Claim in the handoff | Today |
|---|---|
| `IMeshService.GenerateAsync(spec, MeshRequest, IProgress<StepPreview>, ct)` | Unchanged. `MeshResult { Data, Format, Parts }`; `Format` is `glb` or `ply`. HTTP `POST /v1/native/mesh` and `/stream` answer `{ data, format, parts, savedPath }`; `/stream` reports progress for every path except HairDeck, which runs `Task.Run` with no progress. |
| ThreeD toolkit types reachable from the engine | All public: `Mesh`, `MeshDecimator`, `MeshCleaner`, `UvUnwrapper`, `TextureBaker`, `Inpainter`, `Rasterizer`, `TriangleBvh`, `PbrMeshPostProcessor`, `PaintViewRenderer`, `GlbReader`, `GlbWriter`, `PbrGlbWriter`, `PartGlbWriter`, `SkinnedGlbWriter`, `ObjReader`, `ObjWriter`, `PlyWriter`, `SceneMeshReader`. Only the test assembly sees internals. No new visibility change is needed. |
| `IMemoryEstimationService` and mesh | Not covered, and wrong where it answers. `CheckpointMemoryProfile` sums every `*.safetensors` header in the folder as one "denoiser" and adds the generic image activation (1 GiB + 64 B per pixel). Measured on the CPU backend with the real headers: TRELLIS.2 reports 14,457 MiB (every flow in the folder summed, the texture flows included) and Pixal3D 22,068 MiB (its measured shape-only peak at 1024 is 9.3 GB); P3-SAM, Paint 2.1 and SAM 3D throw (`No .safetensors files found`, `Diffusers layout ... contained no component safetensors files`) because they ship pickles or `ckpt` files. `AssessAsync` also returns `Unknown` on a backend that reports no VRAM. |
| `IModelPrefetchService` | Handles `Transcribe`, `Speech` and `Music` only; `Modality.Mesh` returns `Unsupported`. The catalog downloader (`ModelDownloader`, `ModelAsset`) already knows every mesh asset's repo, path, folder and sha256. |
| Recipe describe routes | `POST /v1/native/recipes/describe` takes `image` or `video` only. `ParameterProvenance`, `ResolvedParameter`, `RecipeIdentity` and `VideoPlanIssue` are public and reusable. |
| Request mappers as one source of truth | They are not: `Trellis2RequestMapper`, `Pixal3DRequestMapper`, `HairDeckRequestMapper`, `Sam3dObjectsRequestMapper` hold the rejection rules and defaults, but `MeshService` re-checks per branch (`GeneratePaintAsync`, `GeneratePartAsync`, `GenerateRiggedAsync`, `GenerateSceneAsync` each carry their own `NotSupportedException` lines), and the ranges (grid values, texture size 16 to 8192, field of view) live in the pipelines' `ValidateRequest`. |
| Seeds | `request.Seed < 0 ? null : (int)request.Seed` is unchecked in the generic branch (TripoSR, Hunyuan3D, TRELLIS) and in `GenerateRiggedAsync` (UniRig): 2^32 + 5 runs as 5, and 2^31 to 2^32 - 1 wrap to a negative `int`. TRELLIS.2, Pixal3D, SAM 3D, P3-SAM and Paint throw `ArgumentOutOfRangeException` above `int.MaxValue`. HairDeck takes the full `long` (its `HairDeckOptions.Seed` is `long`). The resolved seed of a negative request is not returned anywhere (HairDeck draws it inside the mapper). |
| Progress | `StepPreview { Step, TotalSteps, Preview* }`, one counter per stage. Pixal3D and TRELLIS.2 report `GenerationProgress` per stage and the service forwards it as is, so the number resets between stages. HairDeck reports none; the preprocessing (SAM 3, BiRefNet, DINOv3) never reads the token (`HairDeckRuntime.Preprocessor.Process(source)` takes none). |
| Cancellation and OOM | `MeshService` does not run under `InferenceEngine.GenerateWithVramCleanup`; only `ImagesService` and the refiner do. A TRELLIS.2 or Pixal3D out-of-memory failure therefore leaves its half-loaded stage weights resident. `FreeMemory()` disposes the whole `MeshService`; there is no listing and no per-model free. |
| Models added since the handoff | Catalog ids `hunyuan3d-paint` (2.0 albedo) and `hunyuan3d-paint21` (2.1 PBR) texture a mesh; `hunyuan3d-part` (P3-SAM) segments a mesh into parts and fills `MeshResult.Parts`; `sam3d-objects` lifts an object from an image plus mask (alpha plane) to a splat PLY; `pixal3d` has `Texture` and F16 flow weights through `numerics.flowF16Weights`; `unirig` rigs. Hunyuan3D-2.1 shape and X-Part have no catalog id and no engine wiring (X-Part's ShapeVAE encoder, conditioner and DiT are ported; sampler, pipeline and wiring are not). `hunyuan3d-mv` is catalogued but has no mesh path. |

Frames today (read from code comments and the status doc, not measured on real output): TRELLIS.2 shape-only is rotated to Y-up in the
pipeline and the textured GLB converts Z-up to Y-up in `ToGlb`; Pixal3D exports in its sampling frame turned a half turn about Y;
Hunyuan3D and TripoSR emit the grid of their decoders, not metres; UniRig works in a normalised box of about [-1, 1]; SAM 3D splats stay
in the object frame; CoLa3D (`GenerateSceneAsync`) returns its objects in the scene mesh's frame; P3-SAM and Paint take a mesh in and keep the
frame it was given. No model records its frame or its unit.

## Proposed contract (additive only)

Rule for every field: a request that does not set it behaves as it does today, so a published consumer keeps working. New
result members are properties with `init` accessors; no positional record is extended (adding a positional parameter changes
the constructor signature). New services are new interfaces, exposed on `IInferenceEngine` as default-interface properties that
throw `NotSupportedException` for an external implementation, the pattern `VideoPlanning` and `MemoryEstimation` use.

### 1. Planning and description (`IMeshPlanningService`)

```csharp
public interface IMeshPlanningService
{
    IReadOnlyList<MeshModelDescriptor> DescribeAll();
    MeshModelDescriptor Describe(string modelId);
    Task<MeshPlan> PlanAsync(ModelSpec spec, MeshRequest request, CancellationToken cancel = default);
}
```

`MeshModelDescriptor`: `Id`, `DisplayName`, `Input` (`Image`, `SceneMesh`, `SceneMeshAndImage`, `ImageAndMask`), `Outputs` (set of
`Mesh`, `TexturedMesh`, `RiggedMesh`, `PartMesh`, `Splats`), `Status` (the catalog's `ModelStatus`), `Fields`
(`MeshFieldSpec` per `MeshRequest` field: accepted, default, legal values or range, and the unit), `Licence`
(summary, `Gated`, `Restrictions`), `VramClass`, `TimeClass`, `Frame`. `MeshPlan`: `Model`, `Descriptor`, `Settings`
(`IReadOnlyDictionary<string, ResolvedParameter>` through `ParameterProvenance`), `Issues` (the existing `VideoPlanIssue` type is
named for video; a mesh plan gets `MeshPlanIssue { Code, Severity, Message, Field }` with the same shape rather than reusing a
type whose name says video), `MissingArtifacts`, `Memory` (a `MemoryEstimate` and a `MemoryFit`), `IsValid`, `ThrowIfInvalid()`.

One source of truth: a `MeshModelProfile` per id (one file each, next to the mapper it extends) declares the fields, ranges and
defaults; `Describe` reads it, `PlanAsync` validates a request against it, and `MeshService` calls the same validation instead of
its per-branch lines. The mappers keep their `ToOptions`, and a test enumerates `MeshRequest`'s properties and fails when a field
has no entry in every profile, so a new field cannot ship undescribed.

HTTP: `GET /v1/native/mesh/describe` (all), `GET /v1/native/mesh/describe/{id}`, `POST /v1/native/mesh/plan` (body as
`NativeMeshRequest`, answers `200` with the plan or `422` with `{ issues, plan }`, as the video route does).

Memory: a mesh memory model (`MeshMemoryProfile`) gives per-stage phases from the catalog asset sizes and the measured peaks in
`MODEL_STATUS_3D.md` (stage by stage, unloading between stages), and feeds the existing `MemoryFitJudge`. `CheckpointMemoryProfile`
is left alone for the image and video families; `MemoryEstimationService` routes a `Modality.Mesh` spec to the mesh profile.

### 2. Asset status and fetching (`IMeshAssetService`, `IModelPrefetchService`)

`MeshAssetStatus GetStatus(ModelSpec)` lists each catalog asset with `Role`, `Required` (an asset whose role says optional, or
that only a request option needs, is optional with the option named), size from the sha-pinned catalog or the file, `Present`,
the located path, `Gated`, and `Fetchable` with a typed `MeshAssetBlocker` (`Gated` with the repo and the token variable,
`ManualPlacement` with the file and folder, `NoCatalogEntry`, `Licence`). The locator reuses `ModelDownloader.FindExisting` and the
legacy-name list, because the real install here keeps the shape decoder in `TRELLIS2-ShapeVAE/` and the DINOv3 encoder in
`DINOv3/`, not in `3D/trellis2`. `IModelPrefetchService.PrefetchAsync` gains a `Modality.Mesh` branch that fetches the fetchable
missing assets with progress and returns the blockers in a new `init` property `Blockers` on `ModelPrefetchResult` (not in the
positional list). Manual cases today: Pixal3D `naf_release.pth`, HairDeck's gated SAM 3 and DINOv3, Hunyuan3D-2.1 shape (no
catalog id).

### 3. Image preparation (`MeshRequest.Preprocess`, `IMeshPreparationService`)

`enum MeshPreprocess { None, Auto }` on a new `MeshRequest.Preprocess` property. Proposed default: `None`, so no existing
caller's bytes change; the host sends `Auto`. (The handoff suggests `Auto` as the default; flipping it changes the output of any
caller that already prepared a TripoSR, Hunyuan3D or TRELLIS input, so it is a decision for the user, below.) `Auto`: when the
image has no usable alpha, matte it with BiRefNet-matting (the checkpoint TRELLIS.2 and HairDeck already use; RMBG-2.0 is
CC BY-NC and gated, so it is not used); then apply the model's contract as the handoff and the status doc state it (each is checked
against the upstream preprocessing code when it is ported): Hunyuan3D `ImageProcessorV2` (recentre the foreground with a 15 % border,
composite on white, 512), TripoSR `resize_foreground` and a 0.5 gray composite, TRELLIS (v1) premultiplied alpha, crop, 518. TRELLIS.2, Pixal3D and HairDeck already matte and prepare in their runtimes, so
for them `Auto` only supplies the matte and does not repeat their work. SAM 3D takes its mask from the alpha plane and `Auto` keeps
it. `Task<PreparedMeshImage> PrepareImageAsync(ModelSpec, ImageData, MeshPreprocess, CancellationToken)` returns the image the
model will see (RGB plus alpha, the matte used, the notes). The ports are plain C# on `ImageData`; resize uses the engine's
existing bicubic or bilinear resampler, and the per-model contracts are pinned by tiny fixtures with hand-computed crops.

### 4. Viewer-ready output

`MeshRequest.PreviewTriangles` (`int`, 0 default = no preview; the handoff suggests 200,000, which doubles the work for a caller
that does not want it). `MeshRequest.Frame` (`enum MeshFrame { Native, GltfYUpMetres }`, default `Native` so today's bytes stay;
the host sends the normalised one). `MeshResult.Preview` (`byte[]?`, a decimated GLB, `MeshCleaner` then `MeshDecimator`, textured
meshes keep their material) and `MeshResult.Info` (`MeshInfo`: vertices, triangles, bounds, components, boundary and non-manifold
edges, watertight, has normals, UVs, texture, skin, animation names, byte size; for a splat PLY: `SplatCount` and bounds).
Normalisation applies one rotation, a uniform scale to metres and the recentring each model needs; each model's real size is
documented in its profile (`Frame`, `UnitScale`, `HeightMetres` where a model has a natural size). A test pins a known asymmetric
mesh per model to a Y-up, +Z-front result. Not measured yet: the front direction of every model's native output (no GLB of real
output is on disk here); it needs one real GLB per model, which the GPU slot request in the report covers.

### 5. Result envelope and metadata

`MeshResult` gains `init` properties: `Warnings` (`IReadOnlyList<string>`, empty), `Settings` (`MeshSettingsUsed`: model id, the
resolved seed, the effective `MeshRequest` fields as `ResolvedParameter`s, engine version, `Frame`), `Preview`, `Info`. The HTTP
`WithSavedPath` object re-emits the same names, so old clients ignore them. GLB `asset.extras` carries the settings (a
`hartsyinference` object) through a new optional `extras` argument on `GlbWriter` and `PbrGlbWriter` (existing overloads stay); PLY
gets `comment` lines. Seeds: one shared `MeshSeed.Resolve(long)` returns the 32-bit seed for every model and throws
`ArgumentOutOfRangeException` above `int.MaxValue`, replacing the unchecked cast. That tightens validation for a caller passing a
seed above `int.MaxValue` to TripoSR, Hunyuan3D, TRELLIS or UniRig (today it wraps silently, and 2^31 to 2^32 - 1 become negative seeds whose meaning depends on the sampler), so it is listed below. HairDeck keeps its
`long` range and says so in its profile.

### 6. Progress and cancellation

`StepPreview` gains `init` properties `Stage` (`string?`), `StageIndex`, `StageCount` and `OverallFraction` (`float?`, 0 to 1); `Step`
and `TotalSteps` keep their meaning (the current stage's counter). `StepPreview` is a `readonly record struct`, so the added fields are additive and cost each progress callback a slightly larger copy, which is negligible. `StepPreviewPayload` adds the same fields as optional JSON. Each
multi-stage pipeline declares its stages with relative weights (a measured share where one exists, equal otherwise) in its profile,
and a small `MeshProgress` adapter maps a pipeline's `GenerationProgress` to the stage, so the fraction never moves backwards. HairDeck
reports its stages (matte, image sampler, colour, chains) and `/stream` forwards them. Cancellation: the HairDeck preprocessor and the
SAM 3 and BiRefNet runs take the token and check it between their blocks. A cancelled run throws `OperationCanceledException`, returns
no partial result, and releases the stage weights (`Trellis2DeviceMemory.ReleaseAll`).

### P1 sketch (not part of the first PRs)

Conversion and thumbnails (`IMeshToolsService`: `Convert(byte[] glb, string format)` with the `GlbReader` non-indexed-primitive limit
reported as a warning or an error, never silently; `RenderThumbnail`; operations on GLB bytes: decimate, fill holes, remove floaters
and small components, recompute normals, smooth, recentre and rescale, each reporting what it could not keep, a texture first of all),
lifecycle (`MeshService` lists its loaded pipelines and frees one by id or all; one automatic retry after the free for
`OutOfVramException`, the same contract `GenerateWithVramCleanup` gives images, with the retry re-resolving the pipeline because the
cached one believes its weights are resident), and the same cleanup around every mesh generation.

## Decisions for the user

1. `Preprocess` default `None` (additive, proposed) or `Auto` (the handoff's suggestion; changes the bytes sent to TripoSR, Hunyuan3D
   and TRELLIS for callers that already prepared the image).
2. `Frame` default `Native` (additive, proposed) or the normalised frame (every current GLB consumer sees a rotated or rescaled model).
3. `PreviewTriangles` default 0 (proposed) or 200,000.
4. The seed range check on the generic branch and UniRig: a seed above `int.MaxValue` now throws instead of wrapping.
5. A mesh-specific issue type (`MeshPlanIssue`) instead of reusing `VideoPlanIssue`, whose name says video.

Items 1, 2 and 3 are not breaking in the form proposed. Item 4 is a validation tightening on paths that were already silently wrong. Item 5 is a naming decision that changes no behaviour.

## Order of the PRs

1. This spike (docs only).
2. Planning and description, with the profiles, the memory model, the describe and plan routes.
3. Asset status and mesh prefetch.
4. Image preparation.
5. Output: preview, info, frame.
6. Envelope, metadata and seed.
7. Progress, stream and cancellation.
Then P1 (conversion, thumbnails, operations, lifecycle with the gate and the OOM retry), and P2 if time allows.
