using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Face;
using HartsyInference.Vision.FaceDetection;
using EngineClipPreprocessor = HartsyInference.Vision.Clip.ClipImagePreprocessor;
using EngineIpAdapter = HartsyInference.Diffusion.Adapters.IpAdapter;
using ImageDataRef = HartsyInference.Engine.Requests.ImageData;
using ImagePromptRequest = HartsyInference.Engine.Requests.IpAdapter;

namespace HartsyInference.Engine.Features;

/// <summary>Resolves the request's image-prompt conditioning into <see cref="IpAdapterConditioning"/>s for an SDXL or
/// SD 1.5 pipeline: loads the IPA checkpoint, runs the variant's image encoder over the reference images (CLIP-Vision for
/// standard/Plus, ArcFace face embedding for FaceID), projects the result into image-prompt tokens, and returns the
/// per-cross-attn wiring.
///
/// <para><b>Scope:</b> SDXL + SD 1.5; standard + Plus + Plus-Face + FaceID + FaceID-Plus/Plus-v2. For FaceID the face is
/// located with a dedicated YOLOv8-Face detector when installed (else YOLO11-pose keypoints), aligned to the ArcFace
/// 112×112 template, and embedded with the in-engine ArcFace IR-50. FaceID-Plus additionally renders the same alignment
/// at 224×224 and feeds its CLIP-Vision-H penultimate hidden states to the two-input projection; the Plus-v2 shortcut
/// strength comes from <see cref="ImagePromptRequest.FaceIdV2Weight"/>. All FaceID checkpoints also ship a rank-128 UNet
/// LoRA whose path is surfaced on <see cref="ResolvedSpec.FaceIdLoraPath"/> so the caller merges it through the normal
/// LoRA path. Single adapter only — stacking multiple IPA models is deferred.</para></summary>
public static class IpAdapterResolver
{
    /// <summary>Converted ArcFace recognition backbone expected under an <c>ipadapter</c> folder. Source: buffalo_l
    /// <c>w600k_r50.onnx</c> converted with <c>tests/python-reference/convert_arcface_onnx.py</c>.</summary>
    public const string ArcFaceWeightsFile = "arcface_w600k_r50.safetensors";

    /// <summary>Folded YOLO11n-pose weights used for the fallback face-keypoint alignment.</summary>
    public const string PoseWeightsFile = "yolo11n-pose-folded.safetensors";

    /// <summary>Side length of the CLIP-Vision face crop FaceID-Plus consumes (insightface <c>norm_crop(image_size=224)</c>).</summary>
    private const int ClipFaceCropSize = 224;

    /// <summary>Models-root-relative folders searched for IP-Adapter checkpoints.</summary>
    private static readonly string[] _ipaFolders = ["ipadapter", "IpAdapter", "IPAdapter", "ip_adapter"];

    /// <summary>Models-root-relative folders searched for the pose / face detector checkpoints.</summary>
    private static readonly string[] _detectorFolders = ["facedetection", "yolov8-face", "face", "ipadapter", "text_encoders", "clip_vision"];

    /// <summary>A known FaceID checkpoint plus its companion UNet-LoRA half, both from <c>h94/IP-Adapter-FaceID</c>.</summary>
    private sealed record FaceIdDownload(string BinFile, string BinSha, string LoraFile, string LoraSha)
    {
        /// <summary>Download descriptor for the adapter half.</summary>
        public ModelAsset BinAsset => new ModelAsset
        {
            Repo = "h94/IP-Adapter-FaceID",
            RepoPath = BinFile,
            TargetSubdir = "ipadapter",
            Role = "ip-adapter",
            Sha256 = BinSha,
        };

        /// <summary>Download descriptor for the companion UNet LoRA half.</summary>
        public ModelAsset LoraAsset => new ModelAsset
        {
            Repo = "h94/IP-Adapter-FaceID",
            RepoPath = LoraFile,
            TargetSubdir = "ipadapter",
            Role = "lora",
            Sha256 = LoraSha,
        };
    }

    private static readonly FaceIdDownload[] _knownFaceIdDownloads =
    [
        new FaceIdDownload("ip-adapter-faceid_sdxl.bin", "f455fed24e207c878ec1e0466b34a969d37bab857c5faa4e8d259a0b4ff63d7e",
            "ip-adapter-faceid_sdxl_lora.safetensors", "4fcf93d6e8dc8dd18f5f9e51c8306f369486ed0aa0780ade9961308aff7f0d64"),
        new FaceIdDownload("ip-adapter-faceid_sd15.bin", "201344e22e6f55849cf07ca7a6e53d8c3b001327c66cb9710d69fd5da48a8da7",
            "ip-adapter-faceid_sd15_lora.safetensors", "70699f0dbfadd47de1f81d263cf4c86bd4b7271d841304af9b340b3a7f38e86a"),
        new FaceIdDownload("ip-adapter-faceid-plusv2_sdxl.bin", "c6945d82b543700cc3ccbb98d363b837e9c596281607857c74b713a876daf5fb",
            "ip-adapter-faceid-plusv2_sdxl_lora.safetensors", "f24b4bb2dad6638a09c00f151cde84991baf374409385bcbab53c1871a30cb7b"),
        new FaceIdDownload("ip-adapter-faceid-plusv2_sd15.bin", "26d0d86a1d60d6cc811d3b8862178b461e1eeb651e6fe2b72ba17aa95411e313",
            "ip-adapter-faceid-plusv2_sd15_lora.safetensors", "8abff87a15a049f3e0186c2e82c1c8e77783baf2cfb63f34c412656052eb57b0"),
        new FaceIdDownload("ip-adapter-faceid-plus_sd15.bin", "252fb53e0d018489d9e7f9b9e2001a52ff700e491894011ada7cfb471e0fadf2",
            "ip-adapter-faceid-plus_sd15_lora.safetensors", "3f00341d11e5e7b5aadf63cbdead09ef82eb28669156161cf1bfc2105d4ff1cd"),
    ];

    /// <summary>One generation's resolved IPA state. Owns the projected image-prompt tokens; the loaded adapter and image
    /// encoder live in an <see cref="IpAdapterCacheEntry"/> the caller's cache holds across generations.</summary>
    public sealed class ResolvedSpec : IDisposable
    {
        /// <summary>The conditionings to hand to the pipeline.</summary>
        public required List<IpAdapterConditioning> Conditionings { get; init; }

        /// <summary>The projected token tensors backing <see cref="Conditionings"/>.</summary>
        public required List<Tensor> ImageTokens { get; init; }

        /// <summary>Path of the FaceID companion UNet LoRA (kohya format) to merge, or null for non-FaceID adapters and
        /// when the companion file couldn't be located.</summary>
        public string? FaceIdLoraPath { get; init; }

        /// <summary>Merge strength for <see cref="FaceIdLoraPath"/>; 1.0 matches the official FaceID pipeline default.</summary>
        public float FaceIdLoraStrength { get; init; } = 1.0f;

        /// <summary>Frees the token tensors.</summary>
        public void Dispose()
        {
            foreach (Tensor t in ImageTokens)
            {
                t.Dispose();
            }
        }
    }

    /// <summary>Resolves IPA for this generation, or null when no adapter is named. <paramref name="baseModel"/> selects
    /// which UNet family the checkpoint must match — mismatches throw.</summary>
    /// <param name="imagePrompt">Reference images plus the FaceID-v2 weight; null or empty images throws when an adapter is named.</param>
    /// <param name="adapterModel">IPA model id or path; null/"None" disables the feature.</param>
    /// <param name="clipVisionModel">CLIP-Vision override; null auto-downloads <see cref="SideModels.ClipVisionH14"/>.</param>
    /// <param name="weightType">Per-layer weight ramp name (e.g. "standard"), passed through to the adapter.</param>
    public static async Task<ResolvedSpec?> ResolveAsync(
        ImagePromptRequest? imagePrompt,
        string? adapterModel,
        string? clipVisionModel,
        double weight,
        double startFraction,
        double endFraction,
        string weightType,
        IBackend backend,
        IpAdapterBaseModel baseModel,
        Action<string> log,
        Func<string, IpAdapterCacheEntry?> cacheLookup,
        Action<IpAdapterCacheEntry> cachePut,
        CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(log);
        ArgumentNullException.ThrowIfNull(cacheLookup);
        ArgumentNullException.ThrowIfNull(cachePut);
        if (string.IsNullOrWhiteSpace(adapterModel) || adapterModel == "None")
        {
            return null;
        }
        IReadOnlyList<ImageDataRef>? promptImages = imagePrompt?.PromptImages;
        if (promptImages is null || promptImages.Count == 0)
        {
            throw new InvalidOperationException(
                "IP-Adapter is enabled but no prompt image was provided. Add a reference image or clear the adapter selection.");
        }
        double startRaw = Math.Clamp(startFraction, 0.0, 1.0);
        double endRaw = Math.Clamp(endFraction, 0.0, 1.0);
        if (endRaw < startRaw)
        {
            endRaw = startRaw;
        }
        if (imagePrompt is not null && !imagePrompt.Grouping && promptImages.Count > 1)
        {
            log("  note: Grouping=false with multiple reference images — the encoder outputs are still averaged into one conditioning (per-image stacking is not wired).");
        }

        string ipaPath = await ResolveAdapterPathAsync(adapterModel, log, ct).ConfigureAwait(false);

        IpAdapterCacheEntry? entry = cacheLookup(ipaPath);
        if (entry is null)
        {
            log($"Loading IP-Adapter: {adapterModel}");
            entry = await LoadIpaEntryAsync(backend, ipaPath, clipVisionModel, baseModel, log, ct).ConfigureAwait(false);
            cachePut(entry);
        }
        else
        {
            log($"IP-Adapter '{adapterModel}' (cached).");
        }
        if (entry.IpAdapter.Config.BaseModel != baseModel)
        {
            throw new InvalidOperationException(
                $"IP-Adapter '{adapterModel}' is for base={entry.IpAdapter.Config.BaseModel}, but the current pipeline expects {baseModel}.");
        }

        // Run the variant's image encoder over all reference images (averaging), then project ONCE.
        Tensor imageTokens;
        if (entry.IpAdapter.Config.IsFaceId && entry.IpAdapter.Config.IsPlus)
        {
            // FaceID-Plus / Plus-v2: ArcFace embedding + CLIP-Vision hidden states of the SAME aligned face, mixed by the
            // two-input projection. The v2 shortcut weight follows the official default of 1.0.
            double v2Weight = imagePrompt?.FaceIdV2Weight ?? 1.0;
            (Tensor faceEmbeds, Tensor? clipHidden) = EmbedFacesCore(backend, entry, promptImages, wantClipCrop: true, log);
            try
            {
                imageTokens = entry.IpAdapter.ProjectImage(backend, faceEmbeds, clipHidden!, (float)v2Weight);
            }
            finally
            {
                faceEmbeds.Dispose();
                clipHidden?.Dispose();
            }
            if (entry.IpAdapter.Config.IsFaceIdV2)
            {
                log($"  FaceID V2 weight: {v2Weight:F2}");
            }
        }
        else
        {
            Tensor encoderOut = entry.IpAdapter.Config.IsFaceId
                ? EmbedFaces(backend, entry, promptImages, log)
                : AverageVisionOutputs(backend, entry, promptImages, log);
            try
            {
                imageTokens = entry.IpAdapter.ProjectImage(backend, encoderOut);
            }
            finally
            {
                encoderOut.Dispose();
            }
        }

        List<IpAdapterConditioning> conditionings =
        [
            new IpAdapterConditioning
            {
                Adapter = entry.IpAdapter,
                ImageTokens = imageTokens,
                Scale = (float)weight,
                WeightType = string.IsNullOrWhiteSpace(weightType) ? "standard" : weightType,
                StartFraction = (float)startRaw,
                EndFraction = (float)endRaw,
            },
        ];
        entry.LastUsedUtc = DateTime.UtcNow;
        log($"IP-Adapter ready: variant={VariantName(entry.IpAdapter.Config)}, base={baseModel}, weight={weight:F2}, "
            + $"weightType={weightType}, window=[{startRaw:F2}, {endRaw:F2}], tokens={entry.IpAdapter.NumImageTokens}.");
        return new ResolvedSpec
        {
            Conditionings = conditionings,
            ImageTokens = [imageTokens],
            FaceIdLoraPath = entry.FaceIdLoraPath,
        };
    }

    /// <summary>Locates the adapter on disk, auto-downloading the known h94 FaceID checkpoints when the name matches one.</summary>
    private static async Task<string> ResolveAdapterPathAsync(string adapterModel, Action<string> log, CancellationToken ct)
    {
        string? path = ModelFileLocator.Find(adapterModel, _ipaFolders);
        if (path is not null)
        {
            return path;
        }
        if (TryGetKnownFaceIdDownload(adapterModel, out FaceIdDownload? dl))
        {
            log($"Downloading IP-Adapter FaceID checkpoint: {dl!.BinFile}");
            return await ModelDownloader.EnsureSideModelAsync(dl.BinAsset, null, ct).ConfigureAwait(false);
        }
        throw new InvalidOperationException(
            $"IP-Adapter model '{adapterModel}' was not found under '{Path.Combine(RepoPaths.ModelsRoot(), "ipadapter")}'.");
    }

    /// <summary>FaceID image encoding: per reference image detect the strongest face, align it to the ArcFace template,
    /// embed with ArcFace IR-50, then average the L2-normalized embeddings and renormalize. Returns <c>[1, 512]</c>.</summary>
    private static Tensor EmbedFaces(IBackend backend, IpAdapterCacheEntry entry, IReadOnlyList<ImageDataRef> images, Action<string> log)
    {
        (Tensor faceEmbeds, Tensor? clipHidden) = EmbedFacesCore(backend, entry, images, wantClipCrop: false, log);
        clipHidden?.Dispose();
        return faceEmbeds;
    }

    private static unsafe (Tensor FaceEmbeds, Tensor? ClipHidden) EmbedFacesCore(
        IBackend backend, IpAdapterCacheEntry entry, IReadOnlyList<ImageDataRef> images, bool wantClipCrop, Action<string> log)
    {
        ArcFaceModel arcFace = entry.ArcFace
            ?? throw new InvalidOperationException("FaceID cache entry has no ArcFace model (loader bug).");
        YoloPosePipeline pose = entry.PosePipeline
            ?? throw new InvalidOperationException("FaceID cache entry has no pose pipeline (loader bug).");
        if (wantClipCrop && entry.ClipVision is null)
        {
            throw new InvalidOperationException("FaceID-Plus cache entry has no CLIP-Vision encoder (loader bug).");
        }

        float[] accumulator = new float[ArcFaceModel.EmbeddingDim];
        Tensor? clipAccumulator = null;
        long clipCount = 0;
        backend.PreloadWeights(arcFace.EnumerateWeights());
        try
        {
            foreach (ImageDataRef image in images)
            {
                byte[] rgb = image.Rgb;
                int width = image.Width;
                int height = image.Height;
                (byte[] crop, byte[]? clipCrop) = DetectAndAlignFace(entry.FaceDetector, pose, rgb, width, height, wantClipCrop, log);
                Tensor inputTensor = ArcFaceModel.PreprocessAligned(crop);
                Tensor embed;
                try
                {
                    embed = arcFace.EmbedNormalized(backend, inputTensor);
                }
                finally
                {
                    inputTensor.Dispose();
                }
                try
                {
                    float* ep = (float*)embed.DataPointer;
                    for (int d = 0; d < ArcFaceModel.EmbeddingDim; d++)
                    {
                        accumulator[d] += ep[d];
                    }
                }
                finally
                {
                    embed.Dispose();
                }

                if (!wantClipCrop)
                {
                    continue;
                }
                EngineClipPreprocessor preprocess = new EngineClipPreprocessor(ClipFaceCropSize);
                Tensor pixels = preprocess.Preprocess(clipCrop!, ClipFaceCropSize, ClipFaceCropSize);
                Tensor hidden;
                try
                {
                    hidden = entry.ClipVision!.EncodeHiddenStates(backend, pixels);
                }
                finally
                {
                    pixels.Dispose();
                }
                if (clipAccumulator is null)
                {
                    clipAccumulator = hidden;
                    clipCount = hidden.ElementCount;
                }
                else
                {
                    try
                    {
                        if (hidden.Shape != clipAccumulator.Shape)
                        {
                            throw new InvalidOperationException(
                                $"CLIP-Vision face-crop output shape mismatch across reference images: {clipAccumulator.Shape} vs {hidden.Shape}.");
                        }
                        float* ap = (float*)clipAccumulator.DataPointer;
                        float* hp = (float*)hidden.DataPointer;
                        for (long e = 0; e < clipCount; e++)
                        {
                            ap[e] += hp[e];
                        }
                    }
                    finally
                    {
                        hidden.Dispose();
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Logs.Error("[Features][IPA] Face embedding failed.", ex);
            clipAccumulator?.Dispose();
            throw;
        }
        finally
        {
            backend.FreeWeights(arcFace.EnumerateWeights());
        }

        if (images.Count > 1)
        {
            log($"  averaged {images.Count} face embeddings (renormalized identity centroid)"
                + (wantClipCrop ? " + CLIP face-crop hidden states (mean)" : ""));
        }
        double norm = 0;
        for (int d = 0; d < ArcFaceModel.EmbeddingDim; d++)
        {
            norm += (double)accumulator[d] * accumulator[d];
        }
        float inv = (float)(1.0 / Math.Max(Math.Sqrt(norm), 1e-12));
        Tensor result = new Tensor(new TensorShape(1, ArcFaceModel.EmbeddingDim), DType.F32);
        float* rp = (float*)result.DataPointer;
        for (int d = 0; d < ArcFaceModel.EmbeddingDim; d++)
        {
            rp[d] = accumulator[d] * inv;
        }
        if (wantClipCrop && images.Count > 1 && clipAccumulator is not null)
        {
            float invN = 1.0f / images.Count;
            float* cp = (float*)clipAccumulator.DataPointer;
            for (long e = 0; e < clipCount; e++)
            {
                cp[e] *= invN;
            }
        }
        return (result, clipAccumulator);
    }

    private static string VariantName(IpAdapterConfig config) => config.IsFaceId
        ? config.IsPlus ? (config.IsFaceIdV2 ? "FaceID-PlusV2" : "FaceID-Plus") : "FaceID"
        : config.IsPlus ? "Plus" : "Standard";

    /// <summary>Detects people, picks the best face (largest inter-eye distance among detections with usable eye+nose
    /// keypoints, else the highest-confidence person), and returns an ArcFace-aligned 112×112 RGB crop plus — when
    /// <paramref name="wantClipCrop"/> — the SAME alignment at 224×224 for CLIP-Vision. Falls back to an unrotated square
    /// face crop when keypoints are missing, and to a center crop when no person is detected at all.</summary>
    private static (byte[] ArcCrop, byte[]? ClipCrop) DetectAndAlignFace(
        FaceDetector? faceDetector, YoloPosePipeline pose, byte[] rgb, int width, int height, bool wantClipCrop, Action<string> log)
    {
        // Preferred path: the dedicated YOLOv8-Face detector → 5 landmarks → ArcFace-template alignment. It is a proper
        // face localizer + landmark regressor (vs the pose model's whole-body keypoints), so the crop is more identity-faithful.
        if (faceDetector is not null)
        {
            IReadOnlyList<DetectedFace> faces = faceDetector.DetectFaces(rgb, width, height, confidenceThreshold: 0.25f, iouThreshold: 0.45f);
            if (faces.Count > 0)
            {
                DetectedFace best = faces[0];
                float bestArea = best.Box.Area;
                foreach (DetectedFace f in faces)
                {
                    if (f.Box.Area > bestArea)
                    {
                        best = f;
                        bestArea = f.Box.Area;
                    }
                }
                log($"  FaceID: detected {faces.Count} face(s) via YOLOv8-Face — aligning the largest (conf={best.Confidence:F2}).");
                return (FaceDetector.AlignedCrop(rgb, width, height, best, ArcFaceModel.InputSize),
                    wantClipCrop ? FaceDetector.AlignedCrop(rgb, width, height, best, ClipFaceCropSize) : null);
            }
            log("  FaceID: YOLOv8-Face found no faces — falling back to pose-keypoint alignment.");
        }

        IReadOnlyList<PoseDetection> people = pose.Detect(rgb, width, height, confidenceThreshold: 0.25f, iouThreshold: 0.45f);
        PoseDetection? bestAligned = null;
        float[]? bestPoints = null;
        float bestEyeDist = -1f;
        PoseDetection? bestAny = null;
        foreach (PoseDetection person in people)
        {
            if (bestAny is null || person.Confidence > bestAny.Confidence)
            {
                bestAny = person;
            }
            if (FaceAlignment.TryGetAlignmentPoints(person, visThreshold: 0.3f, out float[] pts))
            {
                float dx = pts[2] - pts[0];
                float dy = pts[3] - pts[1];
                float eyeDist = MathF.Sqrt(dx * dx + dy * dy);
                if (eyeDist > bestEyeDist)
                {
                    bestEyeDist = eyeDist;
                    bestAligned = person;
                    bestPoints = pts;
                }
            }
        }
        if (bestAligned is not null && bestPoints is not null)
        {
            return (FaceAlignment.AlignToTemplate(rgb, width, height, bestPoints),
                wantClipCrop ? FaceAlignment.AlignToTemplate(rgb, width, height, bestPoints, outputSize: ClipFaceCropSize) : null);
        }
        if (bestAny is not null)
        {
            log("  FaceID: face keypoints not visible — falling back to unrotated square face crop.");
            PoseFaceCrop.Rect rect = PoseFaceCrop.ComputeSquareCrop(bestAny, width, height, expand: 1.6f);
            return (SquareCropTo(rgb, width, height, rect.X, rect.Y, rect.Size, FaceAlignment.CropSize),
                wantClipCrop ? SquareCropTo(rgb, width, height, rect.X, rect.Y, rect.Size, ClipFaceCropSize) : null);
        }
        log("  FaceID: WARNING — no person detected in the reference image; using a center crop. Identity transfer will be weak.");
        float side = Math.Min(width, height);
        float cx = (width - side) * 0.5f;
        float cy = (height - side) * 0.5f;
        return (SquareCropTo(rgb, width, height, cx, cy, side, FaceAlignment.CropSize),
            wantClipCrop ? SquareCropTo(rgb, width, height, cx, cy, side, ClipFaceCropSize) : null);
    }

    /// <summary>Scales a square source region to an <paramref name="outSize"/>² crop via the shared affine warp.</summary>
    private static byte[] SquareCropTo(byte[] rgb, int width, int height, float x, float y, float side, int outSize)
    {
        float s = outSize / Math.Max(side, 1f);
        FaceAlignment.Affine2x3 srcToDst = new FaceAlignment.Affine2x3(s, 0f, -x * s, 0f, s, -y * s);
        return FaceAlignment.WarpAffine(rgb, width, height, srcToDst, outSize, outSize);
    }

    /// <summary>Runs CLIP-Vision on each reference image and averages the outputs. All inputs share a shape after the
    /// preprocess, so the mean has the same shape as a single encode — this merges multiple references into one conditioning.</summary>
    private static unsafe Tensor AverageVisionOutputs(
        IBackend backend, IpAdapterCacheEntry entry, IReadOnlyList<ImageDataRef> images, Action<string> log)
    {
        ClipVisionEncoder clipVision = entry.ClipVision
            ?? throw new InvalidOperationException("IP-Adapter cache entry has no CLIP-Vision encoder (loader bug).");
        if (images.Count > 1)
        {
            log($"  averaging {images.Count} reference images (vision-output mean before projection)");
        }
        EngineClipPreprocessor preprocess = new EngineClipPreprocessor(clipVision.Config.ImageSize);
        Tensor firstPixels = preprocess.Preprocess(images[0].Rgb, images[0].Width, images[0].Height);
        Tensor accumulator;
        try
        {
            accumulator = entry.IpAdapter.Config.IsPlus
                ? clipVision.EncodeHiddenStates(backend, firstPixels)
                : clipVision.EncodeImageEmbeds(backend, firstPixels);
        }
        finally
        {
            firstPixels.Dispose();
        }
        long count = accumulator.ElementCount;
        for (int i = 1; i < images.Count; i++)
        {
            Tensor pixels = preprocess.Preprocess(images[i].Rgb, images[i].Width, images[i].Height);
            Tensor next;
            try
            {
                next = entry.IpAdapter.Config.IsPlus
                    ? clipVision.EncodeHiddenStates(backend, pixels)
                    : clipVision.EncodeImageEmbeds(backend, pixels);
            }
            finally
            {
                pixels.Dispose();
            }
            try
            {
                if (next.Shape != accumulator.Shape || next.DType != accumulator.DType)
                {
                    throw new InvalidOperationException(
                        $"CLIP-Vision output shape mismatch across reference images: {accumulator.Shape} vs {next.Shape}.");
                }
                float* ap = (float*)accumulator.DataPointer;
                float* np = (float*)next.DataPointer;
                for (long e = 0; e < count; e++)
                {
                    ap[e] += np[e];
                }
            }
            finally
            {
                next.Dispose();
            }
        }
        float invN = 1.0f / images.Count;
        float* aPtr = (float*)accumulator.DataPointer;
        for (long e = 0; e < count; e++)
        {
            aPtr[e] *= invN;
        }
        return accumulator;
    }

    private static bool TryGetKnownFaceIdDownload(string requestedName, out FaceIdDownload? download)
    {
        string bare = Path.GetFileNameWithoutExtension(requestedName).ToLowerInvariant();
        foreach (FaceIdDownload dl in _knownFaceIdDownloads)
        {
            if (Path.GetFileNameWithoutExtension(dl.BinFile).Equals(bare, StringComparison.Ordinal))
            {
                download = dl;
                return true;
            }
        }
        download = null;
        return false;
    }

    /// <summary>Loads and constructs the adapter plus its image encoder. Standard/Plus get CLIP-Vision (auto-downloaded
    /// ViT-H/14 unless overridden); FaceID gets the ArcFace IR-50 + YOLO11-pose pair and its companion UNet LoRA path.</summary>
    private static async Task<IpAdapterCacheEntry> LoadIpaEntryAsync(
        IBackend backend, string ipaPath, string? clipVisionModel, IpAdapterBaseModel expectedBase, Action<string> log, CancellationToken ct)
    {
        IpAdapterFile file = IpAdapterLoader.Load(ipaPath);
        SafeTensorsLoader? clipVisionLoader = null;
        SafeTensorsLoader? arcFaceLoader = null;
        YoloPosePipeline? posePipeline = null;
        FaceDetector? faceDetector = null;
        try
        {
            if (file.BaseModel != IpAdapterBaseModel.Sdxl && file.BaseModel != IpAdapterBaseModel.Sd15)
            {
                throw new InvalidOperationException(
                    $"IP-Adapter '{Path.GetFileName(ipaPath)}' detected as base={file.BaseModel}. Only SDXL and SD 1.5 IP-Adapters are "
                    + "supported; Flux IPA uses a DiT cross-attention layout that needs a separate adapter implementation.");
            }
            if (file.BaseModel != expectedBase)
            {
                throw new InvalidOperationException(
                    $"IP-Adapter '{Path.GetFileName(ipaPath)}' is for base={file.BaseModel}, but the current generation uses base={expectedBase}.");
            }
            log($"  variant: {VariantName(file.Config)}, base={file.BaseModel}, tokens={file.Config.NumImageTokens}");

            EngineIpAdapter adapter = new EngineIpAdapter(file.Config);
            adapter.LoadWeights(file.Weights);
            log($"  loaded {adapter.CrossAttentionLayerCount} per-cross-attn projections.");

            if (file.Config.IsFaceId)
            {
                string arcFacePath = ModelFileLocator.Find(ArcFaceWeightsFile, _ipaFolders)
                    ?? throw new InvalidOperationException(
                        $"IP-Adapter FaceID needs the ArcFace face-embedding weights at "
                        + $"'{Path.Combine(RepoPaths.ModelsRoot(), "ipadapter", ArcFaceWeightsFile)}'. Convert insightface buffalo_l's "
                        + "w600k_r50.onnx with tests/python-reference/convert_arcface_onnx.py.");
                arcFaceLoader = new SafeTensorsLoader();
                arcFaceLoader.Load(arcFacePath);
                ArcFaceModel arcFace = new ArcFaceModel();
                arcFace.LoadWeights(arcFaceLoader.GetAllTensors());
                log($"  ArcFace: {Path.GetFileName(arcFacePath)}");

                string posePath = ModelFileLocator.Find(PoseWeightsFile, _detectorFolders)
                    ?? throw new InvalidOperationException(
                        $"IP-Adapter FaceID needs the folded YOLO11n-pose weights ('{PoseWeightsFile}') under the models root. "
                        + "Convert Ultralytics 'yolo11n-pose.pt' with tests/python-reference/convert_yolov8_pt_to_safetensors.py.");
                posePipeline = new YoloPosePipeline(backend, YoloConfig.YoloV11nPose, posePath, inputSize: 640);

                // Optional dedicated face detector (YOLOv8-Face): when installed it replaces the pose-keypoint heuristic.
                faceDetector = TryLoadFaceDetector(backend, log);

                string? loraPath = await ResolveFaceIdLoraAsync(ipaPath, ct).ConfigureAwait(false);
                if (loraPath is null)
                {
                    log("  WARNING: FaceID companion LoRA not found — identity likeness will be much weaker. "
                        + "Place the matching *_lora.safetensors next to the FaceID checkpoint.");
                }
                ClipVisionEncoder? faceClipVision = null;
                if (file.Config.IsPlus)
                {
                    (faceClipVision, clipVisionLoader) = await LoadClipVisionAsync(clipVisionModel, log, ct).ConfigureAwait(false);
                }
                return new IpAdapterCacheEntry
                {
                    FilePath = ipaPath,
                    File = file,
                    IpAdapter = adapter,
                    ClipVision = faceClipVision,
                    ClipVisionLoader = clipVisionLoader,
                    ArcFace = arcFace,
                    ArcFaceLoader = arcFaceLoader,
                    PosePipeline = posePipeline,
                    FaceDetector = faceDetector,
                    FaceIdLoraPath = loraPath,
                };
            }

            ClipVisionEncoder clipVision;
            (clipVision, clipVisionLoader) = await LoadClipVisionAsync(clipVisionModel, log, ct).ConfigureAwait(false);
            return new IpAdapterCacheEntry
            {
                FilePath = ipaPath,
                File = file,
                IpAdapter = adapter,
                ClipVision = clipVision,
                ClipVisionLoader = clipVisionLoader,
            };
        }
        catch (Exception ex)
        {
            Logs.Error($"[Features][IPA] Failed to load '{ipaPath}'.", ex);
            faceDetector?.Dispose();
            posePipeline?.Dispose();
            arcFaceLoader?.Dispose();
            clipVisionLoader?.Dispose();
            file.Dispose();
            throw;
        }
    }

    /// <summary>Loads a dedicated YOLOv8-Face detector when a checkpoint is installed; returns null (→ pose-keypoint
    /// fallback) when absent or on load failure. The variant + landmark stride are inferred from the filename.</summary>
    private static FaceDetector? TryLoadFaceDetector(IBackend backend, Action<string> log)
    {
        string? path = ResolveFaceDetectorWeights();
        if (path is null)
        {
            return null;
        }
        try
        {
            FaceDetector det = new FaceDetector(backend, InferFaceConfig(path), path);
            log($"  Face detector: {Path.GetFileName(path)} (YOLOv8-Face) — proper detect+align for the ArcFace crop.");
            return det;
        }
        catch (Exception ex)
        {
            Logs.Error($"[Features][IPA] Face-detector load failed ({ex.Message}); using the pose-keypoint face crop.", ex);
            return null;
        }
    }

    /// <summary>Locates a YOLOv8-Face safetensors (a filename containing both "face" and "yolo") under a conventional folder.</summary>
    private static string? ResolveFaceDetectorWeights()
    {
        string root = RepoPaths.ModelsRoot();
        foreach (string sub in _detectorFolders)
        {
            string dir = Path.Combine(root, sub);
            if (!Directory.Exists(dir))
            {
                continue;
            }
            try
            {
                foreach (string f in Directory.EnumerateFiles(dir, "*.safetensors").Order(StringComparer.Ordinal))
                {
                    string fn = Path.GetFileNameWithoutExtension(f).ToLowerInvariant();
                    if (fn.Contains("face", StringComparison.Ordinal) && fn.Contains("yolo", StringComparison.Ordinal))
                    {
                        return f;
                    }
                }
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Features][IPA] Face-detector scan of '{dir}' failed: {ex.GetType().Name}: {ex.Message}.");
            }
        }
        return null;
    }

    /// <summary>Infers the YOLOv8-Face variant (n/s/m/l) + landmark stride from the filename; stride defaults to 3
    /// (Ultralytics x/y/visibility), and a name hinting a landmark-only branch selects stride 2.</summary>
    private static YoloConfig InferFaceConfig(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        int kptDims = (n.Contains("kpt2", StringComparison.Ordinal) || n.Contains("lmk", StringComparison.Ordinal)
            || n.Contains("xy", StringComparison.Ordinal)) ? 2 : 3;
        if (n.Contains("yolov8l", StringComparison.Ordinal) || n.Contains("yolov8x", StringComparison.Ordinal))
        {
            return YoloV8FaceConfig.YoloV8lFace(kptDims);
        }
        if (n.Contains("yolov8m", StringComparison.Ordinal))
        {
            return YoloV8FaceConfig.YoloV8mFace(kptDims);
        }
        if (n.Contains("yolov8s", StringComparison.Ordinal))
        {
            return YoloV8FaceConfig.YoloV8sFace(kptDims);
        }
        return YoloV8FaceConfig.YoloV8nFace(kptDims);
    }

    /// <summary>Resolves and loads the CLIP-Vision encoder: an explicit override wins, otherwise the canonical
    /// CLIP-ViT-H/14 is auto-downloaded (every supported IPA — including FaceID-Plus — was trained against it).</summary>
    private static async Task<(ClipVisionEncoder Encoder, SafeTensorsLoader Loader)> LoadClipVisionAsync(
        string? clipVisionModel, Action<string> log, CancellationToken ct)
    {
        string path = ModelFileLocator.Find(clipVisionModel, "clip_vision", "text_encoders")
            ?? await ModelDownloader.EnsureSideModelAsync(SideModels.ClipVisionH14, null, ct).ConfigureAwait(false);
        log($"  CLIP-Vision: {Path.GetFileName(path)}");
        SafeTensorsLoader clipVisionLoader = new SafeTensorsLoader();
        try
        {
            clipVisionLoader.Load(path);
            Dictionary<string, Tensor> cvWeights = clipVisionLoader.GetAllTensors();
            // Some image-encoder files ship under a "vision_model." prefix, others ship rooted — probe for the patch
            // embedding under either naming.
            string cvPrefix = cvWeights.ContainsKey("vision_model.embeddings.patch_embedding.weight")
                ? "vision_model"
                : (cvWeights.ContainsKey("embeddings.patch_embedding.weight") ? "" : "vision_model");
            ClipVisionEncoder clipVision = new ClipVisionEncoder(ClipVisionEncoderConfig.ViTH14);
            clipVision.LoadWeights(cvWeights, prefix: cvPrefix);
            return (clipVision, clipVisionLoader);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Features][IPA] CLIP-Vision load failed for '{path}'.", ex);
            clipVisionLoader.Dispose();
            throw;
        }
    }

    /// <summary>Finds the FaceID companion UNet LoRA: a sibling <c>&lt;name&gt;_lora.safetensors</c> next to the
    /// checkpoint, else the known h94 companion (auto-downloaded). Returns null when unavailable.</summary>
    private static async Task<string?> ResolveFaceIdLoraAsync(string ipaPath, CancellationToken ct)
    {
        string sibling = Path.Combine(
            Path.GetDirectoryName(ipaPath) ?? "",
            Path.GetFileNameWithoutExtension(ipaPath) + "_lora.safetensors");
        if (File.Exists(sibling))
        {
            return sibling;
        }
        if (TryGetKnownFaceIdDownload(Path.GetFileName(ipaPath), out FaceIdDownload? dl))
        {
            try
            {
                return await ModelDownloader.EnsureSideModelAsync(dl!.LoraAsset, null, ct).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                Logs.Error($"[Features][IPA] FaceID companion LoRA download failed: {ex.Message}", ex);
                return null;
            }
        }
        return null;
    }
}
