using System.Linq;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Vision;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Annotators;
using HartsyInference.Vision.Clip;
using HartsyInference.Vision.Codec;
using HartsyInference.Vision.DepthAnything;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Embeddings;
using HartsyInference.Vision.Rmbg;
using HartsyInference.Vision.Siglip;

namespace HartsyInference.Engine.Services;

/// <summary>Vision service: routes an embed/detect/segment request to the pure-C# detectors lifted from the SwarmUI
/// extension (RT-DETR, YOLO, Grounding DINO, CLIPSeg, with SAM 2 box refinement) and to the standalone CLIP /
/// SigLIP / DINOv2 image towers for embeddings, selected by the requested catalog id. Loaded models are cached per
/// checkpoint path for the life of the service.</summary>
public sealed class VisionService : IVisionService, IDisposable
{
    /// <summary>Aux key a caller may set on <see cref="ModelSpec"/> to point at a specific SAM 2 checkpoint.</summary>
    public const string Sam2AuxKey = "sam2-path";

    private readonly InferenceEngine _engine;
    private readonly RtDetrObjectDetector _rtDetr = new();
    private readonly YoloObjectDetector _yolo = new();
    private readonly GroundingDinoObjectDetector _dino = new();
    private readonly ClipSegSegmenter _clipSeg = new();
    private readonly Sam2MaskRefiner _sam2 = new();

    private readonly Dictionary<string, ClipModelLoader> _clipCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, SiglipVisionEncoder> _siglipCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Dinov2VisionEncoder> _dinov2Cache = new(StringComparer.Ordinal);
    // SigLIP/DINOv2 weights are mmap-backed non-owning views (F32(t) returns the source tensor unchanged when
    // it's already F32) — the loader must stay open for as long as the cached encoder is used, same lifecycle
    // ClipModelLoader keeps internally for its own mmap.
    private readonly List<SafeTensorsLoader> _embedLoaders = new();
    private readonly object _embedLock = new();

    private readonly Dictionary<string, DepthAnythingV2Model> _depthCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, HedModel> _hedCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, LineartGenerator> _lineartCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, NormalBaeModel> _normalBaeCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, UperNetSegModel> _upernetCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, BriaRmbg> _rmbgCache = new(StringComparer.Ordinal);
    // All five load via PytorchPickleLoader (raw .pth/.pt), whose Dispose() also disposes its tensors — the
    // loader must stay open for as long as the cached model is used, same as _embedLoaders above.
    private readonly List<PytorchPickleLoader> _annotatorLoaders = new();

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal VisionService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<VisionResult> RunAsync(ModelSpec spec, VisionRequest request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(request);
        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                return request.Mode switch
                {
                    VisionMode.Embed => Embed(spec, request),
                    VisionMode.Detect => Detect(spec, request, cancel),
                    VisionMode.Segment => Segment(spec, request, cancel),
                    VisionMode.Depth => Depth(spec, request),
                    VisionMode.Edge => Edge(spec, request),
                    VisionMode.Lineart => LineartMode(spec, request),
                    VisionMode.Normal => Normal(spec, request),
                    VisionMode.SegMap => SegMap(spec, request),
                    VisionMode.BackgroundRemoval => BackgroundRemoval(spec, request),
                    _ => throw new NotSupportedException($"Unknown vision mode '{request.Mode}'."),
                };
            },
            cancel);
    }

    /// <summary>Embed: routes to the standalone CLIP / SigLIP / DINOv2 tower matching the requested catalog id
    /// (default CLIP ViT-L/14, matching <c>openai/clip-vit-large-patch14</c> — the family with a real-weight
    /// parity test).</summary>
    private VisionResult Embed(ModelSpec spec, VisionRequest request)
    {
        string id = (spec.Catalog?.Id ?? spec.Requested ?? "").ToLowerInvariant();
        string path = spec.LocalPath
            ?? throw new InvalidOperationException(
                $"'{id}' checkpoint not found. Pass --model-path, or select a catalog id with auto-download assets.");
        return id switch
        {
            "siglip" => new VisionResult { Embedding = EmbedSiglip(path, request) },
            "dinov2" => new VisionResult { Embedding = EmbedDinov2(path, request) },
            _ => new VisionResult { Embedding = EmbedClip(path, request) },
        };
    }

    /// <summary>CLIP ViT-L/14 embed via the standalone <see cref="ClipModelLoader"/> (not the diffusion-package's
    /// hardcoded ViT-H/14 IP-Adapter tower — this is the family with a real-weight parity test).</summary>
    private float[] EmbedClip(string path, VisionRequest request)
    {
        ClipModelLoader loader = GetOrLoad(_clipCache, path, () =>
        {
            ClipModelLoader l = new ClipModelLoader(ClipPreset.OpenAiClipLarge);
            l.LoadFromSingleFile(path);
            return l;
        });
        ClipImagePreprocessor preprocessor = new ClipImagePreprocessor(loader.ImageEncoder.Config.ImageSize);
        Tensor pixels = preprocessor.Preprocess(request.Image.Rgb, request.Image.Width, request.Image.Height);
        try
        {
            using ImageEmbedding embedding = loader.ImageEncoder.Encode(Backend, pixels);
            return embedding.AsSpan().ToArray();
        }
        finally
        {
            pixels.Dispose();
        }
    }

    /// <summary>SigLIP embed (attention-pooled, then L2-normalized here — <see cref="SiglipVisionEncoder.Encode"/>
    /// returns the raw projection).</summary>
    private float[] EmbedSiglip(string path, VisionRequest request)
    {
        SiglipVisionEncoder encoder = GetOrLoad(_siglipCache, path, () =>
        {
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(path);
            _embedLoaders.Add(loader);
            SiglipVisionEncoder e = new SiglipVisionEncoder(SiglipPreset.Base16_224);
            e.LoadWeights(loader.GetAllTensors());
            return e;
        });
        SiglipImagePreprocessor preprocessor = new SiglipImagePreprocessor(encoder.Preset.ImageSize);
        Tensor pixels = preprocessor.Preprocess(request.Image.Rgb, request.Image.Width, request.Image.Height);
        try
        {
            Tensor raw = encoder.Encode(Backend, pixels);
            try
            {
                return L2Normalize(raw);
            }
            finally
            {
                raw.Dispose();
            }
        }
        finally
        {
            pixels.Dispose();
        }
    }

    /// <summary>DINOv2 embed: the CLS token (index 0 of the sequence) — DINOv2 has no contrastive projection head,
    /// so the CLS token is the natural pooled representation for downstream cosine comparison.</summary>
    private float[] EmbedDinov2(string path, VisionRequest request)
    {
        Dinov2VisionEncoder encoder = GetOrLoad(_dinov2Cache, path, () =>
        {
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(path);
            _embedLoaders.Add(loader);
            Dinov2VisionEncoder e = new Dinov2VisionEncoder(Dinov2Preset.Small);
            e.LoadWeights(loader.GetAllTensors());
            return e;
        });
        Dinov2ImagePreprocessor preprocessor = new Dinov2ImagePreprocessor(encoder.Preset.ImageSize);
        Tensor pixels = preprocessor.Preprocess(request.Image.Rgb, request.Image.Width, request.Image.Height);
        try
        {
            Tensor hidden = encoder.Encode(Backend, pixels);
            try
            {
                return L2Normalize(SliceClsToken(hidden, encoder.HiddenSize).AsSpan());
            }
            finally
            {
                hidden.Dispose();
            }
        }
        finally
        {
            pixels.Dispose();
        }
    }

    /// <summary>Depth-Anything-V2: relative-depth grayscale map, min-max normalized to [0,1].</summary>
    private VisionResult Depth(ModelSpec spec, VisionRequest request)
    {
        string path = RequirePath(spec, "depth-anything");
        DepthAnythingV2Model model = GetOrLoad(_depthCache, path, () =>
        {
            DepthAnythingV2Model m = new DepthAnythingV2Model(DepthAnythingPreset.Small);
            m.LoadWeights(LoadPickle(path));
            return m;
        });
        DepthAnythingPreprocessor preprocessor = new DepthAnythingPreprocessor();
        ImageData image = request.Image;
        Tensor pixels = preprocessor.Preprocess(image.Rgb, image.Width, image.Height);
        try
        {
            Tensor depth = model.Forward(Backend, pixels);
            try
            {
                float[] unit = DepthAnythingPreprocessor.PostprocessToUnit(depth, image.Width, image.Height);
                return new VisionResult { Image = GrayscaleResult(unit, image.Width, image.Height) };
            }
            finally
            {
                depth.Dispose();
            }
        }
        finally
        {
            pixels.Dispose();
        }
    }

    /// <summary>HED: ControlNet-style soft-edge map (black background, white edges). No resize constraint.</summary>
    private VisionResult Edge(ModelSpec spec, VisionRequest request)
    {
        string path = RequirePath(spec, "hed");
        HedModel model = GetOrLoad(_hedCache, path, () =>
        {
            HedModel m = new HedModel();
            m.LoadWeights(LoadPickle(path));
            return m;
        });
        ImageData image = request.Image;
        HedPreprocessor preprocessor = new HedPreprocessor(model);
        float[] unit = preprocessor.Process(Backend, image.Rgb, image.Width, image.Height);
        return new VisionResult { Image = GrayscaleResult(unit, image.Width, image.Height) };
    }

    /// <summary>Lineart: ControlNet-style line map (white lines on black), realistic by default — pass
    /// <c>-p coarse</c> for the sk_model2 variant.</summary>
    private VisionResult LineartMode(ModelSpec spec, VisionRequest request)
    {
        string path = RequirePath(spec, "lineart");
        bool coarse = string.Equals(request.Prompt?.Trim(), "coarse", StringComparison.OrdinalIgnoreCase);
        string cacheKey = path + (coarse ? "|coarse" : "|realistic");
        LineartGenerator model = GetOrLoad(_lineartCache, cacheKey, () =>
        {
            LineartGenerator m = new LineartGenerator(coarse ? LineartPreset.Coarse : LineartPreset.Realistic);
            m.LoadWeights(LoadPickle(coarse ? SiblingPath(path, "sk_model2.pth") : path));
            return m;
        });
        ImageData image = request.Image;
        (int w, int h) = RoundDownToMultiple(image.Width, image.Height, 4);
        byte[] resized = ResizeRgb(image, w, h);
        Tensor pixels = LineartPreprocessor.Preprocess(resized, w, h);
        try
        {
            Tensor line = model.Forward(Backend, pixels);
            try
            {
                float[] unit = LineartPreprocessor.PostprocessToUnit(line);
                return new VisionResult { Image = GrayscaleResult(unit, w, h) };
            }
            finally
            {
                line.Dispose();
            }
        }
        finally
        {
            pixels.Dispose();
        }
    }

    /// <summary>NormalBAE: ControlNet-style surface-normal RGB map.</summary>
    private VisionResult Normal(ModelSpec spec, VisionRequest request)
    {
        string path = RequirePath(spec, "normalbae");
        NormalBaeModel model = GetOrLoad(_normalBaeCache, path, () =>
        {
            NormalBaeModel m = new NormalBaeModel(NormalBaePreset.Default);
            m.LoadWeights(LoadPickle(path));
            return m;
        });
        ImageData image = request.Image;
        (int w, int h) = RoundDownToMultiple(image.Width, image.Height, 32);
        byte[] resized = ResizeRgb(image, w, h);
        Tensor pixels = NormalBaePreprocessor.Preprocess(resized, w, h);
        try
        {
            bool probe = Environment.GetEnvironmentVariable("HARTSY_VISION_PROBE") == "1";
            Tensor normals = model.Forward(Backend, pixels, probe ? ProbeStats : null);
            try
            {
                byte[] rgb = NormalBaePreprocessor.PostprocessToRgb24(normals);
                return new VisionResult { Image = new ImageData { Rgb = rgb, Width = w, Height = h } };
            }
            finally
            {
                normals.Dispose();
            }
        }
        finally
        {
            pixels.Dispose();
        }
    }

    /// <summary>RMBG-1.4: foreground cutout composited onto neutral gray-0.5, matching what the image→3D
    /// pipelines (TripoSR / Hunyuan3D) expect from their background-removal step.</summary>
    private VisionResult BackgroundRemoval(ModelSpec spec, VisionRequest request)
    {
        string path = RequirePath(spec, "rmbg");
        BriaRmbg model = GetOrLoad(_rmbgCache, path, () =>
        {
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(path);
            _embedLoaders.Add(loader);
            BriaRmbg m = new BriaRmbg();
            m.LoadWeights(loader.GetAllTensors());
            return m;
        });
        RmbgBackgroundRemover remover = new RmbgBackgroundRemover(model);
        ImageData image = request.Image;
        byte[] cutout = remover.CompositeOnGray(Backend, image.Rgb, image.Width, image.Height);
        return new VisionResult { Image = new ImageData { Rgb = cutout, Width = image.Width, Height = image.Height } };
    }

    /// <summary>UperNet-Seg: ADE20K semantic-segmentation palette map. The reference pipeline stretch-resizes
    /// to a fixed 512×512 detect resolution.</summary>
    private VisionResult SegMap(ModelSpec spec, VisionRequest request)
    {
        string path = RequirePath(spec, "upernet-seg");
        UperNetSegModel model = GetOrLoad(_upernetCache, path, () =>
        {
            UperNetSegModel m = new UperNetSegModel();
            m.LoadWeights(LoadPickle(path));
            return m;
        });
        ImageData image = request.Image;
        const int size = UperNetSegPreprocessor.ReferenceSize;
        byte[] resized = ResizeRgb(image, size, size);
        Tensor pixels = UperNetSegPreprocessor.Preprocess(resized, size, size);
        try
        {
            bool probe = Environment.GetEnvironmentVariable("HARTSY_VISION_PROBE") == "1";
            Tensor logits = model.Forward(Backend, pixels, probe ? ProbeStats : null);
            try
            {
                byte[] classMap = UperNetSegPreprocessor.Argmax(logits);
                byte[] palette = Ade20kPalette.Colorize(classMap);
                return new VisionResult { Image = new ImageData { Rgb = palette, Width = size, Height = size } };
            }
            finally
            {
                logits.Dispose();
            }
        }
        finally
        {
            pixels.Dispose();
        }
    }

    /// <summary>Resolves the checkpoint path for a single-asset catalog model, or throws with the model id in
    /// the message (mirrors the Embed-mode error, generalized to any single-primary-asset vision mode).</summary>
    private static string RequirePath(ModelSpec spec, string label) =>
        spec.LocalPath
        ?? throw new InvalidOperationException(
            $"'{label}' checkpoint not found. Pass --model-path, or select -m {label} to auto-download.");

    /// <summary>Loads a raw <c>.pt</c>/<c>.pth</c> checkpoint and tracks its loader for the service's lifetime
    /// (its tensors are only valid while the loader is alive — see <see cref="_annotatorLoaders"/>).</summary>
    private Dictionary<string, Tensor> LoadPickle(string path)
    {
        lock (_embedLock)
        {
            PytorchPickleLoader loader = new PytorchPickleLoader();
            loader.Load(path);
            _annotatorLoaders.Add(loader);
            return loader.GetAllTensors();
        }
    }

    /// <summary>A sibling file in the same directory as <paramref name="path"/> — used to switch the Lineart
    /// checkpoint between its two co-located variants without re-resolving the catalog spec.</summary>
    private static string SiblingPath(string path, string fileName) =>
        Path.Combine(Path.GetDirectoryName(path) ?? ".", fileName);

    /// <summary>Diagnostic: when <c>HARTSY_VISION_PROBE=1</c>, logs min/max/mean/NaN/Inf for a named
    /// intermediate tensor. Used to bisect where a CUDA-backend forward pass first diverges.</summary>
    private static unsafe void ProbeStats(string label, Tensor t)
    {
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float* p = (float*)f32.DataPointer;
        long n = f32.ElementCount;
        float mn = float.MaxValue, mx = float.MinValue; double sum = 0; long nanCount = 0, infCount = 0;
        for (long e = 0; e < n; e++)
        {
            float v = p[e];
            if (float.IsNaN(v)) { nanCount++; continue; }
            if (float.IsInfinity(v)) { infCount++; continue; }
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        Logs.Warning(
            $"[vision-probe] {label}: shape=[{string.Join(",", Enumerable.Range(0, t.Shape.Rank).Select(i => t.Shape[i]))}] min={mn:F4} max={mx:F4} mean={sum / n:F4} nan={nanCount} inf={infCount}");
        if (!ReferenceEquals(f32, t)) f32.Dispose();
    }

    /// <summary>Rounds an image's dimensions down to the nearest multiple, with a floor of one multiple.</summary>
    private static (int Width, int Height) RoundDownToMultiple(int width, int height, int multiple) =>
        (Math.Max(multiple, width / multiple * multiple), Math.Max(multiple, height / multiple * multiple));

    /// <summary>Bicubic-resizes an RGB24 image to new dimensions (no-op copy when the size already matches).</summary>
    private static byte[] ResizeRgb(ImageData image, int width, int height)
    {
        if (image.Width == width && image.Height == height)
        {
            return image.Rgb;
        }
        float[] resizedF = new float[(long)width * height * 3];
        Resample.BicubicHwc8(image.Rgb, image.Width, image.Height, 3, resizedF, width, height, -0.5f, antialias: true);
        byte[] result = new byte[resizedF.Length];
        for (int i = 0; i < resizedF.Length; i++)
        {
            result[i] = (byte)Math.Clamp((int)MathF.Round(resizedF[i]), 0, 255);
        }
        return result;
    }

    /// <summary>Wraps a [0,1] grayscale buffer as a replicated-RGB <see cref="ImageData"/>.</summary>
    private static ImageData GrayscaleResult(float[] unit, int width, int height) =>
        new ImageData { Rgb = ImageTensor.UnitGrayscaleToRgb24(unit), Width = width, Height = height };

    /// <summary>Copies the CLS token (sequence position 0) out of a <c>[1, seqLen, hidden]</c> tensor.</summary>
    private static unsafe float[] SliceClsToken(Tensor hidden, int hiddenSize)
    {
        Tensor f32 = hidden.DType == DType.F32 ? hidden : hidden.CastTo(DType.F32);
        try
        {
            float[] result = new float[hiddenSize];
            float* p = (float*)f32.DataPointer;
            new ReadOnlySpan<float>(p, hiddenSize).CopyTo(result);
            return result;
        }
        finally
        {
            if (!ReferenceEquals(f32, hidden)) f32.Dispose();
        }
    }

    /// <summary>L2-normalizes a <c>[1, dim]</c> (or flat <c>[dim]</c>) tensor into a unit-vector array.</summary>
    private static unsafe float[] L2Normalize(Tensor t)
    {
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        try
        {
            int n = (int)f32.ElementCount;
            float* p = (float*)f32.DataPointer;
            return L2Normalize(new ReadOnlySpan<float>(p, n));
        }
        finally
        {
            if (!ReferenceEquals(f32, t)) f32.Dispose();
        }
    }

    /// <summary>L2-normalizes a flat vector into a unit-vector array.</summary>
    private static float[] L2Normalize(ReadOnlySpan<float> v)
    {
        double sumSq = 0;
        for (int i = 0; i < v.Length; i++) sumSq += (double)v[i] * v[i];
        float invNorm = sumSq < 1e-24 ? 1f : (float)(1.0 / Math.Sqrt(sumSq));
        float[] result = new float[v.Length];
        for (int i = 0; i < v.Length; i++) result[i] = v[i] * invNorm;
        return result;
    }

    /// <summary>Cache-or-load under a lock, keyed by checkpoint path — models stay resident for the service's life.</summary>
    private T GetOrLoad<T>(Dictionary<string, T> cache, string path, Func<T> load) where T : class
    {
        lock (_embedLock)
        {
            if (cache.TryGetValue(path, out T? cached))
            {
                return cached;
            }
            T created = load();
            cache[path] = created;
            return created;
        }
    }

    /// <summary>Detect: RT-DETR / YOLO for the closed-vocabulary targets, Grounding DINO for a free-text prompt.</summary>
    private VisionResult Detect(ModelSpec spec, VisionRequest request, CancellationToken cancel)
    {
        VisionTarget target = VisionTargetRouter.Parse(request.Prompt, VisionMode.Detect);
        IReadOnlyList<Detection> detections = RunDetector(spec, request, target);
        cancel.ThrowIfCancellationRequested();
        return new VisionResult { Detections = Select(detections, target) };
    }

    /// <summary>Segment: CLIPSeg for a free-text prompt, otherwise a detector's boxes refined by SAM 2 (falling back
    /// to box rasterization). Masks come back at source resolution as grayscale replicated into RGB.</summary>
    private VisionResult Segment(ModelSpec spec, VisionRequest request, CancellationToken cancel)
    {
        VisionTarget target = VisionTargetRouter.Parse(request.Prompt, VisionMode.Segment);
        ImageData image = request.Image;
        if (target.Kind == VisionTargetKind.ClipSeg)
        {
            string dir = VisionModelPaths.FindClipSegDirectory(spec.LocalPath)
                ?? throw new InvalidOperationException(
                    "CLIPSeg model not found. Place the 'clipseg-rd64-refined' folder (with model.safetensors) inside "
                    + $"a '{VisionModelPaths.ClipSegFolder}' folder under '{RepoPaths.ModelsRoot()}'.");
            byte[]? mask = _clipSeg.Segment(Backend, dir, image, target.Query, MaskThreshold(request));
            return new VisionResult
            {
                Masks = mask is null ? [] : [VisionMasks.ToImageData(mask, image.Width, image.Height)],
            };
        }
        // "-m sam" selects SAM 2 itself as the model: spec.LocalPath is the SAM 2 refiner checkpoint, not a
        // detector — the box source falls back to whatever detector is conventionally staged (RT-DETR by
        // default, same as an empty prompt with no model selected at all).
        bool isSamPrimary = string.Equals(spec.Catalog?.Id, "sam", StringComparison.OrdinalIgnoreCase);
        ModelSpec detectorSpec = isSamPrimary ? spec with { LocalPath = null } : spec;
        IReadOnlyList<Detection> detections = Select(RunDetector(detectorSpec, request, target), target);
        string? sam2 = isSamPrimary ? spec.LocalPath : ResolveSam2(spec);
        if (sam2 is null)
        {
            Logs.Info("[Vision] SAM 2 checkpoint not installed — returning bounding-box masks. Place a converted "
                + $"sam2_hiera_*.safetensors in a '{VisionModelPaths.Sam2Folder}' folder for pixel-accurate masks.");
        }
        List<ImageData> masks = new List<ImageData>(detections.Count);
        foreach (Detection d in detections)
        {
            cancel.ThrowIfCancellationRequested();
            byte[] mask = _sam2.TryRefine(Backend, sam2, image, d.X, d.Y, d.X + d.Width, d.Y + d.Height)
                ?? VisionMasks.RasterizeBox(d.X, d.Y, d.X + d.Width, d.Y + d.Height, image.Width, image.Height);
            masks.Add(VisionMasks.ToImageData(mask, image.Width, image.Height));
        }
        return new VisionResult { Masks = masks };
    }

    /// <summary>Runs the detector the parsed target routes to and returns its raw pixel-space boxes.</summary>
    private IReadOnlyList<Detection> RunDetector(ModelSpec spec, VisionRequest request, VisionTarget target)
    {
        float threshold = MaskThreshold(request);
        switch (target.Kind)
        {
            case VisionTargetKind.Yolo:
            {
                string path = VisionModelPaths.FindYolo(target.ModelName, spec.LocalPath)
                    ?? throw new InvalidOperationException(
                        $"YOLO model '{target.ModelName}' not found. Place a .safetensors YOLO model in a "
                        + $"'{VisionModelPaths.YoloFolder}' folder under '{RepoPaths.ModelsRoot()}' (the engine loads "
                        + "safetensors, not Ultralytics .pt files).");
                return _yolo.Detect(Backend, path, request.Image, threshold);
            }
            case VisionTargetKind.GroundingDino:
            {
                (string? checkpoint, string? vocab) = VisionModelPaths.FindGroundingDino(spec.LocalPath);
                if (checkpoint is null || vocab is null)
                {
                    throw new InvalidOperationException(
                        "Grounding DINO model not found. Place 'model.safetensors' and 'vocab.txt' (from "
                        + $"IDEA-Research/grounding-dino-tiny) in a '{VisionModelPaths.GroundingDinoFolder}' folder "
                        + $"under '{RepoPaths.ModelsRoot()}'.");
                }
                return _dino.Detect(Backend, checkpoint, vocab, request.Image, target.Query, threshold);
            }
            case VisionTargetKind.RtDetr:
            {
                string path = VisionModelPaths.FindCheckpoint(spec.LocalPath, VisionModelPaths.RtDetrFolder)
                    ?? throw new InvalidOperationException(
                        "RT-DETR model not found. Place a converted rtdetr_r18vd .safetensors in a "
                        + $"'{VisionModelPaths.RtDetrFolder}' folder under '{RepoPaths.ModelsRoot()}'.");
                return _rtDetr.Detect(Backend, path, request.Image, threshold);
            }
            default:
                throw new NotSupportedException($"Vision target '{target.Kind}' produces no bounding boxes.");
        }
    }

    /// <summary>Clamps the request threshold into the open interval detectors expect; out-of-range falls back to 0.25.</summary>
    private static float MaskThreshold(VisionRequest request)
    {
        float threshold = (float)request.Threshold;
        return threshold > 0f && threshold < 1f ? threshold : 0.25f;
    }

    /// <summary>Applies the target's class filter, sorts left-to-right, and narrows to a single detection when the
    /// target carried an explicit index.</summary>
    private static IReadOnlyList<Detection> Select(IReadOnlyList<Detection> detections, VisionTarget target)
    {
        IEnumerable<Detection> query = detections;
        if (target.ClassFilter.Length > 0)
        {
            query = query.Where(d => d.Label.Contains(target.ClassFilter, StringComparison.OrdinalIgnoreCase));
        }
        List<Detection> sorted = query.OrderBy(d => d.X).ToList();
        if (target.Index < 0 || sorted.Count == 0)
        {
            return sorted;
        }
        int index = target.Index < sorted.Count ? target.Index : 0;
        return [sorted[index]];
    }

    /// <summary>SAM 2 checkpoint: an explicit <see cref="Sam2AuxKey"/> aux path wins, else the conventional folder.</summary>
    private static string? ResolveSam2(ModelSpec spec)
    {
        spec.Aux.TryGetValue(Sam2AuxKey, out string? aux);
        return VisionModelPaths.FindSam2(aux);
    }

    /// <summary>The engine's compute backend, created on first use.</summary>
    private IBackend Backend => _engine.Backend;

    /// <inheritdoc/>
    public void Dispose()
    {
        _rtDetr.Dispose();
        _yolo.Dispose();
        _dino.Dispose();
        _clipSeg.Dispose();
        _sam2.Dispose();
        foreach (ClipModelLoader loader in _clipCache.Values) loader.Dispose();
        foreach (SafeTensorsLoader loader in _embedLoaders) loader.Dispose();
        foreach (PytorchPickleLoader loader in _annotatorLoaders) loader.Dispose();
    }
}
