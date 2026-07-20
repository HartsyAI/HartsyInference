using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Vision;

namespace HartsyInference.Engine.Services;

/// <summary>Vision service: routes an embed/detect/segment request to the pure-C# detectors lifted from the SwarmUI
/// extension (RT-DETR, YOLO, Grounding DINO, CLIPSeg, with SAM 2 box refinement) and to the CLIP-Vision image tower
/// for embeddings. Loaded detectors are cached per checkpoint for the life of the service.</summary>
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
    private readonly ClipVisionEmbedder _embedder = new();

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
                    _ => throw new NotSupportedException($"Unknown vision mode '{request.Mode}'."),
                };
            },
            cancel);
    }

    /// <summary>Embed: the pooled CLIP-Vision (ViT-H/14) projection of the image.</summary>
    private VisionResult Embed(ModelSpec spec, VisionRequest request)
    {
        string path = ClipVisionEmbedder.ResolvePath(spec.LocalPath)
            ?? throw new InvalidOperationException(
                "CLIP-Vision model not found. Pass the image encoder as the model path, or place a CLIP-ViT-H/14 "
                + $"safetensors in a 'clip_vision' folder under '{RepoPaths.ModelsRoot()}'.");
        return new VisionResult { Embedding = _embedder.Embed(Backend, path, request.Image) };
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
        IReadOnlyList<Detection> detections = Select(RunDetector(spec, request, target), target);
        string? sam2 = ResolveSam2(spec);
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
        _embedder.Dispose();
    }
}
