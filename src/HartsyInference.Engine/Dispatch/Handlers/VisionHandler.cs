using System.Globalization;
using System.Text;
using HartsyInference.Engine;
using HartsyInference.Core.Backends;
using HartsyInference.Vision.Clip;
using HartsyInference.Vision.Codec;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Embeddings;

namespace HartsyInference.Engine.Dispatch.Handlers;

/// <summary>Vision inference. The model id selects the task: <c>clip</c> → image embedding, <c>yolo8</c>/<c>yolo11</c> →
/// object detection. Weights are local <c>.safetensors</c> (Vision has no downloader). The "prompt" is an image path.</summary>
public sealed class VisionHandler : IModalityHandler
{
    /// <inheritdoc/>
    public Modality Modality => Modality.Vision;

    /// <inheritdoc/>
    public IModalityRunner Load(ModelSpec spec, IBackend backend, IProgressSink progress)
    {
        if (spec.LocalPath is null)
            throw new FileNotFoundException("No vision weights found. Pass a .safetensors checkpoint via --model-path.");

        string id = (spec.Catalog?.Id ?? spec.Requested).ToLowerInvariant();
        progress.Stage($"Loading {id} from {Path.GetFileName(spec.LocalPath)} …");

        if (id.StartsWith("clip", StringComparison.Ordinal))
        {
            ClipModelLoader loader = new ClipModelLoader(ClipPreset.OpenAiClipLarge);
            loader.LoadFromSingleFile(spec.LocalPath);
            ImageEmbeddingPipeline pipeline = new ImageEmbeddingPipeline(backend, new ClipImagePreprocessor(imageSize: 224), loader.ImageEncoder, id);
            return new VisionRunner(id, pipeline, backend, loader as IDisposable);
        }

        if (id.StartsWith("yolo", StringComparison.Ordinal))
        {
            bool v11 = id.Contains("11", StringComparison.Ordinal);
            YoloConfig config = v11 ? YoloConfig.YoloV11n : YoloConfig.YoloV8n;
            YoloPipeline detect = v11
                ? YoloPipeline.LoadV11(backend, config, spec.LocalPath)
                : new YoloPipeline(backend, config, spec.LocalPath);
            return new VisionRunner(id, detect, backend);
        }

        throw new NotSupportedException(
            $"Vision model '{spec.Requested}' is not recognized. Supported: clip (embed), yolo8/yolo11 (detect). " +
            "SAM segmentation and face detection are not yet implemented in the engine.");
    }

    /// <inheritdoc/>
    public GeneratedArtifact Run(IModalityRunner runner, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        VisionRunner vision = (VisionRunner)runner;

        string imagePath = prompt.Trim();
        if (!File.Exists(imagePath))
            throw new FileNotFoundException($"Input image not found: {imagePath}");

        (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(imagePath);
        return vision.Task == VisionTask.Embed
            ? Embed(vision, rgb, width, height, progress)
            : Detect(vision, rgb, width, height, parameters, progress);
    }

    private static GeneratedArtifact Embed(VisionRunner vision, byte[] rgb, int width, int height, IProgressSink progress)
    {
        progress.Stage("Embedding …");
        using ImageEmbedding embedding = vision.Embed!.Embed(rgb, width, height);
        ReadOnlySpan<float> vector = embedding.AsSpan();

        StringBuilder preview = new StringBuilder();
        int shown = Math.Min(8, vector.Length);
        for (int i = 0; i < shown; i++)
            preview.Append(i == 0 ? "" : ", ").Append(vector[i].ToString("F4", CultureInfo.InvariantCulture));

        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Data,
            Extension = "txt",
            Text = $"{embedding.EmbeddingDim}-dim embedding: [{preview}, …]",
        };
        artifact.Meta["model"] = vision.ModelId;
        artifact.Meta["dim"] = embedding.EmbeddingDim.ToString(CultureInfo.InvariantCulture);
        return artifact;
    }

    private static GeneratedArtifact Detect(VisionRunner vision, byte[] rgb, int width, int height, ParamState parameters, IProgressSink progress)
    {
        progress.Stage("Detecting …");
        float confidence = parameters.GetFloat("confidence", 0.25f);
        IReadOnlyList<YoloDetection> detections = vision.Detect!.Detect(rgb, width, height, confidenceThreshold: confidence);

        StringBuilder lines = new StringBuilder();
        foreach (YoloDetection d in detections)
        {
            lines.Append(vision.Detect!.GetLabel(d.ClassId))
                .Append(' ').Append(d.Confidence.ToString("F2", CultureInfo.InvariantCulture))
                .Append("  (").Append((int)d.X1).Append(',').Append((int)d.Y1)
                .Append(")-(").Append((int)d.X2).Append(',').Append((int)d.Y2).Append(")\n");
        }

        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Data,
            Extension = "txt",
            Text = detections.Count == 0 ? "(no detections)" : lines.ToString().TrimEnd('\n'),
        };
        artifact.Meta["model"] = vision.ModelId;
        artifact.Meta["detections"] = detections.Count.ToString(CultureInfo.InvariantCulture);
        return artifact;
    }
}
