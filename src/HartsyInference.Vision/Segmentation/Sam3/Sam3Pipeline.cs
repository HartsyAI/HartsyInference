using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>SAM 3 text-grounded image segmentation: a shared image encoder runs once per image, then each text prompt
/// runs the text tower, detector transformer and segmentation head, and the scored queries are reported as instances.
/// Video and geometric (box or point) prompts are not implemented.</summary>
public sealed unsafe class Sam3Pipeline : ITextGroundedSegmenter
{
    private readonly Sam3Config _config;
    private readonly Sam3Tokenizer _tokenizer;
    private readonly Sam3TextEncoder _textEncoder;
    private readonly Sam3VisionBackbone _vision;
    private readonly Sam3Transformer _transformer;
    private readonly Sam3SegmentationHead _head;
    private readonly AnyFormatCheckpointLoader _loader;
    private bool _disposed;

    private Sam3Pipeline(Sam3Config config, Sam3Tokenizer tokenizer, Sam3TextEncoder textEncoder, Sam3VisionBackbone vision,
        Sam3Transformer transformer, Sam3SegmentationHead head, AnyFormatCheckpointLoader loader)
    {
        _config = config;
        _tokenizer = tokenizer;
        _textEncoder = textEncoder;
        _vision = vision;
        _transformer = transformer;
        _head = head;
        _loader = loader;
    }

    /// <summary>Loads a SAM 3 checkpoint (<c>sam3.pt</c> through the safe-subset pickle reader, or a safetensors export)
    /// and builds a ready pipeline. The pipeline owns the loader and releases it on <see cref="Dispose"/>.</summary>
    public static Sam3Pipeline Load(Sam3Config config, string checkpointPath)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);

        AnyFormatCheckpointLoader loader = new();
        Sam3Tokenizer? tokenizer = null;
        Sam3VisionBackbone? vision = null;
        try
        {
            loader.Load(checkpointPath);
            Sam3WeightGroups groups = Sam3Converter.Convert(loader.GetAllTensors(), config);

            tokenizer = new Sam3Tokenizer();
            Sam3TextEncoder textEncoder = new(config, groups.TextEncoder);
            vision = new Sam3VisionBackbone(config, groups.VisionTrunk, groups.VisionNeck);
            Sam3Transformer transformer = new(config, groups.FusionEncoder, groups.Decoder, groups.GeometryEncoder);
            Sam3SegmentationHead head = new(config, groups.SegmentationHead, groups.Scoring);
            return new Sam3Pipeline(config, tokenizer, textEncoder, vision, transformer, head, loader);
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to load SAM 3 pipeline from '{checkpointPath}'.", ex);
            vision?.Dispose();
            tokenizer?.Dispose();
            loader.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public string ModelName => "sam3";

    /// <inheritdoc/>
    public IReadOnlyList<TextGroundedPromptResult> Segment(IBackend backend, Tensor image, IReadOnlyList<string> prompts)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(prompts);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (prompts.Count == 0)
        {
            throw new ArgumentException("At least one prompt is required.", nameof(prompts));
        }

        Tensor normalized = Sam3Preprocessor.Prepare(backend, image, _config.ImageSize);
        int height = (int)image.Shape[2];
        int width = (int)image.Shape[3];
        Sam3VisionFeatures vision;
        try
        {
            vision = _vision.Forward(backend, normalized);
        }
        finally
        {
            normalized.Dispose();
        }

        try
        {
            TextGroundedPromptResult[] results = new TextGroundedPromptResult[prompts.Count];
            for (int p = 0; p < prompts.Count; p++)
            {
                results[p] = SegmentPrompt(backend, vision, prompts[p], width, height);
            }

            return results;
        }
        finally
        {
            foreach (Tensor level in vision.Features)
            {
                level.Dispose();
            }

            foreach (Tensor position in vision.Positions)
            {
                position.Dispose();
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _vision.Dispose();
        _tokenizer.Dispose();
        _loader.Dispose();
    }

    private TextGroundedPromptResult SegmentPrompt(IBackend backend, Sam3VisionFeatures vision, string prompt, int width, int height)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);
        int[] tokens = _tokenizer.Encode(prompt);
        Sam3TextFeatures text = _textEncoder.Forward(backend, tokens);
        try
        {
            Sam3DecoderOutput decoder = _transformer.Forward(backend, vision, text);
            try
            {
                Sam3InstancePredictions predictions = _head.Forward(backend, vision, decoder);
                try
                {
                    return BuildResult(backend, prompt, decoder, predictions, width, height);
                }
                finally
                {
                    predictions.MaskLogits.Dispose();
                    predictions.ClassLogits.Dispose();
                }
            }
            finally
            {
                decoder.Boxes.Dispose();
                decoder.QueryHidden.Dispose();
                decoder.PresenceLogit.Dispose();
                decoder.EncoderMemory.Dispose();
                decoder.PromptFeatures.Dispose();
            }
        }
        finally
        {
            text.Memory.Dispose();
        }
    }

    private TextGroundedPromptResult BuildResult(IBackend backend, string prompt, Sam3DecoderOutput decoder,
        Sam3InstancePredictions predictions, int width, int height)
    {
        int queries = _config.NumQueries;
        int mask = _config.MaskResolution;
        ValidateShape(decoder.Boxes, "boxes", 1, queries, 4);
        ValidateShape(decoder.PresenceLogit, "presence logit", 1, 1);
        ValidateShape(predictions.ClassLogits, "class logits", 1, queries);
        ValidateShape(predictions.MaskLogits, "mask logits", 1, queries, mask, mask);

        float[] classLogits = ReadF32(predictions.ClassLogits, queries);
        float presence = ReadF32(decoder.PresenceLogit, 1)[0];
        float[] scores = new float[queries];
        Sam3PostProcessor.ComputeScores(classLogits, presence, scores);
        int[] keep = Sam3PostProcessor.SelectAbove(scores, _config.ScoreThreshold);

        float[] boxesNorm = ReadF32(decoder.Boxes, queries * 4);
        float[] boxesPixels = new float[queries * 4];
        Sam3PostProcessor.CxcywhToXyxyPixels(boxesNorm, width, height, boxesPixels);

        float* maskBase = (float*)predictions.MaskLogits.DataPointer;
        long plane = (long)mask * mask;
        TextGroundedInstance[] instances = new TextGroundedInstance[keep.Length];
        for (int k = 0; k < keep.Length; k++)
        {
            int q = keep[k];
            byte[] pixels = ProjectMask(backend, maskBase + q * plane, mask, width, height);
            instances[k] = new TextGroundedInstance(
                scores[q], boxesPixels[q * 4], boxesPixels[q * 4 + 1], boxesPixels[q * 4 + 2], boxesPixels[q * 4 + 3], pixels);
        }

        return new TextGroundedPromptResult(prompt, width, height, instances);
    }

    private byte[] ProjectMask(IBackend backend, float* logits, int side, int width, int height)
    {
        Tensor low = new(new TensorShape(1, 1, side, side), DType.F32);
        Tensor up = new(new TensorShape(1, 1, height, width), DType.F32);
        try
        {
            new ReadOnlySpan<float>(logits, side * side).CopyTo(new Span<float>((float*)low.DataPointer, side * side));
            backend.InterpolateBilinear2D(up, low, alignCorners: false);
            byte[] mask = new byte[width * height];
            ReadOnlySpan<float> upSpan = new((float*)up.DataPointer, width * height);
            Sam3PostProcessor.ThresholdLogits(upSpan, _config.MaskLogitThreshold, mask);
            return mask;
        }
        finally
        {
            low.Dispose();
            up.Dispose();
        }
    }

    private static float[] ReadF32(Tensor tensor, int count)
    {
        float[] values = new float[count];
        new ReadOnlySpan<float>((float*)tensor.DataPointer, count).CopyTo(values);
        return values;
    }

    private static void ValidateShape(Tensor tensor, string name, params long[] expected)
    {
        if (tensor.DType != DType.F32 || tensor.Shape.Rank != expected.Length)
        {
            throw new InvalidOperationException($"SAM 3 {name} has shape {tensor.Shape}; expected rank {expected.Length} F32.");
        }

        for (int i = 0; i < expected.Length; i++)
        {
            if (tensor.Shape[i] != expected[i])
            {
                throw new InvalidOperationException($"SAM 3 {name} has shape {tensor.Shape}; expected dimension {i} = {expected[i]}.");
            }
        }
    }
}
