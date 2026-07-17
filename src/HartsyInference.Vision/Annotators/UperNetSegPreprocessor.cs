using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Annotators;

/// <summary>Semantic-segmentation pre/post-processing matching the diffusers / HF reference pipeline for
/// <c>control_v11p_sd15_seg</c> (transformers <c>SegformerImageProcessor</c> +
/// <c>UperNetForSemanticSegmentation.post_process_semantic_segmentation</c>): the caller stretch-resizes
/// the source to the detect resolution (the reference uses 512×512, PIL bilinear), this class normalizes
/// (/255, ImageNet mean/std), runs the model, argmaxes the upsampled logits, and colorizes with the
/// standard ADE20K palette (<see cref="Ade20kPalette"/>) — the seg conditioning form. Any rescale of the
/// palette map to the generation resolution must be nearest-neighbor to keep colors on exact class
/// values. Host code by design: runs once per generation, outside the per-step hot path.</summary>
public sealed unsafe class UperNetSegPreprocessor
{
    /// <summary>Reference detect resolution (SegformerImageProcessor default for the ADE20K checkpoints).</summary>
    public const int ReferenceSize = 512;

    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    private readonly UperNetSegModel _model;

    public UperNetSegPreprocessor(UperNetSegModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
    }

    /// <summary>Wrapped model — exposed for weight preloading.</summary>
    public UperNetSegModel Model => _model;

    /// <summary>Converts interleaved RGB24 to the network input <c>[1,3,H,W]</c>
    /// (<c>(x/255 − mean)/std</c>, ImageNet statistics).</summary>
    public static Tensor Preprocess(ReadOnlySpan<byte> rgbPixels, int width, int height)
    {
        long expected = (long)width * height * 3;
        if (width <= 0 || height <= 0 || rgbPixels.Length != expected)
            throw new ArgumentException($"rgbPixels length {rgbPixels.Length} != {width}x{height}x3.", nameof(rgbPixels));
        Tensor output = new(new TensorShape(1, 3, height, width), DType.F32);
        float* dst = (float*)output.DataPointer;
        long plane = (long)width * height;
        fixed (byte* src = rgbPixels)
        {
            for (int c = 0; c < 3; c++)
            {
                float mean = Mean[c], invStd = 1f / Std[c];
                for (long i = 0; i < plane; i++)
                    dst[c * plane + i] = (src[i * 3 + c] / 255f - mean) * invStd;
            }
        }
        return output;
    }

    /// <summary>Runs the segmenter and returns the per-pixel ADE20K class map (row-major
    /// <c>width·height</c> bytes, class ids 0..149).</summary>
    public byte[] Process(IBackend backend, ReadOnlySpan<byte> rgbPixels, int width, int height)
    {
        using Tensor input = Preprocess(rgbPixels, width, height);
        Tensor logits = _model.Forward(backend, input);
        byte[] classMap = Argmax(logits);
        logits.Dispose();
        return classMap;
    }

    /// <summary>Runs the segmenter and returns the ADE20K-palette RGB24 conditioning map.</summary>
    public byte[] ProcessToRgb(IBackend backend, ReadOnlySpan<byte> rgbPixels, int width, int height)
        => Ade20kPalette.Colorize(Process(backend, rgbPixels, width, height));

    /// <summary>Per-pixel argmax over class logits <c>[1,K,H,W]</c> (K ≤ 256), ties to the lowest index
    /// like torch argmax.</summary>
    public static byte[] Argmax(Tensor logits)
    {
        if (logits.Shape.Rank != 4 || logits.Shape[0] != 1 || logits.Shape[1] > 256)
            throw new ArgumentException($"logits must be [1,K,H,W] with K <= 256; got {logits.Shape}.", nameof(logits));
        int k = (int)logits.Shape[1];
        long plane = logits.Shape[2] * logits.Shape[3];
        float* src = (float*)logits.DataPointer;
        byte[] classMap = new byte[plane];
        for (long i = 0; i < plane; i++)
        {
            float best = src[i];
            int bestC = 0;
            for (int c = 1; c < k; c++)
            {
                float v = src[c * plane + i];
                if (v > best)
                {
                    best = v;
                    bestC = c;
                }
            }
            classMap[i] = (byte)bestC;
        }
        return classMap;
    }
}
