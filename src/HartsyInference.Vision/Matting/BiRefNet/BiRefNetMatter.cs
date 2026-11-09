using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Matting.BiRefNet;

/// <summary>Trimap-free person/foreground matting with BiRefNet (<c>ZhengPeng7/BiRefNet-matting</c>, matting variant):
/// an image in, a soft alpha matte out at the source resolution. Pre- and post-processing follow the upstream
/// BiRefNet usage: bilinear resize to 1024² (antialiased, as torchvision does on PIL images), 8-bit quantization,
/// ImageNet normalization; sigmoid of the logits, truncated to 8 bits, then a bicubic resize back to the source size.
/// Pure C#: the weight dictionary comes from <c>BiRefNetCheckpointConverter</c> and is loaded once per model.</summary>
public sealed class BiRefNetMatter : IDisposable
{
    /// <summary>Square network input edge, from the upstream <c>Config.size</c>.</summary>
    public const int InputSize = 1024;

    private static readonly float[] ImageNetMean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] ImageNetStd = [0.229f, 0.224f, 0.225f];

    private readonly BiRefNetModel _model = new();
    private int _disposed;

    /// <summary>Loads the converted weight dictionary. Call once before <see cref="Matte"/>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> converted) => _model.LoadWeights(converted);

    /// <summary>Foreground alpha for interleaved RGB24 pixels, row-major <c>height × width</c>, in [0, 1].</summary>
    public float[] Matte(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (width <= 0 || height <= 0)
            throw new ArgumentException($"Image dimensions must be positive; got {width}x{height}.");
        if (rgb.Length < (long)width * height * 3)
            throw new ArgumentException($"RGB buffer holds {rgb.Length} bytes; a {width}x{height} image needs {(long)width * height * 3}.", nameof(rgb));

        using Tensor pixels = Preprocess(rgb, width, height);
        using Tensor logits = _model.Forward(backend, pixels);
        return Postprocess(logits, width, height);
    }

    /// <summary>Foreground alpha in [0, 1] at the source size, for the HairDeck preprocessing: RGB / 255 resized bilinearly to
    /// <paramref name="inputSize"/>² without antialiasing, ImageNet normalisation, float sigmoid, bilinear back to the source size.</summary>
    public unsafe float[] MatteFloat(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height, int inputSize)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (width <= 0 || height <= 0 || inputSize <= 0)
            throw new ArgumentException($"Image and input sizes must be positive; got {width}x{height} at {inputSize}.");
        if (rgb.Length < (long)width * height * 3)
            throw new ArgumentException($"RGB buffer holds {rgb.Length} bytes; a {width}x{height} image needs {(long)width * height * 3}.", nameof(rgb));

        int plane = inputSize * inputSize;
        float[] source = new float[(long)width * height];
        float[] resized = new float[plane];
        using Tensor pixels = new(new TensorShape(1, 3, inputSize, inputSize), DType.F32);
        float* op = (float*)pixels.DataPointer;
        for (int c = 0; c < 3; c++)
        {
            for (int i = 0; i < source.Length; i++)
                source[i] = rgb[i * 3 + c] / 255f;
            Resample.BilinearPlaneF32(source, width, height, resized, inputSize, inputSize);
            for (int i = 0; i < plane; i++)
                op[c * plane + i] = (resized[i] - ImageNetMean[c]) / ImageNetStd[c];
        }

        using Tensor logits = _model.Forward(backend, pixels);
        float* lp = (float*)logits.DataPointer;
        for (int i = 0; i < plane; i++)
            resized[i] = 1f / (1f + MathF.Exp(-lp[i]));

        float[] matte = new float[(long)width * height];
        Resample.BilinearPlaneF32(resized, inputSize, inputSize, matte, width, height);
        return matte;
    }

    /// <summary>Resize to 1024², quantize to 8 bits, scale to [0,1] and normalize per channel: <c>[1,3,1024,1024]</c>.</summary>
    internal static unsafe Tensor Preprocess(ReadOnlySpan<byte> rgb, int width, int height)
    {
        using Tensor resized = new(new TensorShape(InputSize * InputSize * 3), DType.F32);
        Span<float> hwc = resized.AsSpan<float>();
        Resample.BilinearHwc8(rgb[..(width * height * 3)], width, height, 3, hwc, InputSize, InputSize, antialias: true, quantizeU8: true);

        Tensor pixels = new(new TensorShape(1, 3, InputSize, InputSize), DType.F32);
        float* op = (float*)pixels.DataPointer;
        const int plane = InputSize * InputSize;
        for (int i = 0; i < plane; i++)
            for (int c = 0; c < 3; c++)
                op[c * plane + i] = (hwc[i * 3 + c] / 255f - ImageNetMean[c]) / ImageNetStd[c];
        return pixels;
    }

    /// <summary>Sigmoid, truncation to 8 bits (torchvision's <c>ToPILImage</c>), then a bicubic antialiased resize to the
    /// source size, rounded back to 8 bits and returned as <c>byte/255</c>.</summary>
    internal static unsafe float[] Postprocess(Tensor logits, int width, int height)
    {
        const int plane = InputSize * InputSize;
        float* lp = (float*)logits.DataPointer;
        byte[] quantized = new byte[plane];
        for (int i = 0; i < plane; i++)
        {
            float prob = 1f / (1f + MathF.Exp(-lp[i]));
            quantized[i] = (byte)MathF.Floor(prob * 255f);
        }

        // Shared PIL-style resize: quantizeU8 rounds and clamps each separable pass, as PIL's 8-bit intermediate does.
        float[] matte = new float[(long)width * height];
        Resample.BicubicHwc8(quantized, InputSize, InputSize, 1, matte, width, height, a: -0.5f, antialias: true, quantizeU8: true);
        for (int i = 0; i < matte.Length; i++)
            matte[i] = Math.Clamp(MathF.Round(matte[i]), 0f, 255f) / 255f;
        return matte;
    }

    /// <summary>Raw logits <c>[1,1,1024,1024]</c> for a preprocessed input. Exposed for parity checks against the reference.</summary>
    internal Tensor Logits(IBackend backend, Tensor pixels) => _model.Forward(backend, pixels);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _model.Dispose();
        GC.SuppressFinalize(this);
    }
}
