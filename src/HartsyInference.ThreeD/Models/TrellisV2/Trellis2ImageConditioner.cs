using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Codec;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Dinov3;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// Upstream <c>DinoV3FeatureExtractor</c>: Lanczos-resize the prepared RGB square to the conditioning resolution, normalize with
/// ImageNet statistics, run DINOv3 ViT-L/16, and apply a non-affine <c>F.layer_norm</c> (eps 1e-5) to the token sequence after the
/// last block, without DINOv3's own final norm. The tokens are the class token, the four register tokens and every patch, so a
/// 512 px input gives 1029 tokens and a 1024 px input 4101.
/// </summary>
public sealed class Trellis2ImageConditioner : IDisposable
{
    /// <summary>Epsilon of the final <c>F.layer_norm</c>.</summary>
    public const float NormEps = 1e-5f;

    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    private readonly Dinov3VisionEncoder _encoder;
    private readonly IDisposable? _loader;
    private int _disposed;

    /// <param name="encoder">A DINOv3 encoder with weights loaded; borrowed.</param>
    /// <param name="loader">Released with this conditioner; null when the caller owns the weights.</param>
    public Trellis2ImageConditioner(Dinov3VisionEncoder encoder, IDisposable? loader = null)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _loader = loader;
    }

    /// <summary>Loads an HF-layout DINOv3 ViT-L/16 checkpoint.</summary>
    public static Trellis2ImageConditioner Load(string safetensorsPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(safetensorsPath);
        SafeTensorsLoader loader = new();
        try
        {
            loader.Load(safetensorsPath);
            Dictionary<string, Tensor> weights = loader.GetAllTensors();
            Dinov2Preset preset = Dinov2Preset.FromDinov3Weights(weights, 512);
            Dinov3VisionEncoder encoder = new(preset);
            encoder.LoadWeights(weights);
            return new Trellis2ImageConditioner(encoder, loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>The tensors the encoder reads, for <c>IBackend.PreloadWeights</c> and <c>FreeWeights</c>.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _encoder.EnumerateWeights();

    /// <summary>Tokens for an input of <paramref name="resolution"/> pixels: class, register and patch tokens.</summary>
    public int TokenCount(int resolution) => 1 + _encoder.Preset.NumRegisterTokens + (resolution / 16) * (resolution / 16);

    /// <summary>Conditioning tokens <c>[1, TokenCount, 1024]</c> for the RGB24 square <paramref name="rgb"/>; the caller owns the result.</summary>
    public Tensor Encode(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height, int resolution)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (resolution <= 0 || resolution % 16 != 0)
            throw new ArgumentOutOfRangeException(nameof(resolution), resolution, "The conditioning resolution must be a positive multiple of 16.");
        using Tensor pixels = Pixels(rgb, width, height, resolution);
        using Tensor tokens = _encoder.EncodeBlocks(backend, pixels, _encoder.Preset.NumLayers - 1);
        Tensor cond = new(tokens.Shape, DType.F32);
        backend.LayerNormNoAffine(cond, tokens, NormEps);
        return cond;
    }

    /// <summary>The network input <c>[1, 3, R, R]</c>: PIL-style Lanczos resize to 8 bits, then <c>(x / 255 − mean) / std</c>.</summary>
    internal static unsafe Tensor Pixels(ReadOnlySpan<byte> rgb, int width, int height, int resolution)
    {
        if (width <= 0 || height <= 0 || rgb.Length != width * height * 3)
            throw new ArgumentException($"RGB buffer of {rgb.Length} bytes does not fit {width}x{height}.", nameof(rgb));
        int plane = resolution * resolution;
        float[] resized = new float[plane * 3];
        Resample.LanczosHwc8(rgb, width, height, 3, resized, resolution, resolution);
        Tensor pixels = new(new TensorShape(1, 3, resolution, resolution), DType.F32);
        float* p = (float*)pixels.DataPointer;
        for (int i = 0; i < plane; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                float level = Math.Clamp((int)(resized[i * 3 + c] + 0.5f), 0, 255);
                p[c * plane + i] = (level / 255f - Mean[c]) / Std[c];
            }
        }

        return pixels;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _loader?.Dispose();
    }
}
