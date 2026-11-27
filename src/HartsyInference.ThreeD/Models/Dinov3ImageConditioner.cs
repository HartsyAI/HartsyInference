using HartsyInference.Core.Backends;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Dinov3;

namespace HartsyInference.ThreeD.Models;

/// <summary>
/// The DINOv3 image conditioner of the image-to-3D models: Lanczos-resize the prepared RGB square to the conditioning resolution,
/// normalize with ImageNet statistics, run a DINOv3 ViT/16, and apply a non-affine <c>F.layer_norm</c> (eps 1e-5) to the token sequence.
/// TRELLIS.2 and Pixal3D (upstream <c>DinoV3FeatureExtractor</c>, ViT-L) normalise the output of the last block without DINOv3's own final
/// norm; Idol (ViT-H+) takes Hugging Face's <c>last_hidden_state</c>, which includes it (<c>finalNorm</c>). The tokens are the class token,
/// the four register tokens and every patch, so a 512 px input gives 1029 tokens and a 1024 px input 4101.
/// </summary>
public sealed class Dinov3ImageConditioner : IDisposable
{
    /// <summary>Epsilon of the final <c>F.layer_norm</c>.</summary>
    public const float NormEps = 1e-5f;

    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    private readonly Dinov3VisionEncoder _encoder;
    private readonly IDisposable? _loader;
    private bool _ownsEncoder;   // set by Load, which made the encoder
    private int _disposed;

    /// <param name="encoder">A DINOv3 encoder with weights loaded; borrowed.</param>
    /// <param name="loader">Released with this conditioner; null when the caller owns the weights.</param>
    public Dinov3ImageConditioner(Dinov3VisionEncoder encoder, IDisposable? loader = null)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _loader = loader;
    }

    /// <summary>Loads an HF-layout DINOv3 checkpoint (ViT-S/B/L, or the SwiGLU ViT-H+) for inputs of <paramref name="resolution"/> pixels.</summary>
    public static Dinov3ImageConditioner Load(string safetensorsPath, int resolution = 512)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(safetensorsPath);
        SafeTensorsLoader loader = new();
        try
        {
            loader.Load(safetensorsPath);
            Dictionary<string, Tensor> weights = loader.GetAllTensors();
            Dinov2Preset preset = Dinov2Preset.FromDinov3Weights(weights, resolution);
            Dinov3VisionEncoder encoder = new(preset);
            encoder.LoadWeights(weights);
            return new Dinov3ImageConditioner(encoder, loader) { _ownsEncoder = true };
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

    /// <summary>Conditioning tokens <c>[1, TokenCount, hidden]</c> for the RGB24 square <paramref name="rgb"/>; the caller owns the result.
    /// <paramref name="finalNorm"/> applies DINOv3's own final LayerNorm before the non-affine one.</summary>
    public Tensor Encode(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height, int resolution, bool finalNorm = false)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (resolution <= 0 || resolution % 16 != 0)
            throw new ArgumentOutOfRangeException(nameof(resolution), resolution, "The conditioning resolution must be a positive multiple of 16.");
        using Tensor pixels = Pixels(rgb, width, height, resolution);
        using Tensor tokens = finalNorm ? _encoder.Encode(backend, pixels) : _encoder.EncodeBlocks(backend, pixels, _encoder.Preset.NumLayers - 1);
        Tensor cond = new(tokens.Shape, DType.F32);
        backend.LayerNormNoAffine(cond, tokens, NormEps);
        return cond;
    }

    /// <summary>The network input <c>[1, 3, R, R]</c>: PIL-style Lanczos resize to 8 bits, then <c>(x / 255 − mean) / std</c>; with
    /// <paramref name="normalize"/> false the values stay in <c>[0, 1]</c>, the guide image Pixal3D's NAF upsampler reads.</summary>
    internal static unsafe Tensor Pixels(ReadOnlySpan<byte> rgb, int width, int height, int resolution, bool normalize = true)
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
                p[c * plane + i] = normalize ? (level / 255f - Mean[c]) / Std[c] : level / 255f;
            }
        }

        return pixels;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (_ownsEncoder) _encoder.Dispose();
        _loader?.Dispose();
    }
}
