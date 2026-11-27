using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config LTX-2 encoder fixtures written by <c>Fixtures/LtxVideo2VaeEncoder/generate_ltx2_vae_encoder_fixtures.py</c>,
/// plus the tensor helpers the encoder tests share.</summary>
internal static class LtxVideo2VaeEncoderFixtures
{
    public const string WeightsFile = "ltx2_vae_encoder_tiny_weights.safetensors";
    public const string IoFile = "ltx2_vae_encoder_tiny_io.safetensors";

    /// <summary>Must match the <c>TINY</c> dictionary in the generator.</summary>
    public static LtxVideo2VaeEncoderConfig TinyConfig() => new()
    {
        LatentChannels = 8,
        BlockOutChannels = [8, 8, 16],
        LayersPerBlock = [1, 1, 1, 1],
        SpatioTemporalScaling = [true, true, true],
        DownsampleType = ["spatial", "temporal", "spatiotemporal"],
        PatchSize = 4,
    };

    public static SafeTensorsLoader Open(string fileName)
    {
        SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "LtxVideo2VaeEncoder", fileName));
        return loader;
    }

    public static LtxVideo2VaeEncoder LoadEncoder(LtxVideo2VaeEncoderConfig config, IReadOnlyDictionary<string, Tensor> weights)
    {
        LtxVideo2VaeEncoder encoder = new(config);
        encoder.LoadWeights(weights);
        return encoder;
    }

    public static Tensor Clone(Tensor video)
    {
        Tensor copy = new(video.Shape, DType.F32);
        video.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
        return copy;
    }

    /// <summary>Frame <paramref name="t"/> of a batch-1 video as a new <c>[1, C, 1, H, W]</c> tensor.</summary>
    public static Tensor CopyFrame(Tensor video, int t)
    {
        int c = (int)video.Shape[1], f = (int)video.Shape[2], plane = (int)(video.Shape[3] * video.Shape[4]);
        Tensor frame = new(new TensorShape([1, c, 1, video.Shape[3], video.Shape[4]]), DType.F32);
        ReadOnlySpan<float> src = video.AsReadOnlySpan<float>();
        Span<float> dst = frame.AsSpan<float>();
        for (int ci = 0; ci < c; ci++)
        {
            src.Slice((ci * f + t) * plane, plane).CopyTo(dst.Slice(ci * plane, plane));
        }
        return frame;
    }

    /// <summary>Adds <paramref name="delta"/> to every element of frame <paramref name="t"/> of an owned batch-1 video.</summary>
    public static void AddToFrame(Tensor video, int t, float delta)
    {
        int c = (int)video.Shape[1], f = (int)video.Shape[2], plane = (int)(video.Shape[3] * video.Shape[4]);
        Span<float> data = video.AsSpan<float>();
        for (int ci = 0; ci < c; ci++)
        {
            Span<float> slice = data.Slice((ci * f + t) * plane, plane);
            for (int i = 0; i < slice.Length; i++) slice[i] += delta;
        }
    }

    /// <summary>All channels of latent frame <paramref name="t"/> of a batch-1 latent, channel-major.</summary>
    public static float[] LatentFrame(Tensor latent, int t)
    {
        int c = (int)latent.Shape[1], f = (int)latent.Shape[2], plane = (int)(latent.Shape[3] * latent.Shape[4]);
        ReadOnlySpan<float> src = latent.AsReadOnlySpan<float>();
        float[] values = new float[c * plane];
        for (int ci = 0; ci < c; ci++)
        {
            src.Slice((ci * f + t) * plane, plane).CopyTo(values.AsSpan(ci * plane, plane));
        }
        return values;
    }

    public static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float max = 0f;
        for (int i = 0; i < a.Length; i++) max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        return max;
    }

    public static float MaxAbs(ReadOnlySpan<float> a)
    {
        float max = 0f;
        for (int i = 0; i < a.Length; i++) max = MathF.Max(max, MathF.Abs(a[i]));
        return max;
    }
}
