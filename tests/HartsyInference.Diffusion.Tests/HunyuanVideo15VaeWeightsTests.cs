using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The VAE's weight enumeration is the set its decode reads. <c>SyntheticSmoke</c> on the 4-channel tiny VAE: zeroing any tensor from
/// <see cref="HunyuanVideo15Vae.EnumerateWeights"/> must change the decode, so each is read; zeroing any tensor from
/// <see cref="HunyuanVideo15Vae.EnumerateEncodeWeights"/> must leave the decode unchanged, so the decode does not read the encoder.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class HunyuanVideo15VaeWeightsTests
{
    private const int DeclineWarmups = 6;

    [Fact]
    public void EveryDecoderWeightIsOneTheDecodeReads()
    {
        using SafeTensorsLoader weights = Load();
        using CpuBackend backend = new();
        using HunyuanVideo15Vae vae = NewVae(weights);
        using Tensor latent = TorchCompatibleNoise.Randn(new TensorShape([1L, 4, 2, 2, 3]), 3);
        // The conv weight forms settle over the first calls, so the enumeration is taken after them.
        float[] baseline = Warm(vae, backend, latent);

        List<Tensor> decoder = vae.EnumerateWeights().ToList();
        Assert.NotEmpty(decoder);
        List<int> unread = [];
        for (int i = 0; i < decoder.Count; i++)
            if (!Changes(vae, backend, latent, baseline, decoder[i])) unread.Add(i);
        Assert.True(unread.Count == 0,
            $"{unread.Count} of {decoder.Count} decoder weights do not change the decode (indices {string.Join(",", unread)}).");
    }

    [Fact]
    public void ZeroingAnEncoderWeightLeavesTheDecodeUnchanged()
    {
        using SafeTensorsLoader weights = Load();
        using CpuBackend backend = new();
        using HunyuanVideo15Vae vae = NewVae(weights);
        using Tensor latent = TorchCompatibleNoise.Randn(new TensorShape([1L, 4, 2, 2, 3]), 3);
        float[] baseline = Warm(vae, backend, latent);

        List<Tensor> encoder = vae.EnumerateEncodeWeights().ToList();
        Assert.NotEmpty(encoder);
        List<int> read = [];
        for (int i = 0; i < encoder.Count; i++)
            if (Changes(vae, backend, latent, baseline, encoder[i])) read.Add(i);
        Assert.True(read.Count == 0,
            $"The decode changes when {read.Count} encoder weights are zeroed (indices {string.Join(",", read)}).");
    }

    [Fact]
    public void DisposedVaeHoldsNoWeightsAndCannotDecode()
    {
        using SafeTensorsLoader weights = Load();
        using CpuBackend backend = new();
        HunyuanVideo15Vae vae = NewVae(weights);
        using Tensor latent = TorchCompatibleNoise.Randn(new TensorShape([1L, 4, 2, 2, 3]), 3);
        Warm(vae, backend, latent);

        vae.Dispose();

        Assert.Empty(vae.EnumerateWeights());
        Assert.Empty(vae.EnumerateEncodeWeights());
        Assert.Throws<InvalidOperationException>(() => vae.Decode(backend, latent));
    }

    /// <summary>Decodes after the conv forms have settled, and returns that output.</summary>
    private static float[] Warm(HunyuanVideo15Vae vae, IBackend backend, Tensor latent)
    {
        float[] output = [];
        for (int i = 0; i < DeclineWarmups; i++)
            output = Decode(vae, backend, latent);
        return output;
    }

    /// <summary>Whether zeroing <paramref name="weight"/> changes the decode. The weight's bytes are restored afterwards.</summary>
    private static bool Changes(HunyuanVideo15Vae vae, IBackend backend, Tensor latent, float[] baseline, Tensor weight)
    {
        byte[] saved = weight.AsReadOnlySpan<byte>().ToArray();
        weight.AsSpan<byte>().Clear();
        try
        {
            float[] zeroed = Decode(vae, backend, latent);
            return Differs(baseline, zeroed);
        }
        finally
        {
            saved.AsSpan().CopyTo(weight.AsSpan<byte>());
        }
    }

    private static float[] Decode(HunyuanVideo15Vae vae, IBackend backend, Tensor latent)
    {
        using Tensor frames = vae.Decode(backend, latent);
        return frames.AsReadOnlySpan<float>().ToArray();
    }

    private static bool Differs(float[] a, float[] b)
    {
        if (a.Length != b.Length) return true;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return true;
        return false;
    }

    private static SafeTensorsLoader Load()
    {
        SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_vae_tiny_z4.safetensors"));
        return weights;
    }

    private static HunyuanVideo15Vae NewVae(SafeTensorsLoader weights)
    {
        HunyuanVideo15VaeConfig config = new()
        {
            LatentChannels = 4, BlockOutChannels = [4, 8, 16, 32, 32], LayersPerBlock = 2,
            SpatialCompression = 16, TemporalCompression = 4, ScalingFactor = 2f,
        };
        HunyuanVideo15Vae vae = new(config);
        vae.LoadWeights(weights.GetAllTensors()
            .Where(pair => pair.Key.StartsWith("encoder.", StringComparison.Ordinal) || pair.Key.StartsWith("decoder.", StringComparison.Ordinal))
            .ToDictionary(pair => pair.Key, pair => pair.Value));
        return vae;
    }
}
