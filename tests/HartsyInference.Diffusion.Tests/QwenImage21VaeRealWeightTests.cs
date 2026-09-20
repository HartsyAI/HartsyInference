using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Decodes through the real Qwen-Image 2.1 VAE. It is the Wan 2.2 decoder at parameters nothing else in
/// the repo uses — patch 1, temporal kernel 1, five stages, 64 latent channels and a four-channel head — so the
/// wiring that matters is whether those arguments reconstruct the file's actual shapes, which a synthetic test
/// cannot check. Skipped when the VAE is not on this machine.</summary>
[Trait("Category", "Integration")]
public sealed class QwenImage21VaeRealWeightTests
{
    private static string? VaePath()
    {
        foreach (string candidate in new[]
        {
            "/mnt/model-storage/Models/VAE/qwen_image_2.1_vae_bf16.safetensors",
            Path.Combine(AppContext.BaseDirectory, "../../../../../Models/VAE/qwen_image_2.1_vae_bf16.safetensors"),
        })
        {
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    [Fact]
    public void TheDecoderLoadsTheRealFileAndProducesFourChannelsAtSixteenTimesScale()
    {
        string? path = VaePath();
        if (path is null) return;

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        Dictionary<string, Tensor> weights =
            VaePrecisionHelper.CastWeights(loader.GetAllTensors(), [DType.F16, DType.BF16], DType.F32);

        int decDim = (int)weights["decoder.head.0.gamma"].Shape[0];
        int zDim = (int)weights["conv2.weight"].Shape[0];
        Assert.Equal(144, decDim);
        Assert.Equal(64, zDim);

        Wan22VaeDecoder vae = new Wan22VaeDecoder(
            dim: decDim, zDim: zDim, dimMult: [1, 2, 4, 8, 8], numResBlocks: 2,
            temperalUpsample: [true, true, true, false], patchSize: 1, temporalKernel: 1,
            latentMean: QwenImage21LatentNorm.Mean, latentStd: QwenImage21LatentNorm.Std);
        vae.LoadWeights(weights);

        using CpuBackend backend = new CpuBackend();
        const int h = 4, w = 4;
        Tensor latent = new Tensor(new TensorShape([1L, zDim, 1L, h, w]), DType.F32);
        Random rng = new Random(9);
        Span<float> s = latent.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);

        using Tensor decoded = vae.Decode(backend, latent);
        latent.Dispose();

        Assert.Equal(5, decoded.Shape.Rank);
        Assert.Equal(1, decoded.Shape[0]);
        Assert.Equal(4, decoded.Shape[1]);       // RGBA, not RGB
        Assert.Equal(1, decoded.Shape[2]);
        Assert.Equal(h * 16, decoded.Shape[3]);
        Assert.Equal(w * 16, decoded.Shape[4]);
        foreach (float value in decoded.AsReadOnlySpan<float>())
        {
            Assert.True(float.IsFinite(value), $"decoded pixel is {value}");
        }
    }
}
