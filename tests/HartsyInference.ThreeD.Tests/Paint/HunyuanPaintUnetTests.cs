using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Unit lane, tiny random weights on the CPU backend: the 2.5D UNet reduces to the stock Stable Diffusion UNet when its added output projections are zero
/// (as they are at upstream's initialisation), and the reference stream records one feature block per transformer layer.</summary>
public sealed unsafe class HunyuanPaintUnetTests
{
    private const int Views = 2, Size = 8;

    internal static Tensor Random(int seed, params long[] shape)
    {
        Random random = new(seed);
        Tensor t = new(new TensorShape(shape), DType.F32);
        for (int i = 0; i < t.ElementCount; i++)
            ((float*)t.DataPointer)[i] = (float)(random.NextDouble() * 2 - 1);
        return t;
    }

    [Fact]
    public void WithZeroedExtensions_MatchesTheStockUnetOnTheSameWeights()
    {
        HunyuanPaintConfig cfg = HunyuanPaintTestWeights.TinyConfig(v21: true) with { Materials = ["albedo"] };
        Dictionary<string, Tensor> paint = HunyuanPaintTestWeights.Create(cfg, generating: true, seed: 3, extraScale: 0f);
        Dictionary<string, Tensor> stockWeights = HunyuanPaintTestWeights.StockNames(paint);
        UNet stock = new(new UNetConfig
        {
            ModelChannels = 32, BlockOutChannels = [32, 32, 64, 64], NumAttentionHeads = [1, 1, 2, 2], LayersPerBlock = 1, CrossAttentionDim = 16, UseLinearProjection = true,
        });
        stock.LoadWeights(stockWeights);
        HunyuanPaintUnet unet = new(cfg, generating: true);
        unet.LoadWeights(paint);

        using CpuBackend backend = new();
        using Tensor sample = Random(1, Views, 4, Size, Size);
        using Tensor text = Random(2, 1, cfg.TextTokens, cfg.CrossAttentionDim);
        using Tensor dino = Random(4, 1, 6, cfg.CrossAttentionDim);
        using Tensor reference = Random(5, 1, 4, Size, Size);
        using Tensor repeatedText = new(new TensorShape(Views, cfg.TextTokens, cfg.CrossAttentionDim), DType.F32);
        backend.Concat(repeatedText, [text, text], 0);

        HunyuanPaintUnet dual = new(cfg, generating: false);
        dual.LoadWeights(HunyuanPaintTestWeights.Create(cfg, generating: false, seed: 9));
        HunyuanPaintBlockState write = new() { Write = true, Text = [text] };
        foreach (Tensor t in dual.Forward(backend, [reference], 0f, [], write))
            t.Dispose();
        Assert.Equal(CountTransformerBlocks(cfg), write.Embeds.Count);

        HunyuanPaintBlockState state = new() { Embeds = write.Embeds, Text = [text], Dino = dino };
        Tensor[] ours = unet.Forward(backend, [sample], 37f, [], state);
        using Tensor expected = stock.Forward(backend, sample, 37f, repeatedText);

        Assert.Single(ours);
        Assert.Equal(expected.Shape, ours[0].Shape);
        float maxDiff = 0;
        for (int i = 0; i < expected.ElementCount; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(((float*)expected.DataPointer)[i] - ((float*)ours[0].DataPointer)[i]));
        Assert.True(maxDiff < 2e-3f, $"max difference {maxDiff}");
        ours[0].Dispose();
    }

    [Fact]
    public void WithRandomExtensions_ReferenceScaleZeroDropsTheReferenceAndMaterialsStayIndependentOfEachOther()
    {
        HunyuanPaintConfig cfg = HunyuanPaintTestWeights.TinyConfig(v21: true);
        HunyuanPaintUnet unet = new(cfg, generating: true);
        unet.LoadWeights(HunyuanPaintTestWeights.Create(cfg, generating: true, seed: 11));

        using CpuBackend backend = new();
        using Tensor albedo = Random(1, Views, 4, Size, Size), mr = Random(2, Views, 4, Size, Size);
        using Tensor text0 = Random(3, 1, cfg.TextTokens, cfg.CrossAttentionDim), text1 = Random(4, 1, cfg.TextTokens, cfg.CrossAttentionDim);
        using Tensor dino = Random(5, 1, 6, cfg.CrossAttentionDim);
        HunyuanPaintBlockState noReference = new() { ReferenceScale = 0f, Text = [text0, text1], Dino = dino };

        Tensor[] both = unet.Forward(backend, [albedo, mr], 500f, [], noReference);
        Tensor[] albedoOnly = unet.Forward(backend, [albedo], 500f, [], new HunyuanPaintBlockState { ReferenceScale = 0f, Text = [text0], Dino = dino });

        Assert.Equal(2, both.Length);
        // With no reference the albedo stream is the first material's own computation (the second material has no way to reach it).
        for (int i = 0; i < both[0].ElementCount; i++)
            Assert.Equal(albedoOnly[0].As<float>(i), both[0].As<float>(i), 4);
        Assert.All(both.Concat(albedoOnly), t => Assert.True(float.IsFinite(t.As<float>(0))));
        foreach (Tensor t in both.Concat(albedoOnly))
            t.Dispose();
    }

    private static int CountTransformerBlocks(HunyuanPaintConfig cfg)
    {
        int levels = cfg.BlockOutChannels.Length;
        return (levels - 1) * cfg.LayersPerBlock + 1 + (levels - 1) * (cfg.LayersPerBlock + 1);
    }
}

internal static unsafe class TensorTestExtensions
{
    internal static T As<T>(this Tensor t, int index) where T : unmanaged => ((T*)t.DataPointer)[index];
}
