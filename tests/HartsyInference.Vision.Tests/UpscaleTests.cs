using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Vision.Upscale;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Structural tests for the Real-ESRGAN / RRDBNet upscaler: full forward path (conv-first →
/// RRDB dense blocks → upsample → conv-last) on a tiny synthetic-weight network, plus tiled inference
/// geometry and the config inference helper. Numeric parity against the Python Real-ESRGAN reference is
/// a checkpoint-gated follow-up; these confirm the graph runs and produces correctly-sized output.</summary>
public sealed class UpscaleTests
{
    [Fact]
    public void Upscale_WholeImage_ProducesScaledOutput()
    {
        using CpuBackend backend = new CpuBackend();
        RrdbNet net = BuildTinyNet(scale: 4, numBlock: 1, numFeat: 4, numGrowCh: 2, seed: 7);
        UpscalePipeline pipeline = new UpscalePipeline(backend, net, inputTileSize: 0); // no tiling

        const int w = 8, h = 8;
        byte[] src = Gradient(w, h);

        (byte[] outRgb, int outW, int outH) = pipeline.Upscale(src, w, h);

        Assert.Equal(32, outW);
        Assert.Equal(32, outH);
        Assert.Equal(outW * outH * 3, outRgb.Length);
        Assert.Equal(4, pipeline.ScaleFactor);
    }

    [Fact]
    public void Upscale_Tiled_MatchesOutputDimensions()
    {
        using CpuBackend backend = new CpuBackend();
        RrdbNet net = BuildTinyNet(scale: 4, numBlock: 1, numFeat: 4, numGrowCh: 2, seed: 11);
        UpscalePipeline pipeline = new UpscalePipeline(backend, net, inputTileSize: 8, tileOverlapFactor: 0.25f);

        const int w = 16, h = 16;
        byte[] src = Gradient(w, h);

        (byte[] outRgb, int outW, int outH) = pipeline.Upscale(src, w, h);

        Assert.Equal(64, outW);
        Assert.Equal(64, outH);
        Assert.Equal(outW * outH * 3, outRgb.Length);
    }

    [Fact]
    public void InferConfig_DetectsScaleAndBlocks()
    {
        // conv_up2 present → scale 4; highest body index 22 → 23 blocks.
        Dictionary<string, Tensor> w = new()
        {
            ["conv_first.weight"] = new Tensor(new TensorShape(64, 3, 3, 3), DType.F32),
            ["conv_up2.weight"] = new Tensor(new TensorShape(64, 64, 3, 3), DType.F32),
            ["body.0.rdb1.conv1.weight"] = new Tensor(new TensorShape(32, 64, 3, 3), DType.F32),
            ["body.22.rdb1.conv1.weight"] = new Tensor(new TensorShape(32, 64, 3, 3), DType.F32),
        };

        RealEsrganConfig cfg = RealEsrganConverter.InferConfig(w);
        Assert.Equal(4, cfg.Scale);
        Assert.Equal(23, cfg.NumBlock);
        Assert.Equal(64, cfg.NumFeat);
        Assert.Equal(32, cfg.NumGrowCh);

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void Converter_StripsBasicSrPrefix()
    {
        Dictionary<string, Tensor> w = new()
        {
            ["params_ema.conv_first.weight"] = new Tensor(new TensorShape(1), DType.F32),
        };
        Dictionary<string, Tensor> converted = RealEsrganConverter.Convert(w);
        Assert.True(converted.ContainsKey("conv_first.weight"));
        Assert.False(converted.ContainsKey("params_ema.conv_first.weight"));
        foreach (Tensor t in w.Values) t.Dispose();
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private static byte[] Gradient(int w, int h)
    {
        byte[] rgb = new byte[w * h * 3];
        for (int i = 0; i < w * h; i++)
        {
            rgb[i * 3] = (byte)(i % 256);
            rgb[i * 3 + 1] = (byte)((i * 2) % 256);
            rgb[i * 3 + 2] = (byte)((i * 3) % 256);
        }
        return rgb;
    }

    private static RrdbNet BuildTinyNet(int scale, int numBlock, int numFeat, int numGrowCh, int seed)
    {
        RealEsrganConfig cfg = new() { NumFeat = numFeat, NumBlock = numBlock, NumGrowCh = numGrowCh, Scale = scale };
        Dictionary<string, Tensor> w = new();

        uint state = (uint)seed;
        float Next()
        {
            state = state * 1664525u + 1013904223u;
            return ((state >> 8) / (float)(1 << 24) - 0.5f) * 0.1f;
        }

        void Conv(string name, int outC, int inC)
        {
            Tensor weight = new Tensor(new TensorShape(outC, inC, 3, 3), DType.F32);
            Span<float> s = weight.AsSpan<float>();
            for (int i = 0; i < s.Length; i++) s[i] = Next();
            w[$"{name}.weight"] = weight;

            Tensor bias = new Tensor(new TensorShape(outC), DType.F32);
            bias.AsSpan<float>().Clear();
            w[$"{name}.bias"] = bias;
        }

        Conv("conv_first", numFeat, 3);
        for (int b = 0; b < numBlock; b++)
        {
            for (int r = 1; r <= 3; r++)
            {
                string p = $"body.{b}.rdb{r}";
                Conv($"{p}.conv1", numGrowCh, numFeat);
                Conv($"{p}.conv2", numGrowCh, numFeat + numGrowCh);
                Conv($"{p}.conv3", numGrowCh, numFeat + 2 * numGrowCh);
                Conv($"{p}.conv4", numGrowCh, numFeat + 3 * numGrowCh);
                Conv($"{p}.conv5", numFeat, numFeat + 4 * numGrowCh);
            }
        }
        Conv("conv_body", numFeat, numFeat);
        Conv("conv_up1", numFeat, numFeat);
        if (scale == 4) Conv("conv_up2", numFeat, numFeat);
        Conv("conv_hr", numFeat, numFeat);
        Conv("conv_last", 3, numFeat);

        RrdbNet net = new RrdbNet(cfg);
        net.LoadWeights(w);
        return net;
    }
}
