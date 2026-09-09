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
    public void Upscale_X2Model_UnshufflesInputAndDoublesOutput()
    {
        using CpuBackend backend = new CpuBackend();
        RrdbNet net = BuildTinyNet(scale: 2, numBlock: 1, numFeat: 4, numGrowCh: 2, seed: 5);
        UpscalePipeline pipeline = new UpscalePipeline(backend, net, inputTileSize: 0);

        // Odd edges: 9x7 must come back exactly 18x14, not rounded through the even padding.
        const int w = 9, h = 7;
        (byte[] outRgb, int outW, int outH) = pipeline.Upscale(Gradient(w, h), w, h);

        Assert.Equal(2, pipeline.ScaleFactor);
        Assert.Equal(18, outW);
        Assert.Equal(14, outH);
        Assert.Equal(outW * outH * 3, outRgb.Length);
    }

    [Fact]
    public void Upscale_X2Model_Tiled_MatchesOutputDimensions()
    {
        using CpuBackend backend = new CpuBackend();
        RrdbNet net = BuildTinyNet(scale: 2, numBlock: 1, numFeat: 4, numGrowCh: 2, seed: 9);
        // 16 input px per tile → 8 unshuffled cells per tile over a 12-cell unshuffled image: two tiles each way.
        UpscalePipeline pipeline = new UpscalePipeline(backend, net, inputTileSize: 16, tileOverlapFactor: 0.25f);

        const int w = 24, h = 24;
        (byte[] outRgb, int outW, int outH) = pipeline.Upscale(Gradient(w, h), w, h);

        Assert.Equal(48, outW);
        Assert.Equal(48, outH);
        Assert.Equal(outW * outH * 3, outRgb.Length);
    }

    [Fact]
    public void UnshuffledInput_UsesTorchChannelOrderAndReplicatesEdges()
    {
        // 3x1 image, r=2: padded to 4x2 by replicating the last column and the only row.
        byte[] rgb = [10, 11, 12, 20, 21, 22, 30, 31, 32];
        using Tensor t = UpscalePipeline.UnshuffledInput(rgb, 3, 1, 2, padW: 1, padH: 1);

        long[] expectedShape = [1, 12, 1, 2];
        long[] actualShape = [t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]];
        Assert.Equal(expectedShape, actualShape);
        Span<float> s = t.AsSpan<float>();
        // channel = c*4 + i*2 + j; cell (0,0) covers source (0..1, 0..1) → i=0,j=0 is pixel (0,0), j=1 is (0,1).
        Assert.Equal(10 / 255f, s[0 * 2 + 0]);          // R, i=0,j=0, cell 0 → pixel x=0
        Assert.Equal(20 / 255f, s[1 * 2 + 0]);          // R, i=0,j=1, cell 0 → pixel x=1
        Assert.Equal(10 / 255f, s[2 * 2 + 0]);          // R, i=1,j=0 → row replicated
        Assert.Equal(30 / 255f, s[0 * 2 + 1]);          // R, i=0,j=0, cell 1 → pixel x=2
        Assert.Equal(30 / 255f, s[1 * 2 + 1]);          // R, i=0,j=1, cell 1 → column replicated
        Assert.Equal(11 / 255f, s[4 * 2 + 0]);          // G, i=0,j=0, cell 0
        Assert.Equal(32 / 255f, s[8 * 2 + 1]);          // B, i=0,j=0, cell 1
    }

    [Fact]
    public void InferConfig_TwelveChannelConvFirst_IsScale2()
    {
        Dictionary<string, Tensor> w = new()
        {
            ["conv_first.weight"] = new Tensor(new TensorShape(64, 12, 3, 3), DType.F32),
            ["conv_up2.weight"] = new Tensor(new TensorShape(64, 64, 3, 3), DType.F32),
            ["body.0.rdb1.conv1.weight"] = new Tensor(new TensorShape(32, 64, 3, 3), DType.F32),
        };
        RealEsrganConfig cfg = RealEsrganConverter.InferConfig(w);
        Assert.Equal(2, cfg.Scale);
        Assert.Equal(2, cfg.UnshuffleFactor);
        Assert.Equal(12, cfg.InputChannels);
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void InferConfig_DetectsScaleAndBlocks()
    {
        // 3-channel conv_first → scale 4 (conv_up2 is present on every checkpoint); highest body index 22 → 23 blocks.
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

        Conv("conv_first", numFeat, cfg.InputChannels);
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
        Conv("conv_up2", numFeat, numFeat);
        Conv("conv_hr", numFeat, numFeat);
        Conv("conv_last", 3, numFeat);

        RrdbNet net = new RrdbNet(cfg);
        net.LoadWeights(w);
        return net;
    }
}
