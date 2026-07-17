using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.Vision.Codec;
using HartsyInference.Vision.Detection;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>RT-DETR (PekingU/rtdetr_r18vd) tests: fast unit checks for the config, the BN-folding
/// checkpoint converter, and the deformable bilinear sampler; plus an env-gated real-weight parity
/// test against the <c>transformers RTDetrForObjectDetection</c> oracle on <c>bus.png</c>.</summary>
public sealed class RtDetrForwardTests
{
    private readonly ITestOutputHelper _output;

    public RtDetrForwardTests(ITestOutputHelper output) => _output = output;

    /// <summary>The transformers oracle top detections on <c>bus.png</c> (810×1080), pixel xyxy.</summary>
    private static readonly (int Label, float Score, float X1, float Y1, float X2, float Y2)[] OracleBus =
    [
        (5, 0.9534f, 15.5f, 232.8f, 806.2f, 746.9f),   // bus
        (0, 0.9468f, 49.2f, 397.6f, 247.0f, 907.2f),   // person
        (0, 0.9219f, 223.2f, 405.7f, 343.9f, 860.4f),  // person
        (0, 0.8782f, 667.9f, 393.8f, 809.7f, 882.9f),  // person
        (0, 0.7549f, -0.2f, 552.1f, 75.6f, 872.0f),    // person
    ];

    [Fact]
    public void Config_R18vd_Validates()
    {
        RtDetrConfig cfg = RtDetrConfig.R18vd;
        cfg.Validate();
        Assert.Equal(256, cfg.HiddenDim);
        Assert.Equal(32, cfg.HeadDim);
        Assert.Equal(128, cfg.RepHiddenChannels);
        Assert.Equal(3, cfg.NumDecoderLayers);
    }

    [Fact]
    public void Converter_FoldsBatchNorm_AndDropsBnBuffers()
    {
        // One 1×1 conv (2 out, 1 in) + BN. Fold math: scale = gamma/sqrt(var+eps),
        // w' = w*scale, b' = beta - mean*scale.  eps = 1e-5.
        const float eps = 1e-5f;
        Dictionary<string, Tensor> raw = new()
        {
            ["m.conv.weight"] = Vec(new TensorShape(2, 1, 1, 1), 2f, -3f),
            ["m.norm.weight"] = Vec(new TensorShape(2), 4f, 5f),   // gamma
            ["m.norm.bias"] = Vec(new TensorShape(2), 1f, -1f),    // beta
            ["m.norm.running_mean"] = Vec(new TensorShape(2), 0.5f, 2f),
            ["m.norm.running_var"] = Vec(new TensorShape(2), 3f, 0.25f),
            ["m.norm.num_batches_tracked"] = Vec(new TensorShape(1), 7f),
            ["head.weight"] = Vec(new TensorShape(1), 9f),          // untouched passthrough
        };

        Dictionary<string, Tensor> conv = RtDetrCheckpointConverter.Convert(raw);

        Assert.True(conv.ContainsKey("m.conv.weight"));
        Assert.True(conv.ContainsKey("m.conv.bias"));
        Assert.False(conv.ContainsKey("m.norm.weight"));
        Assert.False(conv.ContainsKey("m.norm.running_var"));
        Assert.False(conv.ContainsKey("m.norm.num_batches_tracked"));
        Assert.True(conv.ContainsKey("head.weight"));

        float scale0 = 4f / MathF.Sqrt(3f + eps);
        float scale1 = 5f / MathF.Sqrt(0.25f + eps);
        Span<float> w = conv["m.conv.weight"].AsSpan<float>();
        Span<float> b = conv["m.conv.bias"].AsSpan<float>();
        Assert.Equal(2f * scale0, w[0], 4);
        Assert.Equal(-3f * scale1, w[1], 4);
        Assert.Equal(1f - 0.5f * scale0, b[0], 4);
        Assert.Equal(-1f - 2f * scale1, b[1], 4);
    }

    [Fact]
    public void BilinearSampleZeroPad_KnownGrid_InterpolatesCorrectly()
    {
        // 2x2 single-channel grid laid [y*W + x]: (0,0)=0 (1,0)=1 (0,1)=2 (1,1)=3.
        float[] grid = [0f, 1f, 2f, 3f];
        Span<float> dst = stackalloc float[1];

        RtDetrDecoder.BilinearSampleZeroPad(grid, 0, 2, 2, 1, 0, 1, px: 0.5f, py: 0.5f, dst);
        Assert.Equal(1.5f, dst[0], 5);

        RtDetrDecoder.BilinearSampleZeroPad(grid, 0, 2, 2, 1, 0, 1, px: 1f, py: 1f, dst);
        Assert.Equal(3f, dst[0], 5);

        RtDetrDecoder.BilinearSampleZeroPad(grid, 0, 2, 2, 1, 0, 1, px: -0.5f, py: 0f, dst);
        Assert.Equal(0f, dst[0], 5);
    }

    /// <summary>Real-weight parity — gated on <c>RTDETR_PATH</c> (the rtdetr_r18vd safetensors). Runs
    /// the full pipeline on the bundled bus.png and checks the top detection's label + box IoU + score
    /// against the transformers oracle.</summary>
    [Trait("Category", "Integration")]
    [Fact]
    public void RealWeights_Parity_BusImage()
    {
        string? path = Environment.GetEnvironmentVariable("RTDETR_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            _output.WriteLine("SKIPPED: RTDETR_PATH unset or file missing.");
            return;
        }

        string busPath = Path.Combine(AppContext.BaseDirectory, "TestData", "bus.png");
        Assert.True(File.Exists(busPath), $"bus.png not found at {busPath}");
        (byte[] rgb, int w, int h) = PngDecoder.Decode(File.ReadAllBytes(busPath));

        using IBackend backend = new CpuBackend();
        using RtDetrPipeline pipeline = new(backend, RtDetrConfig.R18vd, path, inputSize: 640);
        Assert.Equal("rtdetr_r18vd", pipeline.ModelName);

        IReadOnlyList<DetectionResult> results = pipeline.Detect(rgb, w, h, confidenceThreshold: 0.3f);
        _output.WriteLine($"C# produced {results.Count} detections on bus.png ({w}x{h}):");
        foreach (DetectionResult d in results)
            _output.WriteLine($"  label={d.ClassIndex} score={d.Confidence:F4} box=[{d.X * w:F1},{d.Y * h:F1},{(d.X + d.Width) * w:F1},{(d.Y + d.Height) * h:F1}]");

        Assert.NotEmpty(results);

        // Top detection must be the bus (label 5) with high IoU + score against the oracle.
        DetectionResult top = results[0];
        (int Label, float Score, float X1, float Y1, float X2, float Y2) oracleTop = OracleBus[0];
        Assert.Equal(oracleTop.Label, top.ClassIndex);
        float topIou = Iou(top, w, h, oracleTop);
        _output.WriteLine($"Top detection IoU vs oracle bus = {topIou:F4}, score {top.Confidence:F4} (oracle {oracleTop.Score:F4})");
        Assert.True(topIou >= 0.9f, $"Top box IoU {topIou:F3} < 0.9");
        Assert.True(MathF.Abs(top.Confidence - oracleTop.Score) < 0.05f, $"Top score {top.Confidence:F3} vs oracle {oracleTop.Score:F3}");

        // Every oracle detection must be matched (same label + IoU >= 0.9); collect score pairs.
        List<float> csScores = new(), refScores = new();
        foreach ((int Label, float Score, float X1, float Y1, float X2, float Y2) oracle in OracleBus)
        {
            DetectionResult? best = null;
            float bestIou = 0f;
            foreach (DetectionResult d in results)
            {
                if (d.ClassIndex != oracle.Label) continue;
                float iou = Iou(d, w, h, oracle);
                if (iou > bestIou) { bestIou = iou; best = d; }
            }
            Assert.True(best is not null && bestIou >= 0.9f,
                $"No C# detection matched oracle (label {oracle.Label}, box [{oracle.X1},{oracle.Y1},{oracle.X2},{oracle.Y2}]); best IoU {bestIou:F3}");
            csScores.Add(best!.Confidence);
            refScores.Add(oracle.Score);
        }

        float corr = Pearson(csScores, refScores);
        _output.WriteLine($"Score correlation vs oracle = {corr:F4}");
        Assert.True(corr >= 0.9f, $"Score correlation {corr:F3} < 0.9");
    }

    private static float Iou(DetectionResult d, int w, int h, (int Label, float Score, float X1, float Y1, float X2, float Y2) o)
    {
        float ax1 = d.X * w, ay1 = d.Y * h, ax2 = (d.X + d.Width) * w, ay2 = (d.Y + d.Height) * h;
        float ix1 = MathF.Max(ax1, o.X1), iy1 = MathF.Max(ay1, o.Y1);
        float ix2 = MathF.Min(ax2, o.X2), iy2 = MathF.Min(ay2, o.Y2);
        float iw = MathF.Max(0f, ix2 - ix1), ih = MathF.Max(0f, iy2 - iy1);
        float inter = iw * ih;
        float areaA = MathF.Max(0f, ax2 - ax1) * MathF.Max(0f, ay2 - ay1);
        float areaB = (o.X2 - o.X1) * (o.Y2 - o.Y1);
        float union = areaA + areaB - inter;
        return union <= 0f ? 0f : inter / union;
    }

    private static float Pearson(List<float> a, List<float> b)
    {
        int n = a.Count;
        if (n < 2) return 1f;
        float ma = 0f, mb = 0f;
        for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
        ma /= n; mb /= n;
        float sab = 0f, saa = 0f, sbb = 0f;
        for (int i = 0; i < n; i++)
        {
            float da = a[i] - ma, db = b[i] - mb;
            sab += da * db; saa += da * da; sbb += db * db;
        }
        return (saa <= 0f || sbb <= 0f) ? 1f : sab / MathF.Sqrt(saa * sbb);
    }

    private static Tensor Vec(TensorShape shape, params float[] values)
    {
        Tensor t = new(shape, DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < values.Length && i < s.Length; i++)
            s[i] = values[i];
        return t;
    }
}
