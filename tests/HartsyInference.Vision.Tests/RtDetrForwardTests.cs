using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Vision.Detection;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>Structural tests for the RT-DETR port: a shape-correct forward over synthetic weights,
/// the pipeline post-processing contract, and a pure-unit check of the deformable bilinear sampler.
/// Numerics vs a reference checkpoint are validation-pending (see the env-gated Integration stub).</summary>
public sealed class RtDetrForwardTests
{
    private readonly ITestOutputHelper _output;

    public RtDetrForwardTests(ITestOutputHelper output) => _output = output;

    /// <summary>Tiny config — small enough for the perf-naive host deformable loop, still exercising
    /// every module (backbone, AIFI, CCFM, query selection, one deformable decoder layer, heads).</summary>
    private static RtDetrConfig TinyConfig => new()
    {
        Name = "rtdetr-tiny",
        BackboneChannels = [8, 16, 16, 24, 32],
        BackboneRepeat = 1,
        HiddenDim = 32,
        NumHeads = 4,
        NumDecoderLayers = 1,
        NumQueries = 8,
        NumClasses = 4,
        NumLevels = 3,
        NumPoints = 2,
        FeedForwardDim = 64,
    };

    private static RtDetrModel BuildTinyModel()
    {
        RtDetrConfig cfg = TinyConfig;
        RtDetrModel model = new(cfg);
        model.LoadWeights(RtDetrSyntheticWeights.Build(cfg));
        return model;
    }

    private static Tensor MakeInput(int size)
    {
        Tensor input = new(new TensorShape(1, 3, size, size), DType.F32);
        Span<float> span = input.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
            span[i] = ((i * 37) % 255) / 255f;
        return input;
    }

    [Fact]
    public void Config_Presets_ValidateAndCoverAllWeights()
    {
        // Config invariants.
        RtDetrConfig.R50.Validate();
        RtDetrConfig.R18.Validate();
        Assert.Equal(RtDetrConfig.R50.HiddenDim, RtDetrConfig.R50.HeadDim * RtDetrConfig.R50.NumHeads);
        Assert.Equal((512, 1024, 2048), RtDetrConfig.R50.NeckInputChannels);

        // Every enumerated model weight is exactly one loaded synthetic tensor — proves LoadWeights
        // and EnumerateWeights agree on the key set (no missing / extra keys).
        RtDetrConfig cfg = TinyConfig;
        Dictionary<string, Tensor> weights = RtDetrSyntheticWeights.Build(cfg);
        RtDetrModel model = new(cfg);
        model.LoadWeights(weights);
        int enumerated = model.EnumerateWeights().Count();
        Assert.Equal(weights.Count, enumerated);
    }

    [Fact]
    public void BilinearSampleZeroPad_KnownGrid_InterpolatesCorrectly()
    {
        // 2x2 single-channel grid laid [y*W + x]: (0,0)=0 (1,0)=1 (0,1)=2 (1,1)=3.
        float[] grid = [0f, 1f, 2f, 3f];
        Span<float> dst = stackalloc float[1];

        // Center sample = mean of the four corners = 1.5.
        RtDetrDecoder.BilinearSampleZeroPad(grid, tokenBase: 0, height: 2, width: 2, rowStride: 1, channelOffset: 0, channels: 1, px: 0.5f, py: 0.5f, dst);
        Assert.Equal(1.5f, dst[0], 5);

        // Exact grid point (1,1) = 3.
        RtDetrDecoder.BilinearSampleZeroPad(grid, 0, 2, 2, 1, 0, 1, px: 1f, py: 1f, dst);
        Assert.Equal(3f, dst[0], 5);

        // Off the left edge — the only in-bounds neighbour is (0,0) whose value is 0, so the
        // zero-padded sample is 0 regardless of weight.
        RtDetrDecoder.BilinearSampleZeroPad(grid, 0, 2, 2, 1, 0, 1, px: -0.5f, py: 0f, dst);
        Assert.Equal(0f, dst[0], 5);
    }

    [Trait("Category", "SyntheticSmoke")]
    [Fact]
    public void Model_Forward_ProducesExpectedShapes()
    {
        using IBackend backend = new CpuBackend();
        RtDetrModel model = BuildTinyModel();
        using Tensor input = MakeInput(64);

        (Tensor cls, Tensor boxes) = model.Forward(backend, input);
        try
        {
            Assert.Equal(3, cls.Shape.Rank);
            Assert.Equal(1, (int)cls.Shape[0]);
            Assert.Equal(model.NumQueries, (int)cls.Shape[1]);
            Assert.Equal(model.NumClasses, (int)cls.Shape[2]);

            Assert.Equal(3, boxes.Shape.Rank);
            Assert.Equal(model.NumQueries, (int)boxes.Shape[1]);
            Assert.Equal(4, (int)boxes.Shape[2]);

            // Boxes are cxcywh in [0,1] (sigmoid output).
            foreach (float v in boxes.AsSpan<float>())
                Assert.InRange(v, 0f, 1f);
        }
        finally
        {
            cls.Dispose();
            boxes.Dispose();
        }
    }

    [Trait("Category", "SyntheticSmoke")]
    [Fact]
    public void DecoderLayer_HiddenState_HasQueryByHiddenShape()
    {
        using IBackend backend = new CpuBackend();
        RtDetrModel model = BuildTinyModel();
        using Tensor input = MakeInput(64);

        using Tensor hidden = model.ForwardHiddenState(backend, input);
        Assert.Equal(3, hidden.Shape.Rank);
        Assert.Equal(1, (int)hidden.Shape[0]);
        Assert.Equal(model.NumQueries, (int)hidden.Shape[1]);
        Assert.Equal(model.Config.HiddenDim, (int)hidden.Shape[2]);
    }

    [Trait("Category", "SyntheticSmoke")]
    [Fact]
    public void Pipeline_Detect_ReturnsCappedNormalizedResults()
    {
        using IBackend backend = new CpuBackend();
        RtDetrModel model = BuildTinyModel();
        using RtDetrPipeline pipeline = new(backend, model, inputSize: 64);

        const int srcW = 80, srcH = 60;
        byte[] rgb = new byte[srcW * srcH * 3];
        for (int i = 0; i < rgb.Length; i++)
            rgb[i] = (byte)((i * 53) % 256);

        IReadOnlyList<DetectionResult> results = pipeline.Detect(rgb, srcW, srcH, confidenceThreshold: 0f);
        _output.WriteLine($"rtdetr-tiny produced {results.Count} detections (cap {model.NumQueries})");

        Assert.True(results.Count <= model.NumQueries, "NMS-free decoder must emit at most NumQueries detections.");
        for (int i = 1; i < results.Count; i++)
            Assert.True(results[i - 1].Confidence >= results[i].Confidence, "results must be sorted by confidence descending.");
        foreach (DetectionResult d in results)
        {
            Assert.InRange(d.X, 0f, 1f);
            Assert.InRange(d.Y, 0f, 1f);
            Assert.InRange(d.Width, 0f, 1f);
            Assert.InRange(d.Height, 0f, 1f);
            Assert.InRange(d.Confidence, 0f, 1f);
            Assert.InRange(d.ClassIndex, 0, model.NumClasses - 1);
        }
    }

    /// <summary>Real-weight parity — gated on <c>RTDETR_PATH</c>. Skips cleanly when unset; a real
    /// checkpoint + reference boxes are needed to graduate this out of a stub.</summary>
    [Trait("Category", "Integration")]
    [Fact]
    public void RealWeights_Parity_Stub()
    {
        string? path = Environment.GetEnvironmentVariable("RTDETR_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            _output.WriteLine("SKIPPED: RTDETR_PATH unset or file missing.");
            return;
        }

        using IBackend backend = new CpuBackend();
        using RtDetrPipeline pipeline = new(backend, RtDetrConfig.R50, path, inputSize: 640);
        Assert.Equal("rtdetr-r50", pipeline.ModelName);
        // Parity assertions vs a reference detection set are pending real RT-DETR weight conversion.
    }
}
