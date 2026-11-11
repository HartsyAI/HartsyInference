using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines.Stages;
using HartsyInference.ThreeD.Tests.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>The sampling sequence both shape stages share, and the Pixal3D stage's refusal of a plain TRELLIS.2 flow, on tiny on-disk checkpoints.</summary>
public sealed class Pixal3DShapeFlowStageTests : IDisposable
{
    private const int ProjChannels = 6;
    private const string ProjectionSidecar = """
        {"name": "ElasticSLatFlowModel", "args": {"resolution": 8, "in_channels": 4, "out_channels": 4, "model_channels": 16,
        "cond_channels": 8, "num_blocks": 2, "num_heads": 2, "mlp_ratio": 2.0, "pe_mode": "rope", "share_mod": true,
        "qk_rms_norm": true, "qk_rms_norm_cross": true, "image_attn_mode": "proj", "proj_in_channels": 6}}
        """;
    private const string PlainSidecar = """
        {"name": "SLatFlowModel", "args": {"resolution": 8, "in_channels": 4, "out_channels": 4, "model_channels": 16,
        "cond_channels": 8, "num_blocks": 2, "num_heads": 2, "mlp_ratio": 2.0, "pe_mode": "rope", "share_mod": true,
        "qk_rms_norm": true, "qk_rms_norm_cross": true}}
        """;

    private static readonly Trellis2SamplerParams Params = new()
    {
        Steps = 2, GuidanceStrength = 3f, GuidanceRescale = 0.5f, IntervalLow = 0.0, IntervalHigh = 1.0, RescaleT = 3.0,
    };

    private readonly string _directory = Directory.CreateTempSubdirectory("pixal3d-shape-stage-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    private string Write(string name, string sidecar)
    {
        Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(sidecar);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 7);
        string path = Path.Combine(_directory, name + ".safetensors");
        SafeTensorsWriter.Save(path, tensors);
        File.WriteAllText(Path.ChangeExtension(path, ".json"), sidecar);
        Trellis2TestData.DisposeAll(tensors.Values);
        return path;
    }

    private static Pixal3DCondition Condition(int seed, bool zero = false)
    {
        Random rng = new(seed);
        Tensor global = zero ? new Tensor(new TensorShape(1, 5, 8), DType.F32) : Trellis2TestData.Random(rng, 1, 5, 8);
        Tensor proj = zero ? new Tensor(new TensorShape(1, 3, ProjChannels), DType.F32) : Trellis2TestData.Random(rng, 1, 3, ProjChannels);
        return new Pixal3DCondition(global, proj);
    }

    [Fact]
    public void Sample_ProducesTheDenormalizedLatentOverTheVoxelsDeterministically()
    {
        string path = Write("flow", ProjectionSidecar);
        int[] coords = [0, 1, 2, 3, 0, 4, 4, 4, 0, 7, 0, 5];
        float[] mean = [1f, 2f, 3f, 4f], std = [2f, 2f, 2f, 2f];
        using CpuBackend backend = new();
        using Pixal3DCondition cond = Condition(1), neg = Condition(1, zero: true);
        Func<Trellis2SlatFlow, Tensor, float, bool, Tensor> forward = (flow, x, t, conditional) =>
        {
            Pixal3DCondition c = conditional ? cond : neg;
            return flow.Forward(backend, x, coords, t, c.Global, c.Proj);
        };

        (TensorShape shape, float[] a) = Run(5);
        (_, float[] b) = Run(5);
        (_, float[] other) = Run(6);

        Assert.Equal(new TensorShape(1, 3, 4), shape);
        Assert.Equal(a, b);
        Assert.NotEqual(a, other);
        Assert.All(a, v => Assert.True(float.IsFinite(v)));

        (TensorShape, float[]) Run(int seed)
        {
            SparseTensor sampled = Trellis2ShapeFlowStage.SampleFlow(
                backend, path, coords, 8, seed, Params, null, mean, std, requireProjection: true, forward);
            try
            {
                Assert.Equal(coords, sampled.Coords);
                return (sampled.Feats.Shape, Trellis2TestData.ToArray(sampled.Feats));
            }
            finally
            {
                sampled.Feats.Dispose();
            }
        }
    }

    [Fact]
    public void Sample_ReportsEveryStepAndStopsOnACancelReleasingTheLatent()
    {
        string path = Write("flow", ProjectionSidecar);
        int[] coords = [0, 1, 2, 3, 0, 4, 4, 4, 0, 7, 0, 5];
        float[] mean = [0f, 0f, 0f, 0f], std = [1f, 1f, 1f, 1f];
        using CpuBackend backend = new();
        using Pixal3DCondition cond = Condition(2), neg = Condition(2, zero: true);
        List<int> steps = [];
        Func<Trellis2SlatFlow, Tensor, float, bool, Tensor> forward = (flow, x, t, conditional) =>
        {
            Pixal3DCondition c = conditional ? cond : neg;
            return flow.Forward(backend, x, coords, t, c.Global, c.Proj);
        };

        SparseTensor sampled = Trellis2ShapeFlowStage.SampleFlow(backend, path, coords, 8, 1, Params, steps.Add, mean, std, true, forward);
        sampled.Feats.Dispose();
        Assert.Equal([1, 2], steps);

        Assert.Throws<OperationCanceledException>(() => Trellis2ShapeFlowStage.SampleFlow(
            backend, path, coords, 8, 1, Params, s => throw new OperationCanceledException(), mean, std, true, forward));
    }

    [Fact]
    public void SampleFlow_RefusesAFlowOfTheOtherModelBeforeAnyStepRuns()
    {
        string projection = Write("projection", ProjectionSidecar), plain = Write("plain", PlainSidecar);
        int[] coords = [0, 1, 2, 3];
        using CpuBackend backend = new();
        int calls = 0;
        Func<Trellis2SlatFlow, Tensor, float, bool, Tensor> forward = (_, _, _, _) =>
        {
            calls++;
            throw new InvalidOperationException("must not run");
        };

        Assert.Throws<InvalidOperationException>(
            () => Trellis2ShapeFlowStage.SampleFlow(backend, plain, coords, 8, 1, Params, null, new float[4], new float[4], true, forward));
        Assert.Throws<InvalidOperationException>(
            () => Trellis2ShapeFlowStage.SampleFlow(backend, projection, coords, 8, 1, Params, null, new float[4], new float[4], false, forward));
        Assert.Equal(0, calls);
    }

    [Fact]
    public void Pixal3DStage_RejectsAPlainFlowCheckpoint()
    {
        string plain = Write("plain", PlainSidecar);
        Trellis2PipelineConfig config = Trellis2PipelineConfig.FromJson(
            """
            {"name": "Trellis2ImageTo3DPipeline", "args": {
              "sparse_structure_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
                "params": {"steps": 1, "guidance_strength": 1.0, "guidance_rescale": 0.0, "guidance_interval": [0.6, 1.0], "rescale_t": 5.0}},
              "shape_slat_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
                "params": {"steps": 1, "guidance_strength": 1.0, "guidance_rescale": 0.0, "guidance_interval": [0.6, 1.0], "rescale_t": 3.0}},
              "shape_slat_normalization": {"mean": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "std": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]}}}
            """);
        Pixal3DShapeFlowStage stage = new(plain, plain, config);
        using CpuBackend backend = new();
        using Pixal3DCondition cond = Condition(3), neg = Condition(3, zero: true);

        Assert.Throws<InvalidOperationException>(() => stage.Sample(backend, Trellis2ShapeFlowKind.Res512, [0, 1, 2, 3], 32, cond, neg, 1, Params, null));
    }
}
