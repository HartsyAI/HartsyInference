using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines.Stages;
using HartsyInference.ThreeD.Tests.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>The shared texture sampling sequence and the Pixal3D texture stage on tiny on-disk projection flows: the shape latent is concatenated to
/// the sample, the projected features reach every block, a plain TRELLIS.2 flow is refused, and the model-file lookup of the optional texture files.</summary>
public sealed class Pixal3DTextureStageTests : IDisposable
{
    private const int Latent = 4, ProjChannels = 6;
    private const string ProjectionSidecar = """
        {"name": "ElasticSLatFlowModel", "args": {"resolution": 8, "in_channels": 8, "out_channels": 4, "model_channels": 16,
        "cond_channels": 8, "num_blocks": 2, "num_heads": 2, "mlp_ratio": 2.0, "pe_mode": "rope", "share_mod": true,
        "qk_rms_norm": true, "qk_rms_norm_cross": true, "image_attn_mode": "proj", "proj_in_channels": 6}}
        """;
    private const string PlainSidecar = """
        {"name": "ElasticSLatFlowModel", "args": {"resolution": 8, "in_channels": 8, "out_channels": 4, "model_channels": 16,
        "cond_channels": 8, "num_blocks": 2, "num_heads": 2, "mlp_ratio": 2.0, "pe_mode": "rope", "share_mod": true,
        "qk_rms_norm": true, "qk_rms_norm_cross": true}}
        """;

    private static readonly Trellis2SamplerParams Params = new()
    {
        Steps = 2, GuidanceStrength = 3f, GuidanceRescale = 0.5f, IntervalLow = 0.0, IntervalHigh = 1.0, RescaleT = 3.0,
    };

    private readonly string _directory = Directory.CreateTempSubdirectory("pixal3d-texture-stage-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    private string Write(string name, string sidecar)
    {
        Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(sidecar);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 11);
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

    private static SparseTensor Shape(int[] coords)
    {
        Tensor feats = Trellis2TestData.Random(new Random(3), 1, coords.Length / 4, Latent);
        return new SparseTensor(feats, coords, 8);
    }

    private static Func<Trellis2TextureFlow, Tensor, Tensor, float, bool, Tensor> Forward(
        CpuBackend backend, int[] coords, Pixal3DCondition cond, Pixal3DCondition neg)
        => (flow, x, shapeNormalized, t, conditional) =>
        {
            Pixal3DCondition c = conditional ? cond : neg;
            return flow.Forward(backend, x, shapeNormalized, coords, t, c.Global, c.Proj);
        };

    [Fact]
    public void SampleFlow_ProducesTheDenormalizedTextureLatentOverTheShapeVoxelsDeterministically()
    {
        string path = Write("tex", ProjectionSidecar);
        int[] coords = [0, 1, 2, 3, 0, 4, 4, 4, 0, 7, 0, 5];
        float[] zero = [0f, 0f, 0f, 0f], one = [1f, 1f, 1f, 1f], mean = [1f, 2f, 3f, 4f], std = [2f, 2f, 2f, 2f];
        using CpuBackend backend = new();
        using Pixal3DCondition cond = Condition(1), neg = Condition(1, zero: true);
        SparseTensor shape = Shape(coords);
        using Tensor shapeFeats = shape.Feats;
        Func<Trellis2TextureFlow, Tensor, Tensor, float, bool, Tensor> forward = Forward(backend, coords, cond, neg);

        float[] a = Run(5), b = Run(5), other = Run(6);

        Assert.Equal(a, b);
        Assert.NotEqual(a, other);
        Assert.All(a, v => Assert.True(float.IsFinite(v)));

        float[] Run(int seed)
        {
            SparseTensor sampled = Trellis2TextureFlowStage.SampleFlow(
                backend, path, shape, seed, Params, null, zero, one, mean, std, requireProjection: true, [cond.Global, cond.Proj, neg.Global, neg.Proj], forward);
            try
            {
                Assert.Equal(coords, sampled.Coords);
                Assert.Equal(new TensorShape(1, 3, Latent), sampled.Feats.Shape);
                return Trellis2TestData.ToArray(sampled.Feats);
            }
            finally
            {
                sampled.Feats.Dispose();
            }
        }
    }

    [Fact]
    public void SampleFlow_UsesTheProjectedFeaturesOfTheConditioning()
    {
        string path = Write("tex", ProjectionSidecar);
        int[] coords = [0, 1, 2, 3, 0, 4, 4, 4, 0, 7, 0, 5];
        float[] zero = [0f, 0f, 0f, 0f], one = [1f, 1f, 1f, 1f];
        using CpuBackend backend = new();
        using Pixal3DCondition cond = Condition(1), neg = Condition(1, zero: true), changed = Condition(9);
        SparseTensor shape = Shape(coords);
        using Tensor shapeFeats = shape.Feats;

        float[] with = Sample(cond), without = Sample(changed);

        Assert.NotEqual(with, without);

        float[] Sample(Pixal3DCondition c)
        {
            SparseTensor sampled = Trellis2TextureFlowStage.SampleFlow(
                backend, path, shape, 1, Params, null, zero, one, zero, one, true, [c.Global, c.Proj], Forward(backend, coords, c, neg));
            try
            {
                return Trellis2TestData.ToArray(sampled.Feats);
            }
            finally
            {
                sampled.Feats.Dispose();
            }
        }
    }

    [Fact]
    public void SampleFlow_RefusesAFlowOfTheOtherKindBeforeAnyStepRuns()
    {
        string projection = Write("proj", ProjectionSidecar), plain = Write("plain", PlainSidecar);
        int[] coords = [0, 1, 2, 3];
        float[] zero = [0f, 0f, 0f, 0f], one = [1f, 1f, 1f, 1f];
        using CpuBackend backend = new();
        using Pixal3DCondition cond = Condition(1), neg = Condition(1, zero: true);
        SparseTensor shape = Shape(coords);
        using Tensor shapeFeats = shape.Feats;
        int steps = 0;

        Assert.Throws<HartsyInferenceException>(() => Trellis2TextureFlowStage.SampleFlow(
            backend, plain, shape, 1, Params, _ => steps++, zero, one, zero, one, true, [], Forward(backend, coords, cond, neg)));
        Assert.Throws<HartsyInferenceException>(() => Trellis2TextureFlowStage.SampleFlow(
            backend, projection, shape, 1, Params, _ => steps++, zero, one, zero, one, false, [], Forward(backend, coords, cond, neg)));
        Assert.Equal(0, steps);
    }

    [Fact]
    public void SampleFlow_RejectsAShapeLatentOfTheWrongWidth()
    {
        string path = Write("tex", ProjectionSidecar);
        using CpuBackend backend = new();
        using Pixal3DCondition cond = Condition(1), neg = Condition(1, zero: true);
        int[] coords = [0, 1, 2, 3];
        SparseTensor wrong = new(Trellis2TestData.Random(new Random(1), 1, 1, 3), coords, 8);
        using Tensor wrongFeats = wrong.Feats;

        Assert.Throws<ArgumentException>(() => Trellis2TextureFlowStage.SampleFlow(
            backend, path, wrong, 1, Params, null, new float[Latent], new float[Latent], new float[Latent], new float[Latent], true, [],
            Forward(backend, coords, cond, neg)));
    }

    [Fact]
    public void TextureFlowStage_RequiresATexturePipelineConfig()
    {
        Trellis2PipelineConfig shapeOnly = Trellis2PipelineConfig.FromJson("""
            {"name": "Trellis2ImageTo3DPipeline", "args": {
              "sparse_structure_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
                "params": {"steps": 3, "guidance_strength": 7.5, "guidance_rescale": 0.7, "guidance_interval": [0.6, 1.0], "rescale_t": 5.0}},
              "shape_slat_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
                "params": {"steps": 2, "guidance_strength": 7.5, "guidance_rescale": 0.5, "guidance_interval": [0.6, 1.0], "rescale_t": 3.0}},
              "shape_slat_normalization": {"mean": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "std": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]} } }
            """);

        Assert.Throws<HartsyInferenceException>(() => new Pixal3DTextureFlowStage(Path.Combine(_directory, "x.safetensors"), shapeOnly));
    }

    [Fact]
    public void ModelFiles_ReportTextureOnlyWhenTheFlowAndTheDecoderAreBothInstalled()
    {
        string root = Directory.CreateTempSubdirectory("pixal3d-texture-files-").FullName;
        string pixal = Path.Combine(root, "3D", "pixal3d"), trellis2 = Path.Combine(root, "3D", "trellis2");
        Directory.CreateDirectory(pixal);
        Directory.CreateDirectory(trellis2);
        try
        {
            foreach (string name in new[]
            {
                Trellis2ModelFiles.PipelineFile, Trellis2ModelFiles.StructureFlowFile, Trellis2ModelFiles.StructureDecoderFile,
                Trellis2ModelFiles.ShapeFlow512File, Trellis2ModelFiles.ShapeFlow1024File, Trellis2ModelFiles.ShapeDecoderFile, Pixal3DModelFiles.NafFile,
            })
                File.WriteAllBytes(Path.Combine(pixal, name), [0]);
            Directory.CreateDirectory(Path.Combine(root, "DINOv3"));
            File.WriteAllBytes(Path.Combine(root, "DINOv3", Trellis2ModelFiles.DinoFile), [0]);

            Pixal3DModelFiles shapeOnly = Pixal3DModelFiles.Resolve(pixal);
            Assert.False(shapeOnly.HasTexture);
            Assert.Null(shapeOnly.TextureFlow);
            Assert.Null(shapeOnly.TextureDecoder);

            File.WriteAllBytes(Path.Combine(pixal, Trellis2ModelFiles.TextureFlow1024File), [0]);
            Assert.False(Pixal3DModelFiles.Resolve(pixal).HasTexture);

            File.WriteAllBytes(Path.Combine(trellis2, Trellis2ModelFiles.TextureDecoderFile), [0]);
            Pixal3DModelFiles textured = Pixal3DModelFiles.Resolve(pixal);
            Assert.True(textured.HasTexture);
            Assert.EndsWith(Trellis2ModelFiles.TextureFlow1024File, textured.TextureFlow, StringComparison.Ordinal);
            Assert.Equal(Path.Combine(trellis2, Trellis2ModelFiles.TextureDecoderFile), textured.TextureDecoder);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
