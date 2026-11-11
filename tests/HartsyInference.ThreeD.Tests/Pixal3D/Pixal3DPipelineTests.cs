using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.ThreeD.Pipelines.Stages;
using HartsyInference.ThreeD.Tests.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>Orchestration of the Pixal3D pipeline over stub stages: stage order and conditioning per stage, the rounding quantizer, request validation
/// before any stage runs, seeds, camera and step overrides, progress, cancellation with tensor cleanup, and disposal.</summary>
public sealed class Pixal3DPipelineTests
{
    private static readonly Trellis2PipelineConfig Config = Trellis2PipelineConfig.FromJson(ConfigJson);

    private const string ConfigJson = """
        {"name": "Trellis2ImageTo3DPipeline", "args": {
          "sparse_structure_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
            "params": {"steps": 3, "guidance_strength": 7.5, "guidance_rescale": 0.7, "guidance_interval": [0.6, 1.0], "rescale_t": 5.0}},
          "shape_slat_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
            "params": {"steps": 2, "guidance_strength": 7.5, "guidance_rescale": 0.5, "guidance_interval": [0.6, 1.0], "rescale_t": 3.0}},
          "shape_slat_normalization": {"mean": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            "std": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]}}}
        """;

    // Foreground x, y 20..59 in a 100 px image: extent 39, so Pixal3D's int(39 * 1.1) = 42 px crop where TRELLIS.2's int(39 * 1) crops 38.
    private static Trellis2InputImage Image() => new()
    {
        Rgb = new byte[100 * 100 * 3],
        Alpha = Enumerable.Range(0, 100 * 100).Select(i => i % 100 is >= 20 and < 60 && i / 100 is >= 20 and < 60 ? (byte)255 : (byte)0).ToArray(),
        Width = 100,
        Height = 100,
    };

    [Fact]
    public void Generate_RunsTheCascadeWithPixal3DsConditioningPerStage()
    {
        using Rig rig = new();

        Mesh mesh = rig.Pipeline.Generate(Image(), new Pixal3DOptions { Seed = 100 });

        Assert.Equal(
            [
                "cond image=512 grid=16 dense naf=0 fov=0.8575560450553894 prepared=42",
                "structure@32 steps=3 seed=100",
                "cond image=512 grid=32 voxels=2 naf=512 fov=0.8575560450553894 prepared=42",
                "shape Res512 grid=32 seed=101 steps=2",
                "upsample x4",
                "cond image=1024 grid=64 voxels=3 naf=512 fov=0.8575560450553894 prepared=42",
                "shape Res1024 grid=64 seed=102 steps=2",
                "decode",
            ],
            rig.Calls);
        Assert.Equal(2, mesh.TriangleCount);
        Assert.NotNull(mesh.Normals);
    }

    [Fact]
    public void Generate_Grid1536UsesTheNinetySixGridWhenTheTokenCapAllowsIt()
    {
        using Rig rig = new();

        rig.Pipeline.Generate(Image(), new Pixal3DOptions { Grid = 1536, Seed = 1 });

        Assert.Contains("cond image=1024 grid=96 voxels=3 naf=512 fov=0.8575560450553894 prepared=42", rig.Calls);
        Assert.Contains("shape Res1024 grid=96 seed=3 steps=2", rig.Calls);
    }

    [Fact]
    public void Generate_Grid1536RetreatsToTheFloorWhenTheTokenCapIsTight()
    {
        using Rig rig = new(maxTokens: 1);

        Mesh mesh = rig.Pipeline.Generate(Image(), new Pixal3DOptions { Grid = 1536, Seed = 1 });

        Assert.Contains("shape Res1024 grid=64 seed=3 steps=2", rig.Calls);
        // The first voxel's dual vertex sits half a voxel in from the origin: x = -0.5 + 0.5 / 1024 at the floor resolution.
        Assert.Equal(-0.5f + 0.5f / 1024f, mesh.Vertices[0], 6);
    }

    [Fact]
    public void Generate_UsesTheRoundingQuantizerForTheFineVoxels()
    {
        using Rig rig = new();

        rig.Pipeline.Generate(Image(), new Pixal3DOptions { Seed = 1 });

        // The stub upsampler returns voxels 16, 17, 40 and 199 on the 512 grid. With 64 cells round((c + 0.5) / 512 * 63) gives 2, 2, 5 and 25; TRELLIS.2's
        // floor((c + 0.5) / 512 * 64) would give 24 for the last one, so only the rounding quantizer yields this set.
        Assert.Equal([0, 2, 2, 2, 0, 5, 5, 5, 0, 25, 25, 25], rig.FineCoords!);
    }

    [Fact]
    public void Generate_PassesTheCameraToEveryConditioning()
    {
        using Rig rig = new();

        rig.Pipeline.Generate(Image(), new Pixal3DOptions { Seed = 1, FovRadians = 0.5, MeshScale = 1.25 });

        Assert.All(rig.Calls.Where(c => c.StartsWith("cond", StringComparison.Ordinal)), c => Assert.Contains("fov=0.5", c, StringComparison.Ordinal));
        Assert.Equal([1.25, 1.25, 1.25], rig.Scales);
    }

    [Fact]
    public void Generate_StepsOverrideBothSamplers()
    {
        using Rig rig = new();

        rig.Pipeline.Generate(Image(), new Pixal3DOptions { Steps = 5, Seed = 1 });

        Assert.Contains("structure@32 steps=5 seed=1", rig.Calls);
        Assert.Contains("shape Res512 grid=32 seed=2 steps=5", rig.Calls);
    }

    [Fact]
    public void Generate_DrawsASeedWhenNoneIsGiven()
    {
        using Rig rig = new();

        rig.Pipeline.Generate(Image(), new Pixal3DOptions());

        Assert.Matches(@"^structure@32 steps=3 seed=\d+$", rig.Calls[1]);
    }

    [Theory]
    [InlineData(512)]
    [InlineData(700)]
    [InlineData(-1024)]
    [InlineData(2048)]
    public void Generate_RejectsAGridThatIsNot1024Or1536BeforeAnyStageRuns(int grid)
    {
        using Rig rig = new();

        Assert.Throws<ArgumentException>(() => rig.Pipeline.Generate(Image(), new Pixal3DOptions { Grid = grid }));

        Assert.Empty(rig.Calls);
    }

    [Fact]
    public void Generate_RejectsBadSeedStepsAndCameraBeforeAnyStageRuns()
    {
        using Rig rig = new();

        Assert.Throws<ArgumentException>(() => rig.Pipeline.Generate(Image(), new Pixal3DOptions { Seed = -1 }));
        Assert.Throws<ArgumentException>(() => rig.Pipeline.Generate(Image(), new Pixal3DOptions { Steps = -1 }));
        Assert.Throws<ArgumentException>(() => rig.Pipeline.Generate(Image(), new Pixal3DOptions { Steps = Pixal3DImageTo3DPipeline.MaxSteps + 1 }));
        Assert.Throws<ArgumentException>(() => rig.Pipeline.Generate(Image(), new Pixal3DOptions { FovRadians = 0 }));
        Assert.Throws<ArgumentException>(() => rig.Pipeline.Generate(Image(), new Pixal3DOptions { FovRadians = 1.6 }));
        Assert.Throws<ArgumentException>(() => rig.Pipeline.Generate(Image(), new Pixal3DOptions { MeshScale = 0 }));

        Assert.Empty(rig.Calls);
    }

    [Fact]
    public void Generate_AnImageWithoutAlphaNeedsAMatterAndSaysSoBeforeAnyStageRuns()
    {
        using Rig rig = new();
        Trellis2InputImage opaque = new() { Rgb = new byte[8 * 8 * 3], Width = 8, Height = 8 };

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => rig.Pipeline.Generate(opaque));

        Assert.Contains("alpha", error.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(rig.Calls);
    }

    [Fact]
    public void ResolveGrid_DefaultsTo1024AndKnowsOnlyTheTwoCascades()
    {
        Assert.Equal(1024, Pixal3DImageTo3DPipeline.ResolveGrid(0));
        Assert.Equal(1024, Pixal3DImageTo3DPipeline.ResolveGrid(1024));
        Assert.Equal(1536, Pixal3DImageTo3DPipeline.ResolveGrid(1536));
        Assert.Throws<ArgumentException>(() => Pixal3DImageTo3DPipeline.ResolveGrid(512));
    }

    [Fact]
    public void Generate_ReportsMonotonicProgressEndingAtTheTotal()
    {
        using Rig rig = new();
        List<GenerationProgress> reports = [];

        rig.Pipeline.Generate(Image(), new Pixal3DOptions { Seed = 1 }, reports.Add);

        // 3 conditionings + upsample + decode + mesh = 6 stage steps, plus 3 structure and 2 + 2 shape sampler steps = 13.
        Assert.All(reports, p => Assert.Equal(13, p.TotalSteps));
        Assert.Equal(Enumerable.Range(1, 13), reports.Select(p => p.Step));
    }

    // A cancel raised from the progress callback at any step must stop the run and release every tensor a stage handed out.
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(8)]
    [InlineData(10)]
    [InlineData(12)]
    public void Generate_PropagatesACancelAndReleasesEveryTensor(int cancelAtStep)
    {
        using Rig rig = new();

        Assert.Throws<OperationCanceledException>(() => rig.Pipeline.Generate(Image(), new Pixal3DOptions { Seed = 1 }, p =>
        {
            if (p.Step == cancelAtStep)
                throw new OperationCanceledException();
        }));

        Assert.NotEmpty(rig.Tensors);
        foreach (Tensor t in rig.Tensors)
            Trellis2TestData.AssertDisposed(t);
    }

    [Fact]
    public void Generate_ReleasesEveryConditioningAfterASuccessfulRun()
    {
        using Rig rig = new();

        rig.Pipeline.Generate(Image(), new Pixal3DOptions { Seed = 1 });

        foreach (Tensor t in rig.Tensors)
            Trellis2TestData.AssertDisposed(t);
    }

    [Fact]
    public void Generate_ThrowsWhenTheDecodedShapeHasNoSurface()
    {
        using Rig rig = new(emptyField: true);

        Assert.Throws<HartsyInferenceException>(() => rig.Pipeline.Generate(Image(), new Pixal3DOptions { Seed = 1 }));
    }

    [Fact]
    public void Dispose_IsIdempotentReleasesOwnedStagesAndBlocksFurtherUse()
    {
        Rig rig = new();
        rig.Pipeline.Dispose();
        rig.Pipeline.Dispose();

        Assert.Equal(1, rig.Owned.Disposals);
        Assert.Throws<ObjectDisposedException>(() => rig.Pipeline.Generate(Image()));
        rig.Dispose();
    }

    private sealed class Rig : IDisposable
    {
        internal Rig(int maxTokens = Trellis2Cascade.DefaultMaxTokens, bool emptyField = false)
        {
            Backend = new CpuBackend();
            Owned = new OwnedStub();
            Pipeline = new Pixal3DImageTo3DPipeline(
                Backend,
                Config,
                null,
                new StubConditioner(this),
                new StubStructure(this),
                new StubShape(this),
                new StubUpsampler(this),
                new StubDecoder(this, emptyField),
                new DualGridMesherStage(),
                [Owned],
                maxTokens);
        }

        internal CpuBackend Backend { get; }

        internal OwnedStub Owned { get; }

        internal Pixal3DImageTo3DPipeline Pipeline { get; }

        internal List<string> Calls { get; } = [];

        internal List<Tensor> Tensors { get; } = [];

        internal List<double> Scales { get; } = [];

        internal int[]? FineCoords { get; set; }

        public void Dispose()
        {
            Pipeline.Dispose();
            Backend.Dispose();
        }
    }

    private sealed class OwnedStub : IDisposable
    {
        internal int Disposals { get; private set; }

        public void Dispose() => Disposals++;
    }

    private sealed class StubConditioner(Rig rig) : IPixal3DConditioner
    {
        public Pixal3DCondition Encode(IBackend backend, Trellis2PreparedImage image, Pixal3DConditionSpec spec, Pixal3DView view)
        {
            string where = spec.Coords is null ? "dense" : $"voxels={spec.Coords.Length / 4}";
            rig.Calls.Add($"cond image={spec.ImageSize} grid={spec.GridResolution} {where} naf={spec.NafSize} fov={view.FovRadians} prepared={image.Width}");
            rig.Scales.Add(view.MeshScale);
            Tensor global = new(new TensorShape(1, 5, 4), DType.F32);
            Tensor proj = new(new TensorShape(1, 3, 4), DType.F32);
            rig.Tensors.Add(global);
            rig.Tensors.Add(proj);
            return new Pixal3DCondition(global, proj);
        }
    }

    private sealed class StubStructure(Rig rig) : IPixal3DStructureStage
    {
        public int[] Sample(
            IBackend backend, Pixal3DCondition cond, Pixal3DCondition negCond, int seed, Trellis2SamplerParams parameters, int outputResolution,
            Action<int>? onStep)
        {
            rig.Calls.Add($"structure@{outputResolution} steps={parameters.Steps} seed={seed}");
            rig.Tensors.Add(negCond.Global);
            rig.Tensors.Add(negCond.Proj);
            for (int s = 1; s <= parameters.Steps; s++)
                onStep?.Invoke(s);
            return [0, 1, 1, 1, 0, 2, 2, 2];
        }
    }

    private sealed class StubShape(Rig rig) : IPixal3DShapeFlowStage
    {
        public SparseTensor Sample(
            IBackend backend, Trellis2ShapeFlowKind kind, int[] coords, int gridResolution, Pixal3DCondition cond, Pixal3DCondition negCond,
            int seed, Trellis2SamplerParams parameters, Action<int>? onStep)
        {
            rig.Calls.Add($"shape {kind} grid={gridResolution} seed={seed} steps={parameters.Steps}");
            rig.Tensors.Add(negCond.Global);
            rig.Tensors.Add(negCond.Proj);
            if (kind == Trellis2ShapeFlowKind.Res1024)
                rig.FineCoords = coords;
            for (int s = 1; s <= parameters.Steps; s++)
                onStep?.Invoke(s);
            Tensor latent = new(new TensorShape(1, coords.Length / 4, 32), DType.F32);
            rig.Tensors.Add(latent);
            return new SparseTensor(latent, coords, gridResolution);
        }
    }

    private sealed class StubUpsampler(Rig rig) : ISparseShapeUpsampler
    {
        public int[] Upsample(IBackend backend, SparseTensor latent, int times)
        {
            rig.Calls.Add($"upsample x{times}");
            return [0, 16, 16, 16, 0, 17, 16, 16, 0, 40, 40, 40, 0, 199, 199, 199];
        }
    }

    private sealed class StubDecoder(Rig rig, bool emptyField) : ISparseShapeDecoder
    {
        public DualGridField Decode(IBackend backend, SparseTensor latent)
        {
            rig.Calls.Add("decode");
            if (emptyField)
                return new DualGridField(new DualGridVoxels([0, 0, 0], [0.5f, 0.5f, 0.5f], new bool[3]), [1f]);
            // Four voxels around one x-axis edge; the first flags it, so the mesher emits one quad (two triangles).
            int[] coords = [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0];
            float[] dual = new float[12];
            for (int i = 0; i < 12; i++)
                dual[i] = coords[i] + 0.5f;
            bool[] intersected = new bool[12];
            intersected[0] = true;
            return new DualGridField(new DualGridVoxels(coords, dual, intersected), [1f, 1f, 1f, 1f]);
        }
    }
}
