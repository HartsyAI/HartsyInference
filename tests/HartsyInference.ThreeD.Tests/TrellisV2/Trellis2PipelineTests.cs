using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.ThreeD.Pipelines.Stages;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>Orchestration of the TRELLIS.2 pipeline over stub stages: stage order per pipeline type, request validation before any
/// stage runs, seeds and step overrides, progress, cancellation with tensor cleanup, and disposal.</summary>
public sealed class Trellis2PipelineTests
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

    private static Trellis2InputImage Image() => new()
    {
        Rgb = new byte[8 * 8 * 3],
        Alpha = Enumerable.Range(0, 64).Select(i => i is >= 18 and <= 45 ? (byte)255 : (byte)0).ToArray(),
        Width = 8,
        Height = 8,
    };

    [Fact]
    public void Generate_DefaultGridRunsTheCascadeInOrder()
    {
        using Rig rig = new();

        Mesh mesh = rig.Pipeline.Generate(Image(), new Trellis2Options { Seed = 100 });

        Assert.Equal(
            [
                "cond@512", "structure@32 steps=3 seed=100", "shape Res512 grid=32 seed=101 steps=2", "upsample x4",
                "cond@1024", "shape Res1024 grid=64 seed=102 steps=2", "decode",
            ],
            rig.Calls);
        Assert.Equal(2, mesh.TriangleCount);
        Assert.NotNull(mesh.Normals);
    }

    [Fact]
    public void Generate_Grid512RunsOneShapeFlowAndNoUpsample()
    {
        using Rig rig = new();

        rig.Pipeline.Generate(Image(), new Trellis2Options { Grid = 512, Seed = 1 });

        Assert.Equal(["cond@512", "structure@32 steps=3 seed=1", "shape Res512 grid=32 seed=2 steps=2", "decode"], rig.Calls);
    }

    [Fact]
    public void Generate_Grid1536UsesTheNinetySixGridWhenTheTokenCapAllowsIt()
    {
        using Rig rig = new();

        rig.Pipeline.Generate(Image(), new Trellis2Options { Grid = 1536, Seed = 1 });

        Assert.Contains("shape Res1024 grid=96 seed=3 steps=2", rig.Calls);
    }

    [Fact]
    public void Generate_Grid1536RetreatsToTheFloorWhenTheTokenCapIsTight()
    {
        using Rig rig = new(maxTokens: 1);

        Mesh mesh = rig.Pipeline.Generate(Image(), new Trellis2Options { Grid = 1536, Seed = 1 });

        Assert.Contains("shape Res1024 grid=64 seed=3 steps=2", rig.Calls);
        // The first voxel's dual vertex sits half a voxel in from the origin: x = -0.5 + 0.5 / 1024 at the floor resolution.
        Assert.Equal(-0.5f + 0.5f / 1024f, mesh.Vertices[0], 6);
    }

    [Fact]
    public void Generate_StepsOverrideBothSamplers()
    {
        using Rig rig = new();

        rig.Pipeline.Generate(Image(), new Trellis2Options { Grid = 512, Steps = 5, Seed = 1 });

        Assert.Equal("structure@32 steps=5 seed=1", rig.Calls[1]);
        Assert.Equal("shape Res512 grid=32 seed=2 steps=5", rig.Calls[2]);
    }

    [Fact]
    public void Generate_DrawsASeedWhenNoneIsGiven()
    {
        using Rig rig = new();

        rig.Pipeline.Generate(Image(), new Trellis2Options { Grid = 512 });

        Assert.Matches(@"^structure@32 steps=3 seed=\d+$", rig.Calls[1]);
    }

    [Theory]
    [InlineData(700)]
    [InlineData(-512)]
    [InlineData(2048)]
    public void Generate_RejectsAGridThatIsNot512Or1024Or1536BeforeAnyStageRuns(int grid)
    {
        using Rig rig = new();

        Assert.Throws<ArgumentException>(() => rig.Pipeline.Generate(Image(), new Trellis2Options { Grid = grid }));

        Assert.Empty(rig.Calls);
    }

    [Fact]
    public void Generate_RejectsANegativeSeedAndBadStepsBeforeAnyStageRuns()
    {
        using Rig rig = new();

        Assert.Throws<ArgumentException>(() => rig.Pipeline.Generate(Image(), new Trellis2Options { Seed = -1 }));
        Assert.Throws<ArgumentException>(() => rig.Pipeline.Generate(Image(), new Trellis2Options { Steps = -1 }));
        int tooMany = Trellis2ImageTo3DPipeline.MaxSteps + 1;
        Assert.Throws<ArgumentException>(() => rig.Pipeline.Generate(Image(), new Trellis2Options { Steps = tooMany }));

        Assert.Empty(rig.Calls);
    }

    [Fact]
    public void Generate_RejectsABrokenImageBeforeAnyStageRuns()
    {
        using Rig rig = new();
        Trellis2InputImage broken = new() { Rgb = new byte[10], Alpha = new byte[64], Width = 8, Height = 8 };

        Assert.Throws<ArgumentException>(() => rig.Pipeline.Generate(broken));

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
    public void ValidateRequest_ResolvesTheGridToUpstreamPipelineTypes()
    {
        Assert.Equal(Trellis2PipelineType.Cascade1024, Trellis2ImageTo3DPipeline.ResolveType(0));
        Assert.Equal(Trellis2PipelineType.Res512, Trellis2ImageTo3DPipeline.ResolveType(512));
        Assert.Equal(Trellis2PipelineType.Cascade1024, Trellis2ImageTo3DPipeline.ResolveType(1024));
        Assert.Equal(Trellis2PipelineType.Cascade1536, Trellis2ImageTo3DPipeline.ResolveType(1536));
        Assert.Throws<ArgumentException>(() => Trellis2ImageTo3DPipeline.ResolveType(256));
    }

    [Fact]
    public void ValidateRequest_AcceptsAnOpaqueImageWhenAMatterExists()
    {
        Trellis2InputImage opaque = new() { Rgb = new byte[12], Width = 2, Height = 2 };

        Trellis2ImageTo3DPipeline.ValidateRequest(new Trellis2Options(), opaque, canMatte: true);

        Assert.Throws<HartsyInferenceException>(() => Trellis2ImageTo3DPipeline.ValidateRequest(new Trellis2Options(), opaque, canMatte: false));
    }

    [Fact]
    public void Generate_ReportsMonotonicProgressEndingAtTheTotal()
    {
        using Rig rig = new();
        List<GenerationProgress> reports = [];

        rig.Pipeline.Generate(Image(), new Trellis2Options { Seed = 1 }, reports.Add);

        // Cascade with 3 structure steps and 2 shape steps per flow: 1 + 3 + 2 + 1 (upsample) + 2 + 1 (decode) + 1 (mesh) = 11.
        Assert.All(reports, p => Assert.Equal(11, p.TotalSteps));
        Assert.Equal(Enumerable.Range(1, 11), reports.Select(p => p.Step));
        Assert.Equal(reports[^1].TotalSteps, reports[^1].Step);
    }

    [Fact]
    public void Generate_ReportsTheShorterTotalOfThe512Type()
    {
        using Rig rig = new();
        List<GenerationProgress> reports = [];

        rig.Pipeline.Generate(Image(), new Trellis2Options { Grid = 512, Seed = 1 }, reports.Add);

        // 1 + 3 + 2 + 2 = 8.
        Assert.Equal(Enumerable.Range(1, 8), reports.Select(p => p.Step));
        Assert.All(reports, p => Assert.Equal(8, p.TotalSteps));
    }

    // A cancel raised from the progress callback at any step must stop the run and release every tensor a stage handed out.
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(10)]
    public void Generate_PropagatesACancelAndReleasesEveryTensor(int cancelAtStep)
    {
        using Rig rig = new();

        Assert.Throws<OperationCanceledException>(() => rig.Pipeline.Generate(Image(), new Trellis2Options { Seed = 1 }, p =>
        {
            if (p.Step == cancelAtStep)
                throw new OperationCanceledException();
        }));

        Assert.NotEmpty(rig.Tensors);
        foreach (Tensor t in rig.Tensors)
            Trellis2TestData.AssertDisposed(t);
    }

    [Fact]
    public void Generate_ReleasesTheConditioningAfterASuccessfulRun()
    {
        using Rig rig = new();

        rig.Pipeline.Generate(Image(), new Trellis2Options { Grid = 512, Seed = 1 });

        foreach (Tensor t in rig.Tensors)
            Trellis2TestData.AssertDisposed(t);
    }

    [Fact]
    public void Generate_ThrowsWhenTheDecodedShapeHasNoSurface()
    {
        using Rig rig = new(emptyField: true);

        Assert.Throws<HartsyInferenceException>(() => rig.Pipeline.Generate(Image(), new Trellis2Options { Grid = 512, Seed = 1 }));
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

    [Fact]
    public void ToYUp_RotatesPositionsAndNormalsFromZUp()
    {
        Mesh mesh = new() { Vertices = [1f, 2f, 3f, -4f, 5f, -6f], Indices = [0, 1, 0], Normals = [0f, 0f, 1f, 0f, 1f, 0f] };

        Trellis2ImageTo3DPipeline.ToYUp(mesh);

        Assert.Equal([1f, 3f, -2f, -4f, -6f, -5f], mesh.Vertices);
        Assert.Equal([0f, 1f, 0f, 0f, 0f, -1f], mesh.Normals);
    }

    [Fact]
    public void Output_IsInTheUnitCubeCentredOnTheOriginAfterTheYUpRotation()
    {
        using Rig rig = new();

        Mesh mesh = rig.Pipeline.Generate(Image(), new Trellis2Options { Grid = 512, Seed = 1 });

        // The stub field sits on voxels (0..1, 0..1, 0) at the grid origin, so the quad lies within a voxel of (-0.5, -0.5, -0.5):
        // x stays, y takes the old z (-0.5 + k/512), z takes minus the old y (about +0.5 - k/512).
        Assert.InRange(mesh.Vertices[0], -0.5f, -0.5f + 4f / 512f);
        Assert.InRange(mesh.Vertices[1], -0.5f, -0.5f + 4f / 512f);
        Assert.InRange(mesh.Vertices[2], 0.5f - 4f / 512f, 0.5f);
    }

    private sealed class Rig : IDisposable
    {
        internal Rig(int maxTokens = Trellis2Cascade.DefaultMaxTokens, bool emptyField = false)
        {
            Backend = new CpuBackend();
            Owned = new OwnedStub();
            Pipeline = new Trellis2ImageTo3DPipeline(
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

        internal Trellis2ImageTo3DPipeline Pipeline { get; }

        internal List<string> Calls { get; } = [];

        internal List<Tensor> Tensors { get; } = [];

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

    private sealed class StubConditioner(Rig rig) : ITrellis2Conditioner
    {
        public Tensor Encode(IBackend backend, Trellis2PreparedImage image, int resolution)
        {
            rig.Calls.Add($"cond@{resolution}");
            Tensor cond = new(new TensorShape(1, 3, 4), DType.F32);
            rig.Tensors.Add(cond);
            return cond;
        }
    }

    private sealed class StubStructure(Rig rig) : ITrellis2StructureStage
    {
        public int[] Sample(
            IBackend backend, Tensor cond, Tensor negCond, int seed, Trellis2SamplerParams parameters, int outputResolution, Action<int>? onStep)
        {
            rig.Calls.Add($"structure@{outputResolution} steps={parameters.Steps} seed={seed}");
            rig.Tensors.Add(negCond);
            for (int s = 1; s <= parameters.Steps; s++)
                onStep?.Invoke(s);
            return [0, 1, 1, 1, 0, 2, 2, 2];
        }
    }

    private sealed class StubShape(Rig rig) : ITrellis2ShapeFlowStage
    {
        public SparseTensor Sample(
            IBackend backend, Trellis2ShapeFlowKind kind, int[] coords, int gridResolution, Tensor cond, Tensor negCond, int seed,
            Trellis2SamplerParams parameters, Action<int>? onStep)
        {
            rig.Calls.Add($"shape {kind} grid={gridResolution} seed={seed} steps={parameters.Steps}");
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
            return [0, 16, 16, 16, 0, 17, 16, 16, 0, 40, 40, 40];
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
