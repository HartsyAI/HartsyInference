using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.CoLa3D;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Stages;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

/// <summary>Checks the pipeline's orchestration with stub stages: resolution default, one mesh per object, prompt checks, and disposal.</summary>
public sealed class CoLa3DPipelineTests
{
    private const string TriangleScene = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";

    [Fact]
    public void Generate_RunsEveryStageAndReturnsOneMeshPerObjectAtTheDefaultResolution()
    {
        RecordingVoxelizer voxelizer = new();
        RecordingDisposable owned = new();
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = NewPipeline(backend, CoLa3DConfig.FreeQ20, voxelizer, new StubDecomposer(), [owned]);

        CoLa3DResult result = pipeline.Generate(ObjReader.Parse(TriangleScene));

        Assert.Equal(CoLa3DPipeline.DefaultResolution, voxelizer.Resolution);
        Assert.Equal(2, result.Objects.Count);
        Assert.False(owned.Disposed);
        pipeline.Dispose();
        Assert.True(owned.Disposed);
    }

    [Fact]
    public void Generate_RejectsAResolutionThatIsNotAMultipleOfTheLatentStride()
    {
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = NewPipeline(backend, CoLa3DConfig.FreeQ20, new RecordingVoxelizer(), new StubDecomposer());

        Assert.Throws<ArgumentException>(() => pipeline.Generate(ObjReader.Parse(TriangleScene), resolution: 1000));
    }

    [Fact]
    public void Generate_PassesFreeQueriesWithoutPromptInputs()
    {
        StubDecomposer decomposer = new();
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = NewPipeline(backend, CoLa3DConfig.FreeQ20, new RecordingVoxelizer(), decomposer);

        pipeline.Generate(ObjReader.Parse(TriangleScene));

        Assert.True(decomposer.WasCalled);
        Assert.Null(decomposer.Received);
    }

    [Fact]
    public void Generate_HandsAMaskPromptToTheDecomposer()
    {
        StubDecomposer decomposer = new();
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = NewPipeline(backend, CoLa3DConfig.MaskQ20, new RecordingVoxelizer(), decomposer);
        CoLa3DPrompt prompt = new() { Queries = 2, MaskCoverage = CoverageOf(2, 3, 3), MaskHeight = 3, MaskWidth = 3 };

        pipeline.Generate(ObjReader.Parse(TriangleScene), prompt: prompt);

        Assert.NotNull(decomposer.Received);
        Assert.Equal(2, decomposer.Received!.Queries);
        Assert.Equal(new TensorShape(1, 2, 3, 3), decomposer.Received.Masks!.Shape);
    }

    [Fact]
    public void Generate_RejectsAPromptThatGivesMasksAndPoints()
    {
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = NewPipeline(backend, CoLa3DConfig.MaskQ20, new RecordingVoxelizer(), new StubDecomposer());
        CoLa3DPrompt prompt = new()
        {
            Queries = 1,
            MaskCoverage = CoverageOf(1, 3, 3),
            MaskHeight = 3,
            MaskWidth = 3,
            Points = [0f, 0f, 0f],
        };

        Assert.Throws<ArgumentException>(() => pipeline.Generate(ObjReader.Parse(TriangleScene), prompt: prompt));
    }

    [Fact]
    public void Generate_RejectsAMaskPromptOnAFreeCheckpointBeforeVoxelizing()
    {
        RecordingVoxelizer voxelizer = new();
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = NewPipeline(backend, CoLa3DConfig.FreeQ20, voxelizer, new StubDecomposer());
        CoLa3DPrompt prompt = new() { Queries = 1, MaskCoverage = CoverageOf(1, 2, 2), MaskHeight = 2, MaskWidth = 2 };

        Assert.Throws<ArgumentException>(() => pipeline.Generate(ObjReader.Parse(TriangleScene), prompt: prompt));
        Assert.False(voxelizer.Called);
    }

    [Fact]
    public void Generate_RejectsAPromptCheckpointRunWithoutAPromptBeforeVoxelizing()
    {
        RecordingVoxelizer voxelizer = new();
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = NewPipeline(backend, CoLa3DConfig.PointQ20, voxelizer, new StubDecomposer());

        Assert.Throws<ArgumentException>(() => pipeline.Generate(ObjReader.Parse(TriangleScene)));
        Assert.False(voxelizer.Called);
    }

    [Fact]
    public void Generate_RejectsMoreObjectsThanTheCheckpointTakesBeforeVoxelizing()
    {
        RecordingVoxelizer voxelizer = new();
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = NewPipeline(backend, CoLa3DConfig.MaskQ20, voxelizer, new StubDecomposer());
        int queries = CoLa3DConfig.MaskQ20.MaxParts + 1;
        CoLa3DPrompt prompt = new() { Queries = queries, MaskCoverage = CoverageOf(queries, 1, 1), MaskHeight = 1, MaskWidth = 1 };

        ArgumentException error = Assert.Throws<ArgumentException>(() => pipeline.Generate(ObjReader.Parse(TriangleScene), prompt: prompt));
        Assert.Contains("at most 20", error.Message, StringComparison.Ordinal);
        Assert.False(voxelizer.Called);
    }

    [Fact]
    public void Generate_RejectsAPointPromptOnAGridThatIsNotTheLatentResolutionBeforeVoxelizing()
    {
        RecordingVoxelizer voxelizer = new();
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = NewPipeline(backend, CoLa3DConfig.PointQ20, voxelizer, new StubDecomposer());
        CoLa3DPrompt prompt = new() { Queries = 1, Points = [0.5f, 0.5f, 0.5f] };

        Assert.Throws<ArgumentException>(() => pipeline.Generate(ObjReader.Parse(TriangleScene), resolution: 512, prompt: prompt));
        Assert.False(voxelizer.Called);
    }

    [Fact]
    public void Generate_AcceptsAPointPromptOnTheLatentResolutionGrid()
    {
        StubDecomposer decomposer = new();
        RecordingVoxelizer voxelizer = new();
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = NewPipeline(backend, CoLa3DConfig.PointQ20, voxelizer, decomposer);
        CoLa3DPrompt prompt = new() { Queries = 1, Points = [0.5f, 0.5f, 0.5f] };

        pipeline.Generate(ObjReader.Parse(TriangleScene), prompt: prompt);

        Assert.Equal(CoLa3DConfig.PointQ20.LatentResolution * CoLa3DPipeline.LatentStride, voxelizer.Resolution);
        Assert.NotNull(decomposer.Received);
        Assert.Equal(1, decomposer.Received!.Queries);
    }

    // Builds a pipeline over the stub stages; only the config, voxelizer and decomposer vary between tests.
    private static CoLa3DPipeline NewPipeline(
        IBackend backend,
        CoLa3DConfig config,
        ISceneVoxelizer voxelizer,
        IObjectDecomposer decomposer,
        IEnumerable<IDisposable>? owned = null) =>
        new(backend, config, voxelizer, new StubEncoder(), new StubFeatures(), decomposer, new StubDecoder(), new StubMesher(), owned ?? []);

    // Mask coverage of `queries` planes, each `height` x `width` and fully covered, so every query is non-empty.
    private static byte[] CoverageOf(int queries, int height, int width)
    {
        byte[] coverage = new byte[queries * height * width];
        Array.Fill(coverage, (byte)255);
        return coverage;
    }

    [Fact]
    public void Generate_RejectsAResolutionAboveTheMaximum()
    {
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = new(
            backend, CoLa3DConfig.ForQuerySource(CoLa3DConfig.QuerySourceKind.Fixed), new RecordingVoxelizer(),new StubEncoder(), new StubFeatures(), new StubDecomposer(), new StubDecoder(), new StubMesher(), []);
        Mesh scene = ObjReader.Parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");

        Assert.Throws<ArgumentException>(() => pipeline.Generate(scene, resolution: CoLa3DPipeline.MaxResolution + 16));
    }

    [Fact]
    public void Generate_ReportsTheFixedStagesThenOncePerObjectWithMonotonicSteps()
    {
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = new(
            backend, CoLa3DConfig.ForQuerySource(CoLa3DConfig.QuerySourceKind.Fixed), new RecordingVoxelizer(),new StubEncoder(), new StubFeatures(), new StubDecomposer(), new StubDecoder(), new StubMesher(), []);
        Mesh scene = ObjReader.Parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
        List<GenerationProgress> reports = [];

        pipeline.Generate(scene, onProgress: reports.Add);

        Assert.Equal(new[] { (1, 3), (2, 3), (3, 5), (4, 5), (5, 5) }, reports.Select(p => (p.Step, p.TotalSteps)));
        Assert.All(reports, p => Assert.True(p.ElapsedMs >= 0));
    }

    // Steps 1 and 2 come before the split, so a cancel there must stop the run as well as one after it.
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void Generate_PropagatesACancelRaisedFromProgress(int cancelAtStep)
    {
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = new(
            backend, CoLa3DConfig.ForQuerySource(CoLa3DConfig.QuerySourceKind.Fixed), new RecordingVoxelizer(),new StubEncoder(), new StubFeatures(), new StubDecomposer(), new StubDecoder(), new StubMesher(), []);
        Mesh scene = ObjReader.Parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");

        Assert.Throws<OperationCanceledException>(() => pipeline.Generate(scene, onProgress: p =>
        {
            if (p.Step == cancelAtStep)
                throw new OperationCanceledException();
        }));
    }

    [Fact]
    public void Generate_ThrowsWhenNoObjectDecodesToTriangles()
    {
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = new(
            backend, CoLa3DConfig.ForQuerySource(CoLa3DConfig.QuerySourceKind.Fixed), new RecordingVoxelizer(),new StubEncoder(), new StubFeatures(), new StubDecomposer(), new StubDecoder(), new EmptyMesher(), []);
        Mesh scene = ObjReader.Parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");

        Assert.Throws<HartsyInferenceException>(() => pipeline.Generate(scene));
    }

    private sealed class EmptyMesher : IDualGridMesher
    {
        public Mesh Extract(DualGridField field, SceneGrid grid) => new() { Vertices = [], Indices = [] };
    }

    private sealed class RecordingVoxelizer : ISceneVoxelizer
    {
        public int Resolution { get; private set; }

        public bool Called { get; private set; }

        public DualGridVoxels Voxelize(Mesh scene, SceneGrid grid)
        {
            Called = true;
            Resolution = grid.Resolution;
            return new DualGridVoxels([0, 0, 0, 1, 0, 0], [0.5f, 0.5f, 0.5f, 1.5f, 0.5f, 0.5f], new bool[6]);
        }
    }

    private sealed class StubEncoder : ISparseShapeEncoder
    {
        public SparseTensor Encode(IBackend backend, DualGridVoxels voxels, int resolution) =>
            new(new Tensor(new TensorShape(2, 4), DType.F32), [0, 0, 0, 0, 0, 1, 0, 0], resolution / 16);
    }

    private sealed class StubFeatures : IPointFeatureExtractor
    {
        public SparseTensor Extract(IBackend backend, int[] coords, int resolution) =>
            new(new Tensor(new TensorShape(2, 5), DType.F32), [0, 0, 0, 0, 0, 1, 0, 0], resolution / 16);
    }

    private sealed class StubDecomposer : IObjectDecomposer
    {
        public CoLa3DQueryPrompts? Received { get; private set; }

        public bool WasCalled { get; private set; }

        public IReadOnlyList<SparseTensor> Decompose(
            IBackend backend,
            SparseTensor latent,
            SparseTensor features,
            CoLa3DQueryPrompts? prompts = null)
        {
            WasCalled = true;
            Received = prompts;
            return
            [
                new SparseTensor(new Tensor(new TensorShape(1, 4), DType.F32), [0, 0, 0, 0], latent.Resolution),
                new SparseTensor(new Tensor(new TensorShape(1, 4), DType.F32), [0, 0, 1, 0], latent.Resolution),
            ];
        }
    }

    private sealed class StubDecoder : ISparseShapeDecoder
    {
        public DualGridField Decode(IBackend backend, SparseTensor latent) =>
            new(new DualGridVoxels([0, 0, 0], [0.5f, 0.5f, 0.5f], new bool[3]), [1f]);
    }

    private sealed class StubMesher : IDualGridMesher
    {
        public Mesh Extract(DualGridField field, SceneGrid grid) =>
            new() { Vertices = [0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f], Indices = [0, 1, 2] };
    }

    private sealed class RecordingDisposable : IDisposable
    {
        public bool Disposed { get; private set; }

        public void Dispose() => Disposed = true;
    }
}
