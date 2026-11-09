using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Stages;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

/// <summary>Checks the pipeline's orchestration with stub stages: resolution default, one mesh per object, and disposal.</summary>
public sealed class CoLa3DPipelineTests
{
    [Fact]
    public void Generate_RunsEveryStageAndReturnsOneMeshPerObjectAtTheDefaultResolution()
    {
        RecordingVoxelizer voxelizer = new();
        RecordingDisposable owned = new();
        using CpuBackend backend = new();
        using CoLa3DPipeline pipeline = new(
            backend, voxelizer, new StubEncoder(), new StubFeatures(), new StubDecomposer(), new StubDecoder(), new StubMesher(), [owned]);
        Mesh scene = ObjReader.Parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");

        CoLa3DResult result = pipeline.Generate(scene);

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
        using CoLa3DPipeline pipeline = new(
            backend, new RecordingVoxelizer(), new StubEncoder(), new StubFeatures(), new StubDecomposer(), new StubDecoder(), new StubMesher(), []);
        Mesh scene = ObjReader.Parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");

        Assert.Throws<ArgumentException>(() => pipeline.Generate(scene, resolution: 1000));
    }

    private sealed class RecordingVoxelizer : ISceneVoxelizer
    {
        public int Resolution { get; private set; }

        public DualGridVoxels Voxelize(Mesh scene, SceneGrid grid)
        {
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
        public IReadOnlyList<SparseTensor> Decompose(IBackend backend, SparseTensor latent, SparseTensor features) =>
        [
            new SparseTensor(new Tensor(new TensorShape(1, 4), DType.F32), [0, 0, 0, 0], latent.Resolution),
            new SparseTensor(new Tensor(new TensorShape(1, 4), DType.F32), [0, 0, 1, 0], latent.Resolution),
        ];
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
