using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.UniRig;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.ThreeD.Tests.UniRig;

public sealed class UniRigPipelineTests
{
    private static Mesh Quad() => new() { Vertices = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], Indices = [0, 1, 2, 0, 2, 3] };

    [Fact]
    public void Constructor_RequiresBothCheckpoints()
    {
        string empty = Path.Combine(Path.GetTempPath(), "unirig-missing-" + Guid.NewGuid().ToString("N"));
        using CpuBackend backend = new();
        FileNotFoundException error = Assert.Throws<FileNotFoundException>(() => new UniRigPipeline(backend, empty));
        Assert.Contains("skeleton", error.Message);
    }

    [Fact]
    public void ValidateMesh_RejectsAMeshWithoutTriangles()
    {
        UniRigPipeline.ValidateMesh(Quad());
        Assert.Throws<HartsyInferenceException>(() => UniRigPipeline.ValidateMesh(new Mesh { Vertices = [0, 0, 0], Indices = [] }));
        Assert.Throws<ArgumentNullException>(() => UniRigPipeline.ValidateMesh(null!));
    }

    [Fact]
    public void ToSource_MapsHeadsAndTailsBackIntoTheInputMeshFrame()
    {
        Mesh mesh = new() { Vertices = [10, 0, 0, 14, 0, 0, 12, 2, 0], Indices = [0, 1, 2] };
        UniRigPreparedMesh prepared = UniRigMeshPrep.Prepare(mesh);
        UniRigSkeleton skeleton = new()
        {
            Joints = [-1f, 0f, 0f, 1f, 0f, 0f],
            Tails = [0f, 0f, 0f, 1f, 1f, 0f],
            Parents = [-1, 0],
            Names = ["bone_0", "bone_1"],
            Parts = [],
        };

        UniRigSkeleton source = UniRigPipeline.ToSource(skeleton, prepared);

        Assert.Equal([10f, 1f, 0f, 14f, 1f, 0f], source.Joints);
        Assert.Equal([12f, 1f, 0f, 14f, 3f, 0f], source.Tails);
        Assert.Equal(skeleton.Parents, source.Parents);
    }
}

/// <summary>End-to-end run on real weights. Integration lane: skipped unless HARTSY_UNIRIG_DIR holds both checkpoints.</summary>
[Trait("Category", "Integration")]
public sealed class UniRigPipelineRealWeightsTests
{
    [UniRigRealWeightsFact]
    public void Generate_RigsAHumanoidIntoAStructurallyValidGlb()
    {
        string root = Environment.GetEnvironmentVariable("HARTSY_UNIRIG_DIR")!;
        using CpuBackend backend = new();
        using UniRigPipeline pipeline = new(backend, root);
        Mesh mesh = Humanoid();

        UniRigResult result = pipeline.Generate(mesh, seed: 3);

        Assert.True(result.Skeleton.JointCount >= 3);
        int jsonLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(result.Glb.AsSpan(12));
        using JsonDocument doc = JsonDocument.Parse(result.Glb.AsMemory(20, jsonLength));
        JsonElement root0 = doc.RootElement;
        Assert.Equal(result.Skeleton.JointCount, root0.GetProperty("skins")[0].GetProperty("joints").GetArrayLength());
        Assert.Equal(result.Skeleton.JointCount + 1, root0.GetProperty("nodes").GetArrayLength());
        JsonElement attributes = root0.GetProperty("meshes")[0].GetProperty("primitives")[0].GetProperty("attributes");
        Assert.True(attributes.TryGetProperty("JOINTS_0", out _));
        Assert.True(attributes.TryGetProperty("WEIGHTS_0", out _));

        int binStart = 20 + jsonLength + 8;
        JsonElement accessors = root0.GetProperty("accessors");
        JsonElement views = root0.GetProperty("bufferViews");
        JsonElement weightsAccessor = accessors[attributes.GetProperty("WEIGHTS_0").GetInt32()];
        int offset = binStart + views[weightsAccessor.GetProperty("bufferView").GetInt32()].GetProperty("byteOffset").GetInt32();
        for (int v = 0; v < mesh.VertexCount; v++)
        {
            float sum = 0f;
            for (int k = 0; k < 4; k++)
            {
                float w = BinaryPrimitives.ReadSingleLittleEndian(result.Glb.AsSpan(offset + 16 * v + 4 * k));
                Assert.True(w >= 0f);
                sum += w;
            }

            Assert.Equal(1f, sum, 3);
        }
    }

    private static Mesh Humanoid()
    {
        List<float> vertices = [];
        List<int> indices = [];
        AddBox(vertices, indices, 0f, 1.0f, 0f, 0.25f, 0.35f, 0.12f);
        AddBox(vertices, indices, 0f, 1.6f, 0f, 0.12f, 0.14f, 0.12f);
        AddBox(vertices, indices, -0.6f, 1.25f, 0f, 0.35f, 0.07f, 0.07f);
        AddBox(vertices, indices, 0.6f, 1.25f, 0f, 0.35f, 0.07f, 0.07f);
        AddBox(vertices, indices, -0.12f, 0.35f, 0f, 0.09f, 0.35f, 0.09f);
        AddBox(vertices, indices, 0.12f, 0.35f, 0f, 0.09f, 0.35f, 0.09f);
        return new Mesh { Vertices = [.. vertices], Indices = [.. indices] };
    }

    private static void AddBox(List<float> vertices, List<int> indices, float cx, float cy, float cz, float hx, float hy, float hz)
    {
        int baseIndex = vertices.Count / 3;
        for (int i = 0; i < 8; i++)
        {
            vertices.AddRange([cx + ((i & 1) == 0 ? -hx : hx), cy + ((i & 2) == 0 ? -hy : hy), cz + ((i & 4) == 0 ? -hz : hz)]);
        }

        int[][] quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
        foreach (int[] q in quads)
        {
            indices.AddRange([baseIndex + q[0], baseIndex + q[1], baseIndex + q[2], baseIndex + q[0], baseIndex + q[2], baseIndex + q[3]]);
        }
    }
}
