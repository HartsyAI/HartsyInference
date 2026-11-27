using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.UniRig;

namespace HartsyInference.ThreeD.Tests.UniRig;

/// <summary>Real-weight checks of the skeleton checkpoint. Integration lane: set HARTSY_UNIRIG_DIR or place Models/3D/unirig beside the repo; a missing file fails the test.</summary>
[Trait("Category", "Integration")]
public sealed class UniRigSkeletonCheckpointTests
{
    private const string RelativePath = "skeleton/articulation-xl_quantization_256/model.ckpt";

    [Fact]
    public void Checkpoint_KeysAndShapesMatchTheModelExactly()
    {
        using UniRigCheckpoint checkpoint = UniRigCheckpoint.Load(CheckpointPath());
        HashSet<string> expected = ExpectedKeys();
        HashSet<string> actual = [.. checkpoint.Tensors.Keys];
        Assert.Empty(expected.Except(actual).Order(StringComparer.Ordinal));
        Assert.Empty(actual.Except(expected).Order(StringComparer.Ordinal));
        Assert.Equal(585, actual.Count);
        Assert.Equal(new TensorShape(267, 1024), checkpoint.Tensors["transformer.lm_head.weight"].Shape);
        Assert.Equal(new TensorShape(3078, 1024), checkpoint.Tensors["transformer.model.decoder.embed_positions.weight"].Shape);
        Assert.Equal(new TensorShape(512, 54), checkpoint.Tensors["mesh_encoder.encoder.input_proj.weight"].Shape);
    }

    [Fact]
    public void Model_BuildsFromTheRealCheckpointAndGeneratesAStructurallyValidSkeleton()
    {
        using UniRigCheckpoint checkpoint = UniRigCheckpoint.Load(CheckpointPath());
        using CpuBackend backend = new();
        using UniRigSkeletonModel model = new(backend, checkpoint);
        UniRigPreparedMesh mesh = UniRigMeshPrep.Prepare(Humanoid());
        UniRigPointSample sample = UniRigSurfaceSampler.Sample(mesh, 65536, 8192, seed: 1);

        UniRigSkeleton skeleton = model.Generate(sample.Points, sample.Normals, new UniRigSamplingOptions { Seed = 1 });

        Assert.True(skeleton.JointCount >= 3, $"only {skeleton.JointCount} joints");
        Assert.Equal(-1, skeleton.Parents[0]);
        for (int i = 1; i < skeleton.JointCount; i++)
        {
            Assert.InRange(skeleton.Parents[i], 0, i - 1);
        }

        Assert.All(skeleton.Joints, v => Assert.InRange(v, -1f, 1f));
        Assert.All(skeleton.Tails, v => Assert.True(float.IsFinite(v)));
    }

    private static HashSet<string> ExpectedKeys()
    {
        HashSet<string> keys = [];
        string enc = "mesh_encoder.encoder";
        keys.UnionWith([$"{enc}.input_proj.weight", $"{enc}.input_proj.bias", $"{enc}.ln_post.weight", $"{enc}.ln_post.bias", "output_proj.weight", "output_proj.bias"]);
        foreach (string ln in new[] { "ln_1", "ln_2", "ln_3" })
        {
            keys.UnionWith([$"{enc}.cross_attn.{ln}.weight", $"{enc}.cross_attn.{ln}.bias"]);
        }

        keys.UnionWith([$"{enc}.cross_attn.attn.c_q.weight", $"{enc}.cross_attn.attn.c_kv.weight", $"{enc}.cross_attn.attn.c_proj.weight", $"{enc}.cross_attn.attn.c_proj.bias"]);
        AddMlp(keys, $"{enc}.cross_attn.mlp");
        for (int i = 0; i < 16; i++)
        {
            string b = $"{enc}.self_attn.resblocks.{i}";
            keys.UnionWith([$"{b}.ln_1.weight", $"{b}.ln_1.bias", $"{b}.ln_2.weight", $"{b}.ln_2.bias", $"{b}.attn.c_qkv.weight", $"{b}.attn.c_proj.weight", $"{b}.attn.c_proj.bias"]);
            AddMlp(keys, $"{b}.mlp");
        }

        string dec = "transformer.model.decoder";
        keys.UnionWith(["transformer.lm_head.weight", $"{dec}.embed_positions.weight", $"{dec}.embed_tokens.weight", $"{dec}.final_layer_norm.weight", $"{dec}.final_layer_norm.bias"]);
        for (int i = 0; i < 24; i++)
        {
            string l = $"{dec}.layers.{i}";
            foreach (string name in new[] { "fc1", "fc2", "final_layer_norm", "self_attn.k_proj", "self_attn.out_proj", "self_attn.q_proj", "self_attn.v_proj", "self_attn_layer_norm" })
            {
                keys.UnionWith([$"{l}.{name}.weight", $"{l}.{name}.bias"]);
            }
        }

        return keys;
    }

    private static void AddMlp(HashSet<string> keys, string prefix) =>
        keys.UnionWith([$"{prefix}.c_fc.weight", $"{prefix}.c_fc.bias", $"{prefix}.c_proj.weight", $"{prefix}.c_proj.bias"]);

    private static string CheckpointPath()
    {
        string? root = Environment.GetEnvironmentVariable("HARTSY_UNIRIG_DIR");
        if (string.IsNullOrWhiteSpace(root))
        {
            DirectoryInfo? dir = new(AppContext.BaseDirectory);
            while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "Models", "3D", "unirig")))
            {
                dir = dir.Parent;
            }

            root = dir is null ? AppContext.BaseDirectory : Path.Combine(dir.FullName, "Models", "3D", "unirig");
        }

        string path = Path.Combine(root, RelativePath);
        Assert.True(File.Exists(path), $"UniRig skeleton checkpoint not found at {path}. Set HARTSY_UNIRIG_DIR to the folder holding {RelativePath}.");
        return path;
    }

    /// <summary>Boxes for torso, head, two arms and two legs: a crude T-pose figure about 1.8 units tall.</summary>
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
