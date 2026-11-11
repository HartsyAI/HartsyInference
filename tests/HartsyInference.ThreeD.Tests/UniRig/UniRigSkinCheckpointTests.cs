using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.UniRig;

namespace HartsyInference.ThreeD.Tests.UniRig;

/// <summary>Real-weight checks of the skin checkpoint. Integration lane: set HARTSY_UNIRIG_DIR or place Models/3D/unirig beside the repo; a missing file fails the test.</summary>
[Trait("Category", "Integration")]
public sealed class UniRigSkinCheckpointTests
{
    private const string RelativePath = "skin/articulation-xl/model.ckpt";

    [Fact]
    public void Checkpoint_KeysMatchTheModelExactly()
    {
        using UniRigCheckpoint checkpoint = UniRigCheckpoint.Load(CheckpointPath());
        HashSet<string> expected = ExpectedKeys();
        HashSet<string> actual = [.. checkpoint.Tensors.Keys];
        Assert.Empty(expected.Except(actual).Order(StringComparer.Ordinal));
        Assert.Empty(actual.Except(expected).Order(StringComparer.Ordinal));
        Assert.Equal(new TensorShape(new long[] { 32, 5, 5, 5, 9 }), checkpoint.Tensors["mesh_encoder.embedding.stem.conv.weight"].Shape);
        Assert.Equal(new TensorShape(new long[] { 384, 3, 3, 3, 384 }), checkpoint.Tensors["mesh_encoder.enc.enc4.block15.cpe.0.weight"].Shape);
        Assert.Equal(new TensorShape(12288, 768), checkpoint.Tensors["qmesh.weight"].Shape);
    }

    [Fact]
    public void Model_BuildsFromTheRealCheckpoint_AndSkinsACubeWithValidWeights()
    {
        using UniRigCheckpoint checkpoint = UniRigCheckpoint.Load(CheckpointPath());
        using CpuBackend backend = new();
        using UniRigSkinModel model = new(backend, checkpoint);
        UniRigPreparedMesh mesh = UniRigMeshPrep.Prepare(Column());
        UniRigSkeleton skeleton = new()
        {
            Joints = [0f, -0.8f, 0f, 0f, 0f, 0f, 0f, 0.8f, 0f],
            Tails = [0f, 0f, 0f, 0f, 0.8f, 0f, 0f, 0.9f, 0f],
            Parents = [-1, 0, 1],
            Names = ["bone_0", "bone_1", "bone_2"],
            Parts = [],
        };

        UniRigSkinResult skin = UniRigSkinner.Skin(model, mesh, skeleton, new UniRigVoxelSkinOptions { Grid = 48 }, seed: 1);

        Assert.Equal(mesh.VertexCount, skin.VertexCount);
        Assert.Equal(3, skin.JointCount);
        for (int v = 0; v < skin.VertexCount; v++)
        {
            float total = 0f;
            for (int j = 0; j < 3; j++)
            {
                float w = skin.Weights[v * 3 + j];
                Assert.True(w >= 0f && float.IsFinite(w), $"vertex {v} joint {j}: {w}");
                total += w;
            }

            Assert.Equal(1f, total, 3);
        }
    }

    private static HashSet<string> ExpectedKeys()
    {
        HashSet<string> keys = [];
        void Pair(string path) => keys.UnionWith([$"{path}.weight", $"{path}.bias"]);

        keys.UnionWith(["mesh_encoder.embedding.stem.conv.weight", "mesh_encoder.embedding.stem.norm.running_mean", "mesh_encoder.embedding.stem.norm.running_var"]);
        Pair("mesh_encoder.embedding.stem.norm");
        Pair("mesh_encoder.embedding.res_linear");
        int[] depths = [3, 3, 3, 6, 16];
        for (int s = 0; s < depths.Length; s++)
        {
            if (s > 0)
            {
                Pair($"mesh_encoder.enc.enc{s}.0");
            }

            for (int i = 0; i < depths[s]; i++)
            {
                string b = $"mesh_encoder.enc.enc{s}.block{i}";
                foreach (string name in new[] { "attn.qkv", "attn.proj", "cpe.0", "cpe.1", "cpe.2", "mlp.0.fc1", "mlp.0.fc2", "norm1.0", "norm2.0" })
                {
                    Pair($"{b}.{name}");
                }
            }
        }

        string g = "global_encoder.encoder";
        Pair($"{g}.input_proj");
        Pair($"{g}.ln_post");
        foreach (string ln in new[] { "ln_1", "ln_2", "ln_3" })
        {
            Pair($"{g}.cross_attn.{ln}");
        }

        keys.UnionWith([$"{g}.cross_attn.attn.c_q.weight", $"{g}.cross_attn.attn.c_kv.weight"]);
        Pair($"{g}.cross_attn.attn.c_proj");
        Pair($"{g}.cross_attn.mlp.c_fc");
        Pair($"{g}.cross_attn.mlp.c_proj");
        for (int i = 0; i < 16; i++)
        {
            string b = $"{g}.self_attn.resblocks.{i}";
            Pair($"{b}.ln_1");
            Pair($"{b}.ln_2");
            keys.Add($"{b}.attn.c_qkv.weight");
            Pair($"{b}.attn.c_proj");
            Pair($"{b}.mlp.c_fc");
            Pair($"{b}.mlp.c_proj");
        }

        foreach (string name in new[] { "feat_map.0", "feat_map.1", "out_proj.0", "out_proj.1", "qmesh", "kmesh", "voxel_skin_embed", "voxel_skin_norm", "attn_skin_norm", "downscale.0", "downscale.1" })
        {
            Pair(name);
        }

        foreach (int index in new[] { 1, 2, 4, 5, 7, 8 })
        {
            Pair($"bone_encoder.bone_encoder.{index}");
        }

        foreach (int index in new[] { 0, 1, 3, 4, 6, 7, 9, 10, 12 })
        {
            Pair($"skinweight_pred.net.{index}");
        }

        for (int i = 0; i < 8; i++)
        {
            AddCrossAttn(keys, $"bone_encoder.attn.{i}");
        }

        for (int i = 0; i < 16; i++)
        {
            AddCrossAttn(keys, $"mesh_bone_attn.{i}");
        }

        return keys;
    }

    private static void AddCrossAttn(HashSet<string> keys, string path)
    {
        foreach (string name in new[] { "attention.Wq", "attention.Wkv", "attention.out_proj", "norm1", "norm2", "ffn.0", "ffn.2" })
        {
            keys.UnionWith([$"{path}.{name}.weight", $"{path}.{name}.bias"]);
        }
    }

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
        Assert.True(File.Exists(path), $"UniRig skin checkpoint not found at {path}. Set HARTSY_UNIRIG_DIR to the folder holding {RelativePath}.");
        return path;
    }

    /// <summary>A tall box (a crude limb) as 8 shared vertices.</summary>
    private static Mesh Column() => new()
    {
        Vertices = [-0.2f, -1, -0.2f, 0.2f, -1, -0.2f, 0.2f, 1, -0.2f, -0.2f, 1, -0.2f, -0.2f, -1, 0.2f, 0.2f, -1, 0.2f, 0.2f, 1, 0.2f, -0.2f, 1, 0.2f],
        Indices = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3],
    };
}
