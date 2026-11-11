using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.UniRig;

namespace HartsyInference.ThreeD.Tests.UniRig;

/// <summary>Random weights with the skin checkpoint's key names for tiny configurations.</summary>
internal static class UniRigSkinTestWeights
{
    public static readonly UniRigPtv3Config TinyPoint = new() { Depths = [1, 1], Channels = [8, 16], Heads = [1, 2], PatchSize = 64 };

    public static readonly UniRigShapeEncoderConfig TinyGlobal = new() { Width = 16, Heads = 2, Layers = 1, NumFreqs = 2, PointFeats = 3, TokenNum = 4 };

    public static readonly UniRigSkinConfig TinySkin = new()
    {
        FeatDim = 16,
        NumHeads = 2,
        NumBoneAttn = 1,
        NumMeshBoneAttn = 2,
        BoneEmbedDim = 8,
        MlpDim = 8,
        PointChunk = 16,
        AttentionChunk = 32,
    };

    public static Dictionary<string, Tensor> Point(UniRigPtv3Config c, string prefix, int seed)
    {
        Random random = new(seed);
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        int c0 = c.Channels[0];
        w[$"{prefix}.embedding.stem.conv.weight"] = UniRigTestWeights.Random(random, 0.1, c0, 5, 5, 5, c.InChannels);
        UniRigTestWeights.Norm(w, random, $"{prefix}.embedding.stem.norm", c0);
        w[$"{prefix}.embedding.stem.norm.running_mean"] = UniRigTestWeights.Random(random, 0.1, c0);
        Tensor variance = UniRigTestWeights.Random(random, 0.2, c0);
        foreach (ref float v in variance.AsSpan<float>())
        {
            v += 1f;
        }

        w[$"{prefix}.embedding.stem.norm.running_var"] = variance;
        UniRigTestWeights.Linear(w, random, $"{prefix}.embedding.res_linear", c0, c.InChannels, bias: true);
        for (int s = 0; s < c.Depths.Length; s++)
        {
            string stage = $"{prefix}.enc.enc{s}";
            int ch = c.Channels[s];
            if (s > 0)
            {
                UniRigTestWeights.Linear(w, random, $"{stage}.0", ch, c.Channels[s - 1], bias: true);
            }

            for (int i = 0; i < c.Depths[s]; i++)
            {
                string b = $"{stage}.block{i}";
                w[$"{b}.cpe.0.weight"] = UniRigTestWeights.Random(random, 0.1, ch, 3, 3, 3, ch);
                w[$"{b}.cpe.0.bias"] = UniRigTestWeights.Random(random, 0.05, ch);
                UniRigTestWeights.Linear(w, random, $"{b}.cpe.1", ch, ch, bias: true);
                UniRigTestWeights.Norm(w, random, $"{b}.cpe.2", ch);
                UniRigTestWeights.Norm(w, random, $"{b}.norm1.0", ch);
                UniRigTestWeights.Norm(w, random, $"{b}.norm2.0", ch);
                UniRigTestWeights.Linear(w, random, $"{b}.attn.qkv", 3 * ch, ch, bias: true);
                UniRigTestWeights.Linear(w, random, $"{b}.attn.proj", ch, ch, bias: true);
                UniRigTestWeights.Linear(w, random, $"{b}.mlp.0.fc1", 4 * ch, ch, bias: true);
                UniRigTestWeights.Linear(w, random, $"{b}.mlp.0.fc2", ch, 4 * ch, bias: true);
            }
        }

        return w;
    }

    public static Dictionary<string, Tensor> Skin(int seed)
    {
        Random random = new(seed);
        UniRigSkinConfig s = TinySkin;
        int d = s.FeatDim;
        Dictionary<string, Tensor> w = Point(TinyPoint, "mesh_encoder", seed);
        foreach (KeyValuePair<string, Tensor> pair in UniRigTestWeights.Encoder(TinyGlobal, "global_encoder.encoder", seed + 1))
        {
            w[pair.Key] = pair.Value;
        }

        UniRigTestWeights.Linear(w, random, "feat_map.0", d, TinyPoint.Channels[^1], bias: true);
        UniRigTestWeights.Norm(w, random, "feat_map.1", d);
        UniRigTestWeights.Linear(w, random, "out_proj.0", d, TinyGlobal.Width, bias: true);
        UniRigTestWeights.Norm(w, random, "out_proj.1", d);
        int width = 3 * (2 * s.BoneFreqs + 1);
        UniRigTestWeights.Linear(w, random, "bone_encoder.bone_encoder.1", s.BoneEmbedDim, width, bias: true);
        UniRigTestWeights.Norm(w, random, "bone_encoder.bone_encoder.2", s.BoneEmbedDim);
        UniRigTestWeights.Linear(w, random, "bone_encoder.bone_encoder.4", 4 * s.BoneEmbedDim, s.BoneEmbedDim, bias: true);
        UniRigTestWeights.Norm(w, random, "bone_encoder.bone_encoder.5", 4 * s.BoneEmbedDim);
        UniRigTestWeights.Linear(w, random, "bone_encoder.bone_encoder.7", d, 4 * s.BoneEmbedDim, bias: true);
        UniRigTestWeights.Norm(w, random, "bone_encoder.bone_encoder.8", d);
        for (int i = 0; i < s.NumBoneAttn; i++)
        {
            CrossAttn(w, random, $"bone_encoder.attn.{i}", d);
        }

        for (int i = 0; i < s.NumMeshBoneAttn; i++)
        {
            CrossAttn(w, random, $"mesh_bone_attn.{i}", d);
        }

        UniRigTestWeights.Linear(w, random, "qmesh", d * s.NumHeads, d, bias: true);
        UniRigTestWeights.Linear(w, random, "kmesh", d * s.NumHeads, d, bias: true);
        UniRigTestWeights.Linear(w, random, "voxel_skin_embed", s.NumHeads, 1, bias: true);
        UniRigTestWeights.Norm(w, random, "voxel_skin_norm", s.NumHeads);
        UniRigTestWeights.Norm(w, random, "attn_skin_norm", s.NumHeads);
        UniRigTestWeights.Linear(w, random, "downscale.0", s.NumHeads, 2 * s.NumHeads, bias: true);
        UniRigTestWeights.Norm(w, random, "downscale.1", s.NumHeads);
        int m = s.MlpDim;
        UniRigTestWeights.Linear(w, random, "skinweight_pred.net.0", m, s.NumHeads, bias: true);
        UniRigTestWeights.Norm(w, random, "skinweight_pred.net.1", m);
        foreach ((int linear, int norm) in new[] { (3, 4), (6, 7), (9, 10) })
        {
            UniRigTestWeights.Linear(w, random, $"skinweight_pred.net.{linear}", m, m, bias: true);
            UniRigTestWeights.Norm(w, random, $"skinweight_pred.net.{norm}", m);
        }

        UniRigTestWeights.Linear(w, random, "skinweight_pred.net.12", 1, m, bias: true);
        return w;
    }

    private static void CrossAttn(Dictionary<string, Tensor> w, Random random, string path, int d)
    {
        UniRigTestWeights.Linear(w, random, $"{path}.attention.Wq", d, d, bias: true);
        UniRigTestWeights.Linear(w, random, $"{path}.attention.Wkv", 2 * d, d, bias: true);
        UniRigTestWeights.Linear(w, random, $"{path}.attention.out_proj", d, d, bias: true);
        UniRigTestWeights.Norm(w, random, $"{path}.norm1", d);
        UniRigTestWeights.Norm(w, random, $"{path}.norm2", d);
        UniRigTestWeights.Linear(w, random, $"{path}.ffn.0", 4 * d, d, bias: true);
        UniRigTestWeights.Linear(w, random, $"{path}.ffn.2", d, 4 * d, bias: true);
    }
}
