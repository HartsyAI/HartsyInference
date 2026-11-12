using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Sam3dObjects;
using HartsyInference.Vision.Dinov2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamStageConditionTests
{
    // Tiny stage: DINO width 16 on a 2 x 2 grid of 14-pixel patches (28 px), two layers, four registers; a point-patch embedder of width 8 with four windows.
    private static readonly Dinov2Preset Dino = new()
    {
        Name = "tiny", HiddenSize = 16, NumLayers = 2, NumHeads = 2, IntermediateSize = 32, ImageSize = 28, PatchSize = 14, NumRegisterTokens = 4,
    };

    private static readonly SamPointPatchConfig Patch = new(InputSize: 8, PatchSize: 4, EmbedDim: 8, Heads: 2, MlpRatio: 2.0, Remap: SamPointRemapType.Linear);

    private const int Size = 28;
    private const int PointSide = 8;

    private sealed class Owned(IEnumerable<Tensor> tensors) : IDisposable
    {
        public void Dispose()
        {
            foreach (Tensor t in tensors) t.Dispose();
        }
    }

    private static Tensor Filled(Random rng, double scale, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>())
            v = (float)((rng.NextDouble() - 0.5) * scale);
        return t;
    }

    private static void AddDino(Dictionary<string, Tensor> w, Random rng, string prefix)
    {
        int d = Dino.HiddenSize;
        w[prefix + "patch_embed.proj.weight"] = Filled(rng, 0.2, d, 3, Dino.PatchSize, Dino.PatchSize);
        w[prefix + "patch_embed.proj.bias"] = Filled(rng, 0.3, d);
        w[prefix + "cls_token"] = Filled(rng, 1.0, 1, 1, d);
        w[prefix + "pos_embed"] = Filled(rng, 1.0, 1, 1 + Dino.NumPatches, d);
        w[prefix + "register_tokens"] = Filled(rng, 1.0, 1, Dino.NumRegisterTokens, d);
        w[prefix + "norm.weight"] = Filled(rng, 1.0, d);
        w[prefix + "norm.bias"] = Filled(rng, 0.4, d);
        for (int i = 0; i < Dino.NumLayers; i++)
        {
            string b = $"{prefix}blocks.{i}.";
            w[b + "norm1.weight"] = Filled(rng, 1.0, d);
            w[b + "norm1.bias"] = Filled(rng, 0.4, d);
            w[b + "norm2.weight"] = Filled(rng, 1.0, d);
            w[b + "norm2.bias"] = Filled(rng, 0.4, d);
            w[b + "attn.qkv.weight"] = Filled(rng, 0.8, 3 * d, d);
            w[b + "attn.qkv.bias"] = Filled(rng, 0.4, 3 * d);
            w[b + "attn.proj.weight"] = Filled(rng, 0.8, d, d);
            w[b + "attn.proj.bias"] = Filled(rng, 0.4, d);
            w[b + "ls1.gamma"] = Filled(rng, 1.0, d);
            w[b + "ls2.gamma"] = Filled(rng, 1.0, d);
            w[b + "mlp.fc1.weight"] = Filled(rng, 0.8, Dino.IntermediateSize, d);
            w[b + "mlp.fc1.bias"] = Filled(rng, 0.4, Dino.IntermediateSize);
            w[b + "mlp.fc2.weight"] = Filled(rng, 0.8, d, Dino.IntermediateSize);
            w[b + "mlp.fc2.bias"] = Filled(rng, 0.4, d);
        }
    }

    private static void AddPointPatch(Dictionary<string, Tensor> w, Random rng, string prefix)
    {
        int d = Patch.EmbedDim, side = Patch.WindowsPerSide, hidden = Patch.MlpHidden;
        w[prefix + "point_proj.weight"] = Filled(rng, 1.0, d, 3);
        w[prefix + "point_proj.bias"] = Filled(rng, 0.5, d);
        w[prefix + "invalid_xyz_token"] = Filled(rng, 1.0, d);
        w[prefix + "pos_embed"] = Filled(rng, 1.0, 1, d, side, side);
        w[prefix + "pos_embed_window"] = Filled(rng, 1.0, 1, Patch.WindowTokens, d);
        w[prefix + "cls_token"] = Filled(rng, 1.0, 1, 1, d);
        w[prefix + "blocks.0.norm1.weight"] = Filled(rng, 1.0, d);
        w[prefix + "blocks.0.norm1.bias"] = Filled(rng, 0.4, d);
        w[prefix + "blocks.0.attn.qkv.weight"] = Filled(rng, 1.0, 3 * d, d);
        w[prefix + "blocks.0.attn.qkv.bias"] = Filled(rng, 0.4, 3 * d);
        w[prefix + "blocks.0.attn.proj.weight"] = Filled(rng, 1.0, d, d);
        w[prefix + "blocks.0.attn.proj.bias"] = Filled(rng, 0.4, d);
        w[prefix + "blocks.0.norm2.weight"] = Filled(rng, 1.0, d);
        w[prefix + "blocks.0.norm2.bias"] = Filled(rng, 0.4, d);
        w[prefix + "blocks.0.mlp.fc1.weight"] = Filled(rng, 1.0, hidden, d);
        w[prefix + "blocks.0.mlp.fc1.bias"] = Filled(rng, 0.4, hidden);
        w[prefix + "blocks.0.mlp.fc2.weight"] = Filled(rng, 1.0, d, hidden);
        w[prefix + "blocks.0.mlp.fc2.bias"] = Filled(rng, 0.4, d);
    }

    /// <summary>A condition-embedder section: both towers, the fuser (<paramref name="embedderWidths"/> projection nets) and, for the structure stage, the pointmap embedder.</summary>
    private static Dictionary<string, Tensor> Section(Random rng, bool withPointmap)
    {
        Dictionary<string, Tensor> w = [];
        AddDino(w, rng, "module_list.0.backbone.");
        AddDino(w, rng, "module_list.1.backbone.");
        int[] widths = withPointmap ? [Dino.HiddenSize, Dino.HiddenSize, Patch.EmbedDim] : [Dino.HiddenSize, Dino.HiddenSize];
        if (withPointmap)
            AddPointPatch(w, rng, "module_list.2.");
        int fused = Dino.HiddenSize, hidden = SamEmbedderFuser.HiddenDim(fused, 4.0);
        w["idx_emb"] = Filled(rng, 2.0, 3, fused);
        for (int i = 0; i < widths.Length; i++)
        {
            w[$"projection_nets.{i}.0.weight"] = Filled(rng, 1.0, widths[i]);
            w[$"projection_nets.{i}.0.bias"] = Filled(rng, 1.0, widths[i]);
            w[$"projection_nets.{i}.1.w1.weight"] = Filled(rng, 0.5, hidden, widths[i]);
            w[$"projection_nets.{i}.1.w3.weight"] = Filled(rng, 0.5, hidden, widths[i]);
            w[$"projection_nets.{i}.1.w2.weight"] = Filled(rng, 0.5, fused, hidden);
        }

        return w;
    }

    private static float[] Random01(Random rng, int count)
    {
        float[] p = new float[count];
        for (int i = 0; i < p.Length; i++) p[i] = (float)rng.NextDouble();
        return p;
    }

    private static SamPointmapInput Points(Random rng) => new(Random01(rng, 3 * PointSide * PointSide), PointSide, PointSide);

    private static SamConditionInputs Inputs(Random rng, bool pointmaps) => new(
        Random01(rng, 3 * Size * Size), Random01(rng, Size * Size), Random01(rng, 3 * Size * Size), Random01(rng, Size * Size),
        pointmaps ? Points(rng) : null, pointmaps ? Points(rng) : null);

    [Fact]
    public void Structure_ConcatenatesTheSixKeywordsInEmbedderOrder()
    {
        Random rng = new(51);
        Dictionary<string, Tensor> w = Section(rng, withPointmap: true);
        using Owned owned = new(w.Values);
        using CpuBackend backend = new();
        using SamStageCondition condition = new(SamStage.Structure, w, Dino, Patch);
        SamConditionInputs inputs = Inputs(rng, pointmaps: true);

        using Tensor built = condition.Build(backend, inputs);

        // 4 DINO keywords of 1 + 4 tokens, then two pointmaps of 4 windows: 28 rows of the fused width.
        Assert.Equal(new TensorShape(1, 4 * (1 + Dino.NumPatches) + 2 * Patch.Windows, Dino.HiddenSize), built.Shape);

        // The same condition composed by hand from the parts, in the order the YAML lists the keywords.
        using SamDinoEmbedder image = new(Dino, w, "module_list.0.backbone.");
        using SamDinoEmbedder mask = new(Dino, w, "module_list.1.backbone.");
        Dictionary<string, Tensor> pointWeights = w.Where(kv => kv.Key.StartsWith("module_list.2.", StringComparison.Ordinal)).ToDictionary(kv => kv.Key["module_list.2.".Length..], kv => kv.Value);
        SamPointPatchEmbed points = new(Patch, pointWeights);
        List<Tensor> parts = [];
        try
        {
            parts.Add(Flat(image.StructureTokens(backend, inputs.CroppedImage)));
            parts.Add(Flat(image.StructureTokens(backend, inputs.FullImage)));
            parts.Add(Flat(mask.StructureTokensOfMask(backend, inputs.CroppedMask)));
            parts.Add(Flat(mask.StructureTokensOfMask(backend, inputs.FullMask)));
            parts.Add(points.Embed(backend, inputs.CroppedPointmap!.Planar, PointSide, PointSide));
            parts.Add(points.Embed(backend, inputs.FullPointmap!.Planar, PointSide, PointSide));
            SamFuserEmbedder[] embedders =
            [
                new(Dino.HiddenSize, [("image", "cropped"), ("rgb_image", "full")]),
                new(Dino.HiddenSize, [("mask", "cropped"), ("rgb_image_mask", "full")]),
                new(Patch.EmbedDim, [("pointmap", "cropped"), ("rgb_pointmap", "full")]),
            ];
            string[] keys = ["image", "rgb_image", "mask", "rgb_image_mask", "pointmap", "rgb_pointmap"];
            Dictionary<string, Tensor> byKey = keys.Select((k, i) => (k, i)).ToDictionary(x => x.k, x => parts[x.i]);
            using Tensor expected = new SamEmbedderFuser(embedders, Dino.HiddenSize, w).Fuse(backend, byKey);

            ReadOnlySpan<float> got = built.AsReadOnlySpan<float>(), want = expected.AsReadOnlySpan<float>();
            Assert.Equal(want.Length, got.Length);
            for (int i = 0; i < want.Length; i++)
                Assert.True(Math.Abs(want[i] - got[i]) < 1e-4f * (1 + Math.Abs(want[i])), $"row element {i}: expected {want[i]} got {got[i]}");
        }
        finally
        {
            foreach (Tensor part in parts) part.Dispose();
        }
    }

    /// <summary>A <c>[1, n, d]</c> tensor as an independent <c>[n, d]</c> one.</summary>
    private static Tensor Flat(Tensor batched)
    {
        using (batched)
        {
            Tensor rows = new(new TensorShape(batched.Shape[1], batched.Shape[2]), DType.F32);
            batched.AsReadOnlySpan<float>().CopyTo(rows.AsSpan<float>());
            return rows;
        }
    }

    [Fact]
    public void Slat_NeverReadsThePointmapEmbedderAndKeepsTheRegisters()
    {
        Random rng = new(52);
        Dictionary<string, Tensor> w = Section(rng, withPointmap: false);
        using Owned owned = new(w.Values);
        using CpuBackend backend = new();
        Assert.DoesNotContain(w.Keys, k => k.StartsWith("module_list.2.", StringComparison.Ordinal));
        using SamStageCondition condition = new(SamStage.Slat, w, Dino, Patch);

        using Tensor built = condition.Build(backend, Inputs(rng, pointmaps: false));

        // Four keywords of 1 + 4 registers + 4 patches.
        Assert.Equal(new TensorShape(1, 4 * Dino.SequenceLength, Dino.HiddenSize), built.Shape);
    }

    [Fact]
    public void Structure_NeedsThePointmapsUnlessTheyAreDropped()
    {
        Random rng = new(53);
        Dictionary<string, Tensor> w = Section(rng, withPointmap: true);
        using Owned owned = new(w.Values);
        using CpuBackend backend = new();
        using SamStageCondition condition = new(SamStage.Structure, w, Dino, Patch);
        SamConditionInputs without = Inputs(rng, pointmaps: false);

        Assert.Throws<ArgumentException>(() => condition.Build(backend, without));
        HashSet<string> dropBoth = ["pointmap", "rgb_pointmap"];
        using Tensor objectOnly = condition.Build(backend, without, dropBoth);

        // The same images with real pointmaps and the same drop: dropped tokens are zero whatever the pointmap held.
        SamConditionInputs with = without with { CroppedPointmap = Points(rng), FullPointmap = Points(rng) };
        using Tensor dropped = condition.Build(backend, with, dropBoth);
        Assert.Equal(objectOnly.Shape, dropped.Shape);
        ReadOnlySpan<float> a = objectOnly.AsReadOnlySpan<float>(), b = dropped.AsReadOnlySpan<float>();
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i]);

        // The dropped rows (the last 2 x 4) are exactly zero.
        int fused = Dino.HiddenSize, droppedRows = 2 * Patch.Windows;
        foreach (float v in a[^(droppedRows * fused)..])
            Assert.Equal(0f, v);
    }
}
