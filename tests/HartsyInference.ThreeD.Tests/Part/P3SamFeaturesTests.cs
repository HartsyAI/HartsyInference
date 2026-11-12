using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Part;
using HartsyInference.ThreeD.Models.Utonia;
using HartsyInference.ThreeD.Tests.Utonia;

namespace HartsyInference.ThreeD.Tests.Part;

/// <summary>Unit lane: Sonata's input transform for P3-SAM and the per-point feature path, on a tiny random encoder.</summary>
public sealed class P3SamFeaturesTests
{
    [Fact]
    public void GridSample_CentersZStartsAtZeroKeepsOnePointPerVoxelAndMapsEveryPointBack()
    {
        // Two points in the same 0.005 voxel, one far away. Centre shift: x and y centred on the bounding box, z from the lowest point.
        double[] coord = [1.0, 2.0, 3.0, 1.001, 2.001, 3.001, 1.5, 2.5, 3.5];

        P3SamGridSample.Result result = P3SamGridSample.Run(coord, 3, P3SamVoxelPick.First);

        Assert.Equal(2, result.Selected.Length);
        Assert.Equal(result.Inverse[0], result.Inverse[1]);
        Assert.NotEqual(result.Inverse[0], result.Inverse[2]);
        Assert.Equal(0.0, result.Shifted[2], 12); // lowest z -> 0
        Assert.Equal(-0.25, result.Shifted[0], 12); // x: 1.0 - (1.0 + 1.5) / 2
        Assert.Equal(0.25, result.Shifted[6], 12);
        Assert.Equal(0, result.GridCoord.Min());
        Assert.Contains(result.Selected[result.Inverse[0]], new[] { 0, 1 });
    }

    [Fact]
    public void GridSample_RandomPickStaysInsideTheVoxelAndFollowsTheSeed()
    {
        double[] coord = [0.0, 0.0, 0.0, 0.0005, 0.0005, 0.0005, 0.001, 0.001, 0.001, 0.9, 0.9, 0.9];

        int[] first = P3SamGridSample.Run(coord, 4, P3SamVoxelPick.Random, new Random(1)).Selected;
        int[] again = P3SamGridSample.Run(coord, 4, P3SamVoxelPick.Random, new Random(1)).Selected;

        Assert.Equal(first, again);
        P3SamGridSample.Result result = P3SamGridSample.Run(coord, 4, P3SamVoxelPick.Random, new Random(7));
        for (int row = 0; row < result.Selected.Length; row++)
            Assert.Equal(row, result.Inverse[result.Selected[row]]);
    }

    private static Dictionary<string, Tensor> TinyState(UtoniaConfig config, int gathered)
    {
        Dictionary<string, Tensor> byName = UtoniaTestData.RandomTensors(config, 5);
        Dictionary<string, Tensor> state = [];
        foreach (UtoniaParameterSpec spec in UtoniaCheckpointKeys.Parameters(config))
            state["sonata." + spec.TorchKey] = byName[spec.CSharpName];
        Random rng = new(2);
        void Layer(string name, int outDim, int inDim)
        {
            state[$"{name}.weight"] = UtoniaTestData.Make([outDim, inDim], [.. Enumerable.Range(0, outDim * inDim).Select(_ => (float)(rng.NextDouble() * 0.4 - 0.2))]);
            state[$"{name}.bias"] = UtoniaTestData.Make([outDim], [.. Enumerable.Range(0, outDim).Select(_ => (float)(rng.NextDouble() * 0.2 - 0.1))]);
        }

        Layer("mlp.0", 8, gathered);
        Layer("mlp.2", 8, 8);
        Layer("mlp.4", 8, 8);
        return state;
    }

    private static UtoniaConfig TinyNoRope() =>
        UtoniaTestData.TinyConfig() with { UseRope = false, EncChannels = [8, 16, 24], EncNumHeads = [2, 4, 6], LayerScaleInit = null };

    [Fact]
    public void Extract_GivesEveryPointTheFeaturesOfItsVoxelAndIsDeterministic()
    {
        UtoniaConfig config = TinyNoRope();
        Dictionary<string, Tensor> state = TinyState(config, 8 + 16 + 24);
        using CpuBackend backend = new();
        using P3SamFeatures features = new(backend, state, config);
        Random rng = new(4);
        int count = 60;
        double[] points = Enumerable.Range(0, count * 3).Select(_ => rng.NextDouble() * 0.2 - 0.1).ToArray();
        points[3] = points[0]; points[4] = points[1]; points[5] = points[2]; // point 1 sits on point 0, so they share a voxel whatever the grid offset
        float[] normals = Enumerable.Range(0, count * 3).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();

        float[] a = features.Extract(points, normals, count, P3SamVoxelPick.First);
        float[] b = features.Extract(points, normals, count, P3SamVoxelPick.First);

        Assert.Equal(count * 8, a.Length);
        Assert.Equal(a, b);
        Assert.All(a, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(a.AsSpan(0, 8).ToArray(), a.AsSpan(8, 8).ToArray()); // same voxel, same feature row
        Assert.NotEqual(a.AsSpan(0, 8).ToArray(), a.AsSpan(16, 8).ToArray());
    }

    [Fact]
    public void Checkpoint_RefusesAMissingSonataTensorAndIgnoresTheTrainingMaskToken()
    {
        UtoniaConfig config = TinyNoRope();
        Dictionary<string, Tensor> state = TinyState(config, 48);
        state["sonata.embedding.mask_token"] = UtoniaTestData.Make([1, 8], new float[8]);

        _ = P3SamCheckpoint.SonataWeights(state, config);

        state.Remove("sonata.enc.enc0.block0.attn.qkv.weight");
        Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => P3SamCheckpoint.SonataWeights(state, config));
    }
}
