using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Idol;
using HartsyInference.ThreeD.Models.Idol.Budget;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Idol.Budget;

public sealed class BudgetAdapterTests
{
    private static readonly CpuBackend Backend = new();

    private static Dictionary<string, Tensor> Tensors(int seed)
    {
        Dictionary<string, Tensor> t = new(StringComparer.Ordinal);
        int s = seed * 1000;
        foreach ((string key, long[] shape) in BudgetAdapterWeights.ExpectedShapes(new IdolConfig()))
            t[key] = IdolModelTests.Random(new TensorShape(shape), s++, key.EndsWith("out_norm.weight", StringComparison.Ordinal) ? 0.1f : 0.05f);
        return t;
    }

    private static void Zero(Dictionary<string, Tensor> tensors, params string[] keys)
    {
        foreach (string key in keys)
            tensors[key].AsSpan<float>().Clear();
    }

    private static void Dispose(Dictionary<string, Tensor> tensors)
    {
        foreach (Tensor t in tensors.Values)
            t.Dispose();
    }

    [Theory]
    [InlineData(2, 3)]
    [InlineData(8, 2)]
    public void AMixerWithAZeroedLearnedBranchIsTheMeanOfItsChildren(int children, int parents)
    {
        Dictionary<string, Tensor> tensors = Tensors(1);
        string p = $"bp_mixers.{(children == 2 ? 2 : 4)}";
        Zero(tensors, $"{p}.up.weight", $"{p}.up.bias");
        try
        {
            BudgetAdapterWeights weights = BudgetAdapterWeights.FromTensors(new IdolConfig(), tensors);
            float[] input = new float[parents * children * 1024];
            Random rng = new(5);
            for (int i = 0; i < input.Length; i++)
                input[i] = (float)rng.NextDouble();

            float[] parent = new BudgetScaleMixer(weights, children).Apply(Backend, input, parents, 1024);

            for (int q = 0; q < parents; q++)
                for (int c = 0; c < 1024; c += 97)
                {
                    float mean = 0;
                    for (int s = 0; s < children; s++)
                        mean += input[(q * children + s) * 1024 + c];
                    Assert.Equal(mean / children, parent[q * 1024 + c], 5);
                }
        }
        finally
        {
            Dispose(tensors);
        }
    }

    [Theory]
    [InlineData(2)]
    [InlineData(8)]
    public void ARefineStageWithAZeroedOutputIsTheNearestRepeatOfItsParent(int children)
    {
        Dictionary<string, Tensor> tensors = Tensors(2);
        string p = $"bp_refiners.{(children == 2 ? 2 : 4)}";
        Zero(tensors, $"{p}.out.weight", $"{p}.out.bias");
        try
        {
            BudgetAdapterWeights weights = BudgetAdapterWeights.FromTensors(new IdolConfig(), tensors);
            int rows = 3;
            float[] parent = new float[rows * 16], cond = new float[rows * 1024], noisy = new float[rows * children * 16];
            Random rng = new(6);
            foreach (float[] a in new[] { parent, cond, noisy })
                for (int i = 0; i < a.Length; i++)
                    a[i] = (float)rng.NextDouble() - 0.5f;
            float[] offsets = children == 2 ? BudgetOctree.CellChildOffsets() : BudgetOctree.CubeChildOffsets();

            (float[] child, float[] hidden) = new BudgetRefineStage(weights, children).Forward(Backend, parent, rows, cond, 1024, 1, offsets, noisy, default);

            for (int r = 0; r < rows; r++)
                for (int j = 0; j < children; j++)
                    for (int c = 0; c < 16; c++)
                        Assert.Equal(parent[r * 16 + c], child[(r * children + j) * 16 + c], 6);
            Assert.Equal(rows * children * 128, hidden.Length);
            Assert.All(hidden, v => Assert.True(float.IsFinite(v)));
        }
        finally
        {
            Dispose(tensors);
        }
    }

    [Fact]
    public void ACondGroupServesItsRepeatedRowsAlike()
    {
        // The second stage of the s = 4 cascade gives all eight cells of a cube the cube's hidden state: eight rows of one group equal eight rows with the group's state written out.
        Dictionary<string, Tensor> tensors = Tensors(3);
        try
        {
            BudgetAdapterWeights weights = BudgetAdapterWeights.FromTensors(new IdolConfig(), tensors);
            BudgetRefineStage stage = new(weights, 2);
            int rows = 8;
            float[] parent = new float[rows * 16], group = new float[1024], expanded = new float[rows * 1024], noisy = new float[rows * 2 * 16], carry = new float[rows * 128];
            Random rng = new(7);
            foreach (float[] a in new[] { parent, group, noisy, carry })
                for (int i = 0; i < a.Length; i++)
                    a[i] = (float)rng.NextDouble() - 0.5f;
            for (int r = 0; r < rows; r++)
                group.CopyTo(expanded, r * 1024);

            (float[] a1, _) = stage.Forward(Backend, parent, rows, group, 1024, rows, BudgetOctree.CellChildOffsets(), noisy, carry);
            (float[] a2, _) = stage.Forward(Backend, parent, rows, expanded, 1024, 1, BudgetOctree.CellChildOffsets(), noisy, carry);

            Assert.Equal(a2, a1);
        }
        finally
        {
            Dispose(tensors);
        }
    }

    [Fact]
    public void AnAdapterFileWithAWrongShapeOrAMissingTensorIsRefusedByName()
    {
        Dictionary<string, Tensor> tensors = Tensors(4);
        string path = Path.Combine(Path.GetTempPath(), "budget-adapters-" + Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            tensors.Remove("bp_scale_emb", out Tensor? gone);
            gone?.Dispose();
            HartsyInference.ModelAssets.SafeTensors.SafeTensorsWriter.Save(path, tensors);

            HartsyInference.Core.Exceptions.HartsyInferenceException error = Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => BudgetAdapterWeights.Load(path));

            Assert.Contains("bp_scale_emb", error.Message);
        }
        finally
        {
            Dispose(tensors);
            if (File.Exists(path))
                File.Delete(path);
        }
    }
}
