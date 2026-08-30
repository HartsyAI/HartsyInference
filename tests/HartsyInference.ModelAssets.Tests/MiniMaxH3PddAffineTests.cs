using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.MiniMaxH3;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class MiniMaxH3PddAffineTests
{
    [Fact]
    public void Fit_RecoversAnExactDenseTimeCurveInF64()
    {
        Dictionary<string, Tensor> full = CreateSmallTimeEmbedder();
        using Tensor table = CreateExactTable(33);
        Dictionary<string, Tensor> pruned = new() { ["adaln_t_table"] = table };
        try
        {
            using MiniMaxH3PddAffineBasis basis = MiniMaxH3PddAffineFitter.Fit(full, pruned,
                maxResidual: 1e-7, requirePublishedShape: false);

            Assert.True(basis.RelativeResidual < 1e-7, $"residual={basis.RelativeResidual:E6}");
            Assert.Equal(0.0f, basis.Intercept.AsSpan<float>()[0], 4);
            Assert.Equal(1.0f, basis.Projection.AsSpan<float>()[0], 4);
        }
        finally
        {
            foreach (Tensor tensor in full.Values) tensor.Dispose();
        }
    }

    [Fact]
    public void Rebase_EmitsWeightAndMandatoryDcBiasDiffs()
    {
        using Tensor intercept = Filled(new TensorShape(1), [5]);
        using Tensor projection = Filled(new TensorShape(1, 1), [7]);
        using MiniMaxH3PddAffineBasis basis = new MiniMaxH3PddAffineBasis(intercept, projection, 0.0);
        using Tensor down = Filled(new TensorShape(1, 1), [2]);
        using Tensor up = Filled(new TensorShape(2, 1), [3, 4]);
        LoraLayer layer = new LoraLayer
        {
            TargetKey = "blocks.0.adaln_proj.linear.weight",
            Target = LoraTarget.Transformer,
            LoraDown = down,
            LoraUp = up,
            Alpha = 1.0f,
            Rank = 1,
            Variant = LoraVariant.StandardLora,
        };

        using MiniMaxH3PddRebaseResult result = MiniMaxH3PddPrunedRebaser.Rebase([layer], basis,
            requireCompleteAdapter: false);

        Assert.Empty(result.Layers);
        Assert.Equal(2, result.FullWeightDiffs.Count);
        LoraFullWeightDiff weight = Assert.Single(result.FullWeightDiffs, diff => !diff.IsBias);
        LoraFullWeightDiff bias = Assert.Single(result.FullWeightDiffs, diff => diff.IsBias);
        Assert.Equal(new float[] { 42, 56 }, weight.Diff.AsSpan<float>().ToArray());
        Assert.Equal(new float[] { 30, 40 }, bias.Diff.AsSpan<float>().ToArray());
        Assert.EndsWith(".bias", bias.TargetKey, StringComparison.Ordinal);
    }

    private static Dictionary<string, Tensor> CreateSmallTimeEmbedder() => new()
    {
        ["time_embedder.proj_in.weight"] = Filled(new TensorShape(1, 2), [0.3f, -0.2f]),
        ["time_embedder.proj_in.bias"] = Filled(new TensorShape(1), [0.1f]),
        ["time_embedder.proj_out.weight"] = Filled(new TensorShape(1, 1), [0.7f]),
        ["time_embedder.proj_out.bias"] = Filled(new TensorShape(1), [-0.05f]),
    };

    private static Tensor CreateExactTable(int rows)
    {
        Tensor tensor = new Tensor(new TensorShape(rows, 1), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int row = 0; row < rows; row++)
        {
            double timestep = row / (double)(rows - 1);
            double embedding0 = Math.Cos(timestep);
            double embedding1 = Math.Sin(timestep);
            double hidden = Silu(0.3 * embedding0 - 0.2 * embedding1 + 0.1);
            values[row] = (float)Silu(0.7 * hidden - 0.05);
        }
        return tensor;
    }

    private static Tensor Filled(TensorShape shape, float[] values)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static double Silu(double value) => value / (1.0 + Math.Exp(-value));
}
