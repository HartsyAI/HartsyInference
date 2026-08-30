using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Projects dense PDD AdaLN deltas onto a pruned checkpoint's fitted curve basis in F64.</summary>
public static unsafe class MiniMaxH3PddPrunedRebaser
{
    /// <summary>Rewrites every block AdaLN LoRA as a weight diff plus mandatory DC-bias diff.</summary>
    public static MiniMaxH3PddRebaseResult Rebase(IReadOnlyList<LoraLayer> layers,
        MiniMaxH3PddAffineBasis basis, bool requireCompleteAdapter = true)
    {
        ArgumentNullException.ThrowIfNull(layers);
        ArgumentNullException.ThrowIfNull(basis);
        if (basis.Intercept.DType != DType.F32 || basis.Projection.DType != DType.F32
            || basis.Intercept.Shape.Rank != 1 || basis.Projection.Shape.Rank != 2
            || basis.Projection.Shape[0] != basis.Intercept.Shape[0])
        {
            throw new HartsyInferenceException(
                $"Invalid PDD affine basis: c={basis.Intercept.Shape}/{basis.Intercept.DType}, "
                + $"V={basis.Projection.Shape}/{basis.Projection.DType}.");
        }

        List<LoraLayer> remaining = new(layers.Count);
        List<LoraFullWeightDiff> diffs = [];
        List<Tensor> owned = [];
        int rebased = 0;
        try
        {
            foreach (LoraLayer layer in layers)
            {
                if (!layer.TargetKey.EndsWith(".adaln_proj.linear.weight", StringComparison.Ordinal))
                {
                    remaining.Add(layer);
                    continue;
                }
                (Tensor weightDiff, Tensor biasDiff) = RebaseLayer(layer, basis);
                owned.Add(weightDiff);
                owned.Add(biasDiff);
                diffs.Add(new LoraFullWeightDiff
                {
                    TargetKey = layer.TargetKey,
                    Target = LoraTarget.Transformer,
                    Diff = weightDiff,
                    IsBias = false,
                });
                diffs.Add(new LoraFullWeightDiff
                {
                    TargetKey = layer.TargetKey[..^".weight".Length] + ".bias",
                    Target = LoraTarget.Transformer,
                    Diff = biasDiff,
                    IsBias = true,
                });
                rebased++;
            }
            if (requireCompleteAdapter && rebased != 50)
            {
                throw new HartsyInferenceException(
                    $"A complete MiniMax-H3 PDD adapter must rebase 50 block AdaLN targets; got {rebased}.");
            }
            return new MiniMaxH3PddRebaseResult(remaining, diffs, owned);
        }
        catch
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
            throw;
        }
    }

    private static (Tensor Weight, Tensor Bias) RebaseLayer(LoraLayer layer, MiniMaxH3PddAffineBasis basis)
    {
        int rank = layer.Rank;
        int dense = (int)basis.Intercept.Shape[0];
        int curve = (int)basis.Projection.Shape[1];
        int output = (int)layer.LoraUp.Shape[0];
        if (layer.LoraDown.Shape.Rank != 2 || layer.LoraUp.Shape.Rank != 2
            || layer.LoraDown.Shape[0] != rank || layer.LoraDown.Shape[1] != dense
            || layer.LoraUp.Shape[1] != rank)
        {
            throw new HartsyInferenceException(
                $"Dense PDD AdaLN '{layer.TargetKey}' does not match basis width {dense}: "
                + $"down={layer.LoraDown.Shape}, up={layer.LoraUp.Shape}, rank={rank}.");
        }

        double scale = layer.Alpha / rank;
        double[] projectedDown = new double[rank * curve];
        double[] interceptDown = new double[rank];
        for (int r = 0; r < rank; r++)
        {
            for (int d = 0; d < dense; d++)
            {
                double a = MiniMaxH3TensorReader.Read(layer.LoraDown, (long)r * dense + d);
                interceptDown[r] += a * MiniMaxH3TensorReader.Read(basis.Intercept, d);
                for (int k = 0; k < curve; k++)
                {
                    projectedDown[r * curve + k] += a
                        * MiniMaxH3TensorReader.Read(basis.Projection, (long)d * curve + k);
                }
            }
        }

        Tensor weightDiff = new Tensor(new TensorShape(output, curve), DType.F32);
        Tensor biasDiff = new Tensor(new TensorShape(output), DType.F32);
        float* weightPointer = (float*)weightDiff.DataPointer;
        float* biasPointer = (float*)biasDiff.DataPointer;
        try
        {
            Parallel.For(0, output, outputRow =>
            {
                double bias = 0.0;
                for (int r = 0; r < rank; r++)
                {
                    double b = MiniMaxH3TensorReader.Read(layer.LoraUp, (long)outputRow * rank + r);
                    bias += b * interceptDown[r];
                }
                biasPointer[outputRow] = (float)(scale * bias);
                for (int k = 0; k < curve; k++)
                {
                    double value = 0.0;
                    for (int r = 0; r < rank; r++)
                    {
                        double b = MiniMaxH3TensorReader.Read(layer.LoraUp, (long)outputRow * rank + r);
                        value += b * projectedDown[r * curve + k];
                    }
                    weightPointer[(long)outputRow * curve + k] = (float)(scale * value);
                }
            });
            return (weightDiff, biasDiff);
        }
        catch
        {
            weightDiff.Dispose();
            biasDiff.Dispose();
            throw;
        }
    }
}
