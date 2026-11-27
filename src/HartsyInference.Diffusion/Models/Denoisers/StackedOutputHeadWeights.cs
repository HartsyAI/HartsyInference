using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Builds one sampler step's effective output projection from a checkpoint's stacked heads: block 0 enters at
/// weight 1 and each offset block at its step weight. Runs on backend ops, once per step rather than per token.</summary>
public static class StackedOutputHeadWeights
{
    /// <summary>Returns the step's effective weight <c>[out, in]</c> and bias <c>[out]</c>, both owned by the caller.</summary>
    public static (Tensor Weight, Tensor Bias) Combine(IBackend backend, Tensor stackedWeight, Tensor stackedBias,
        int outRows, OutputHeadStep step)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(stackedWeight);
        ArgumentNullException.ThrowIfNull(stackedBias);
        ArgumentNullException.ThrowIfNull(step);
        if (outRows <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outRows), "The head must have at least one output row.");
        }

        int heads = step.Weights.Count;
        long stackedRows = (long)heads * outRows;
        if (stackedWeight.Shape.Rank != 2 || stackedWeight.Shape[0] != stackedRows || stackedWeight.DType != DType.F32)
        {
            throw new ArgumentException(
                $"Stacked output-head weight must be F32 [{stackedRows}, in]; got {stackedWeight.DType} {stackedWeight.Shape}.",
                nameof(stackedWeight));
        }

        if (stackedBias.Shape.Rank != 1 || stackedBias.Shape[0] != stackedRows || stackedBias.DType != DType.F32)
        {
            throw new ArgumentException(
                $"Stacked output-head bias must be F32 [{stackedRows}]; got {stackedBias.DType} {stackedBias.Shape}.",
                nameof(stackedBias));
        }

        int inDim = (int)stackedWeight.Shape[1];
        // Accumulators are owned tensors that only ever receive backend writes. A view is read, never written: on
        // GPU backends a write lands in the destination's cache, and disposing a view discards it.
        Tensor weight = new Tensor(new TensorShape(outRows, inDim), DType.F32);
        Tensor biasAcc = new Tensor(new TensorShape(outRows, 1), DType.F32);
        Tensor biasRows = stackedBias.Reshape(new TensorShape((int)stackedRows, 1));
        Tensor? bias = null;
        bool ok = false;
        try
        {
            backend.SliceRows(weight, stackedWeight, 0);
            backend.SliceRows(biasAcc, biasRows, 0);
            for (int n = 1; n < heads; n++)
            {
                float scale = (float)step.Weights[n];
                if (scale == 0f)
                {
                    continue;
                }

                AddScaledWeightBlock(backend, ref weight, stackedWeight, n * outRows, scale);
                AddScaledBiasBlock(backend, ref biasAcc, biasRows, n * outRows, scale);
            }

            bias = new Tensor(new TensorShape(outRows), DType.F32);
            using (Tensor flat = biasAcc.Reshape(new TensorShape(outRows)))
            {
                backend.Scale(bias, flat, 1f);
            }

            ok = true;
            return (weight, bias);
        }
        finally
        {
            biasRows.Dispose();
            biasAcc.Dispose();
            if (!ok)
            {
                weight.Dispose();
                bias?.Dispose();
            }
        }
    }

    /// <summary>Replaces <paramref name="target"/> with target + scale * the block of rows starting at
    /// <paramref name="rowOffset"/>.</summary>
    private static void AddScaledWeightBlock(IBackend backend, ref Tensor target, Tensor source, int rowOffset, float scale)
    {
        int rows = (int)target.Shape[0], columns = (int)target.Shape[1];
        Tensor block = new Tensor(new TensorShape(rows, columns), DType.F32);
        Tensor scaled = new Tensor(new TensorShape(rows, columns), DType.F32);
        Tensor next = new Tensor(new TensorShape(rows, columns), DType.F32);
        try
        {
            backend.SliceRows(block, source, rowOffset);
            backend.Scale(scaled, block, scale);
            backend.Add(next, target, scaled);
        }
        finally
        {
            block.Dispose();
            scaled.Dispose();
        }

        target.Dispose();
        target = next;
    }

    /// <summary>Replaces the owned <c>[out, 1]</c> bias accumulator with accumulator + scale * the block of stacked bias
    /// rows starting at <paramref name="rowOffset"/>.</summary>
    private static void AddScaledBiasBlock(IBackend backend, ref Tensor target, Tensor sourceRows, int rowOffset, float scale)
    {
        int rows = (int)target.Shape[0];
        Tensor block = new Tensor(new TensorShape(rows, 1), DType.F32);
        Tensor scaled = new Tensor(new TensorShape(rows, 1), DType.F32);
        Tensor next = new Tensor(new TensorShape(rows, 1), DType.F32);
        try
        {
            backend.SliceRows(block, sourceRows, rowOffset);
            backend.Scale(scaled, block, scale);
            backend.Add(next, target, scaled);
        }
        finally
        {
            block.Dispose();
            scaled.Dispose();
        }

        target.Dispose();
        target = next;
    }
}
