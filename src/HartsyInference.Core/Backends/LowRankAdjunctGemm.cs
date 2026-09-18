using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Adds a weight's <see cref="LowRankAdjunct"/> to a GEMM's result: <c>y += Σ scale·(x·Downᵀ)·Upᵀ</c>.</summary>
/// <remarks><para>Built from <see cref="IBackend.Linear"/>, <see cref="IBackend.Scale"/> and
/// <see cref="IBackend.Add"/> only, so one implementation serves CUDA, CPU and Vulkan and the arithmetic cannot drift
/// between them. The factors carry no adjunct of their own, so the inner <c>Linear</c> calls do not recurse.</para>
/// <para>Every GEMM entry a quantized weight can reach either calls this or refuses the weight by name. A silent miss
/// reads as "the LoRA looks weak", which is the failure the zero-match refusal in the LoRA applier exists to
/// prevent.</para></remarks>
public static class LowRankAdjunctGemm
{
    /// <summary>Accumulates every term of <paramref name="adjunct"/> into <paramref name="output"/>, which must already hold <c>x·Wᵀ (+ bias)</c>.</summary>
    /// <param name="input">The same activation the base GEMM consumed, <c>[M, inFeatures]</c> (leading batch dims fold into M).</param>
    public static void Accumulate(IBackend backend, Tensor output, Tensor input, LowRankAdjunct adjunct)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(adjunct);

        long inFeatures = adjunct.InFeatures;
        long outFeatures = adjunct.OutFeatures;
        long rows = input.ElementCount / inFeatures;
        if (rows * inFeatures != input.ElementCount)
        {
            throw new HartsyInferenceException(
                $"LoRA adjunct expects an activation whose last axis is {inFeatures}, got {input.Shape}.");
        }
        if (output.ElementCount != rows * outFeatures)
        {
            throw new HartsyInferenceException(
                $"LoRA adjunct expects a [{rows}, {outFeatures}] result, got {output.Shape}.");
        }

        TensorShape deltaShape = new(rows, outFeatures);
        foreach (LowRankAdjunctTerm term in adjunct.Terms)
        {
            if (term.Up is null)
            {
                using Tensor fullDelta = new(deltaShape, output.DType);
                backend.Linear(fullDelta, input, term.Down, null);
                backend.Scale(fullDelta, fullDelta, term.Scale);
                backend.Add(output, output, fullDelta);
                continue;
            }
            // The strength folds on the [M, rank] intermediate, not on the [M, outFeatures] result: same value, and
            // at LoRA ranks that is two orders of magnitude less traffic.
            using Tensor projected = new(new TensorShape(rows, term.Down.Shape[0]), output.DType);
            backend.Linear(projected, input, term.Down, null);
            backend.Scale(projected, projected, term.Scale);
            using Tensor delta = new(deltaShape, output.DType);
            backend.Linear(delta, projected, term.Up, null);
            backend.Add(output, output, delta);
        }
    }

    /// <summary>Refuses an operand carrying an adjunct at an op that cannot apply one, naming the op so the miss is an error rather than a weaker-looking LoRA.</summary>
    public static void RefuseAdjunct(Tensor operand, string operation)
    {
        if (operand.LowRankAdjunct is not null)
        {
            throw new NotSupportedException(
                $"{operation} was handed a weight carrying a LoRA adjunct, which it cannot apply. Route this weight "
                + "through Linear/LinearWeightRows, or merge the LoRA into a dequantized copy of it.");
        }
    }
}
