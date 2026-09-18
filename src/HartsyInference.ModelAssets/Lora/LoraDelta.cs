using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>One LoRA layer's weight-space delta, independent of which decomposition produced it. A merge asks for <see cref="ComputeF32"/> and adds <c>strength · <see cref="Scale"/> · ΔW</c>; nothing downstream needs to know whether the file stored a low-rank pair, a Hadamard product or a Kronecker product.</summary>
/// <remarks>The delta is always materialized as a rank-2 <c>[<see cref="OutFeatures"/>, <see cref="InFeatures"/>]</c>
/// matrix. A conv adapter's trailing kernel axes are folded into the columns — the same flattening ComfyUI's
/// <c>calculate_weight</c> does with <c>flatten(start_dim=1)</c> before reshaping onto the weight — so a rank-4 target
/// is a reshape of this matrix rather than a separate code path.</remarks>
public abstract class LoraDelta
{
    /// <summary>Which decomposition this delta came from.</summary>
    public abstract LoraVariant Variant { get; }

    /// <summary>Rows of the delta: the target weight's output dimension.</summary>
    public abstract long OutFeatures { get; }

    /// <summary>Columns of the flattened delta: the target weight's input dimension times its kernel footprint.</summary>
    public abstract long InFeatures { get; }

    /// <summary>The decomposition's intrinsic scale (<c>alpha / rank</c>). A merge multiplies by this and the user strength together, once, so the result matches ComfyUI's <c>(strength · alpha) · lora_diff</c> rounding for rounding.</summary>
    public abstract float Scale { get; }

    /// <summary>DoRA magnitude vector when the file carries one, else null. Orthogonal to the decomposition — LoHa and LoKr adapters ship it too — and applied by <see cref="LoraDoraDecompose"/>, never by <see cref="ComputeF32"/>, because rescaling the decomposed weight needs the base weight this delta knows nothing about.</summary>
    public Tensor? DoraScale { get; init; }

    /// <summary>Materializes the UNSCALED delta as a freshly allocated F32 <c>[OutFeatures, InFeatures]</c> tensor. The caller owns and disposes it.</summary>
    public abstract Tensor ComputeF32(IBackend backend);

    /// <summary>Whether this delta is the shape of <paramref name="baseWeight"/> — rows equal and the base's remaining axes folding to <see cref="InFeatures"/>, so a rank-4 conv weight matches its flattened delta.</summary>
    public bool MatchesShape(Tensor baseWeight)
    {
        if (baseWeight.Shape.Rank < 2 || baseWeight.Shape[0] != OutFeatures)
        {
            return false;
        }
        return baseWeight.Shape.ElementCount / baseWeight.Shape[0] == InFeatures;
    }

    /// <summary>Returns <paramref name="tensor"/> as a rank-2 view with its trailing axes folded into the columns, or the tensor itself when it is already rank-2. A view is borrowed: dispose it only when it is not the input.</summary>
    protected static Tensor FlattenFromDim1(Tensor tensor)
    {
        if (tensor.Shape.Rank == 2)
        {
            return tensor;
        }
        if (tensor.Shape.Rank < 2)
        {
            throw new HartsyInferenceException($"LoRA decomposition matrix must be at least rank-2, got {tensor.Shape}.");
        }
        return tensor.Reshape(new TensorShape(tensor.Shape[0], tensor.Shape.ElementCount / tensor.Shape[0]));
    }

    /// <summary>Allocates an F32 <c>[rows, columns]</c> result and fills it with <c>a @ b</c>.</summary>
    protected static Tensor MatMulF32(IBackend backend, Tensor a, Tensor b)
    {
        Tensor result = new(new TensorShape(a.Shape[0], b.Shape[1]), DType.F32);
        try
        {
            backend.MatMul(result, a, b);
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    /// <summary>Rebuilds a LyCORIS CP/Tucker factor: <c>einsum("i j k l, j r, i p -&gt; p r k l", core, right, left)</c>, returned already flattened to <c>[p, r·k·l]</c>. Host math — this runs once per layer on the load path.</summary>
    protected static Tensor TuckerRebuildF32(Tensor core, Tensor left, Tensor right)
    {
        if (core.Shape.Rank != 4)
            throw new HartsyInferenceException($"LyCORIS Tucker core must be rank-4, got {core.Shape}.");
        if (left.Shape.Rank != 2 || right.Shape.Rank != 2)
            throw new HartsyInferenceException(
                $"LyCORIS Tucker factors must be rank-2, got {left.Shape} and {right.Shape}.");

        long i = core.Shape[0], j = core.Shape[1], kh = core.Shape[2], kw = core.Shape[3];
        if (left.Shape[0] != i)
            throw new HartsyInferenceException(
                $"LyCORIS Tucker left factor {left.Shape} does not match core leading dim {i}.");
        if (right.Shape[0] != j)
            throw new HartsyInferenceException(
                $"LyCORIS Tucker right factor {right.Shape} does not match core second dim {j}.");
        long p = left.Shape[1], r = right.Shape[1];

        using Tensor coreF32 = core.CastTo(DType.F32);
        using Tensor leftF32 = left.CastTo(DType.F32);
        using Tensor rightF32 = right.CastTo(DType.F32);
        long spatial = kh * kw;
        Tensor result = new(new TensorShape(p, r * spatial), DType.F32);
        try
        {
            ReadOnlySpan<float> coreData = coreF32.AsReadOnlySpan<float>();
            ReadOnlySpan<float> leftData = leftF32.AsReadOnlySpan<float>();
            ReadOnlySpan<float> rightData = rightF32.AsReadOnlySpan<float>();
            Span<float> output = result.AsSpan<float>();
            output.Clear();
            for (long ii = 0; ii < i; ii++)
            {
                for (long jj = 0; jj < j; jj++)
                {
                    long coreBase = (ii * j + jj) * spatial;
                    for (long rr = 0; rr < r; rr++)
                    {
                        float rightValue = rightData[(int)(jj * r + rr)];
                        if (rightValue == 0.0f)
                        {
                            continue;
                        }
                        for (long pp = 0; pp < p; pp++)
                        {
                            float weight = leftData[(int)(ii * p + pp)] * rightValue;
                            if (weight == 0.0f)
                            {
                                continue;
                            }
                            long outBase = pp * r * spatial + rr * spatial;
                            for (long s = 0; s < spatial; s++)
                            {
                                output[(int)(outBase + s)] += coreData[(int)(coreBase + s)] * weight;
                            }
                        }
                    }
                }
            }
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }
}
