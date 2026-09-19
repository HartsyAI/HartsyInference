using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Prompting;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the tensor form of ComfyUI's emphasis blend against <see cref="EmphasisMath.ApplyComfy"/>, the host
/// implementation the CLIP encoders have always used. The two must agree token for token, because the only difference
/// between a CLIP family and a T5/LLM family under <see cref="PromptWeightingMode.ComfyBlend"/> is where the
/// conditioning came from, never what is done to it.</summary>
public sealed class ComfyBlendWeightingTests
{
    private const int Seq = 5;
    private const int Dim = 4;

    private static Tensor Filled(Func<int, float> value)
    {
        Tensor tensor = new Tensor(new TensorShape(1, Seq, Dim), DType.F32);
        Span<float> span = tensor.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = value(i);
        }
        return tensor;
    }

    private static float[] ReferenceBlend(Tensor hidden, Tensor empty, float[] weights)
    {
        float[] reference = hidden.AsReadOnlySpan<float>().ToArray();
        EmphasisMath.ApplyComfy(reference, empty.AsReadOnlySpan<float>(), weights, Seq, Dim);
        return reference;
    }

    /// <summary>The invariant that catches most mistakes: all-ones must not touch a byte, which is why the blend is
    /// short-circuited entirely rather than run as a multiply by one.</summary>
    [Fact]
    public void AllOnesLeavesTheConditioningUntouched()
    {
        IBackend backend = new CpuBackend();
        using Tensor hidden = Filled(i => (i * 0.37f) - 2f);
        using Tensor empty = Filled(i => 0.11f * i);
        float[] weights = [1f, 1f, 1f, 1f, 1f];
        Assert.Null(ComfyBlend.Apply(backend, hidden, empty, weights));
    }

    [Fact]
    public void TheBlendMatchesTheComfyReferenceTokenForToken()
    {
        IBackend backend = new CpuBackend();
        using Tensor hidden = Filled(i => (i * 0.37f) - 2f);
        using Tensor empty = Filled(i => 0.11f * i);
        float[] weights = [1f, 1.5f, 0.5f, 1f, 2f];
        float[] reference = ReferenceBlend(hidden, empty, weights);
        using Tensor? blended = ComfyBlend.Apply(backend, hidden, empty, weights);
        Assert.NotNull(blended);
        ReadOnlySpan<float> got = blended!.AsReadOnlySpan<float>();
        for (int i = 0; i < reference.Length; i++)
        {
            Assert.Equal(reference[i], got[i], 5);
        }
    }

    /// <summary>Within a partly-weighted prompt the untouched tokens must come back bit-for-bit, not merely close:
    /// the <c>z + (z − zEmpty)·(w − 1)</c> form exists so a weight of exactly 1 multiplies the delta by exactly 0.</summary>
    [Fact]
    public void RowsAtWeightOneAreBitIdenticalInsideAPartlyWeightedPrompt()
    {
        IBackend backend = new CpuBackend();
        using Tensor hidden = Filled(i => (i * 0.37f) - 2f);
        using Tensor empty = Filled(i => 0.11f * i);
        float[] weights = [1f, 1.5f, 1f, 1f, 0.25f];
        float[] source = hidden.AsReadOnlySpan<float>().ToArray();
        using Tensor? blended = ComfyBlend.Apply(backend, hidden, empty, weights);
        Assert.NotNull(blended);
        ReadOnlySpan<float> got = blended!.AsReadOnlySpan<float>();
        for (int row = 0; row < Seq; row++)
        {
            if (weights[row] != 1f)
            {
                continue;
            }
            for (int d = 0; d < Dim; d++)
            {
                Assert.True(source[(row * Dim) + d] == got[(row * Dim) + d],
                    $"row {row} drifted at {d}: {source[(row * Dim) + d]} != {got[(row * Dim) + d]}");
            }
        }
    }

    /// <summary>A zero weight collapses the token onto the empty-prompt baseline — the reference's degenerate case,
    /// and the cheapest check that the interpolation runs in the right direction.</summary>
    [Fact]
    public void AZeroWeightCollapsesTheTokenOntoTheEmptyBaseline()
    {
        IBackend backend = new CpuBackend();
        using Tensor hidden = Filled(i => i + 1f);
        using Tensor empty = Filled(i => -1f);
        float[] weights = [0f, 1f, 1f, 1f, 1f];
        using Tensor? blended = ComfyBlend.Apply(backend, hidden, empty, weights);
        Assert.NotNull(blended);
        ReadOnlySpan<float> got = blended!.AsReadOnlySpan<float>();
        for (int d = 0; d < Dim; d++)
        {
            Assert.Equal(-1f, got[d], 5);
        }
    }

    [Fact]
    public void TheSourceConditioningIsLeftUnmodified()
    {
        IBackend backend = new CpuBackend();
        using Tensor hidden = Filled(i => (i * 0.37f) - 2f);
        using Tensor empty = Filled(i => 0.11f * i);
        float[] before = hidden.AsReadOnlySpan<float>().ToArray();
        float[] weights = [2f, 1f, 1f, 1f, 1f];
        using Tensor? blended = ComfyBlend.Apply(backend, hidden, empty, weights);
        Assert.NotNull(blended);
        Assert.Equal(before, hidden.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void AMismatchedEmptyBaselineIsRefusedRatherThanBroadcast()
    {
        IBackend backend = new CpuBackend();
        using Tensor hidden = Filled(i => i);
        using Tensor empty = new Tensor(new TensorShape(1, Seq + 1, Dim), DType.F32);
        float[] weights = [1f, 2f, 1f, 1f, 1f];
        Assert.Throws<ArgumentException>(() =>
        {
            ComfyBlend.Apply(backend, hidden, empty, weights);
        });
    }

    [Fact]
    public void AWeightCountThatDoesNotMatchTheSequenceIsRefused()
    {
        IBackend backend = new CpuBackend();
        using Tensor hidden = Filled(i => i);
        using Tensor empty = Filled(i => 0f);
        float[] weights = [1f, 2f];
        Assert.Throws<ArgumentException>(() =>
        {
            ComfyBlend.Apply(backend, hidden, empty, weights);
        });
    }
}
