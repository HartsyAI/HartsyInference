using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Prompting;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins <see cref="CondTokenWeights"/> against SwarmUI's <c>multiply_cond_by_token_weights</c>
/// (<c>SwarmText.py:256-271</c>) and the uniform fallback in <c>encode_leaves</c> (<c>:578-583</c>). The alignment is
/// the part that silently produces a plausible-but-wrong image: <c>pos = condLen − len(batch) + i</c> means a pipeline
/// that trimmed a template prefix off its encoder output has a NEGATIVE offset, and the leading weights must fall off
/// the front rather than shifting every remaining one.</summary>
public sealed class CondTokenWeightsTests
{
    private static Tensor Ramp(int rows, int dim, int batch = 1)
    {
        Tensor cond = new Tensor(new TensorShape(batch, rows, dim), DType.F32);
        Span<float> span = cond.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = i + 1;
        }
        return cond;
    }

    /// <summary>The invariant the whole mechanism rests on: weight 1 everywhere must not touch a single byte, so a
    /// plain prompt cannot drift just because the weighting path is now reachable.</summary>
    [Fact]
    public void AllOnesLeavesTheConditioningUntouched()
    {
        IBackend backend = new CpuBackend();
        using Tensor cond = Ramp(4, 3);
        float[] weights = [1f, 1f, 1f, 1f];
        Assert.Null(CondTokenWeights.ScaleRightAligned(backend, cond, weights));
    }

    [Fact]
    public void EachRowIsScaledByItsOwnTokenWeight()
    {
        IBackend backend = new CpuBackend();
        using Tensor cond = Ramp(3, 2);
        float[] weights = [1f, 2f, 0.5f];
        using Tensor? scaled = CondTokenWeights.ScaleRightAligned(backend, cond, weights);
        Assert.NotNull(scaled);
        ReadOnlySpan<float> got = scaled!.AsReadOnlySpan<float>();
        float[] expected = [1f, 2f, 6f, 8f, 2.5f, 3f];
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], got[i], 5);
        }
    }

    /// <summary>A trimmed template prefix makes <c>condLen − weights.Length</c> negative; the weights that land before
    /// row 0 are dropped and everything else keeps its absolute distance from the END of the sequence.</summary>
    [Fact]
    public void ANegativeOffsetDropsTheTrimmedPrefixWeights()
    {
        IBackend backend = new CpuBackend();
        using Tensor cond = Ramp(2, 1);
        // Four tokenizer positions, two surviving cond rows: offset -2, so only the last two weights apply.
        float[] weights = [9f, 9f, 2f, 3f];
        using Tensor? scaled = CondTokenWeights.ScaleRightAligned(backend, cond, weights);
        Assert.NotNull(scaled);
        ReadOnlySpan<float> got = scaled!.AsReadOnlySpan<float>();
        Assert.Equal(2f, got[0], 5);
        Assert.Equal(6f, got[1], 5);
    }

    /// <summary>The mirror case: a cond LONGER than the token batch (a padded or register-prefixed sequence) puts the
    /// weights at the end, never at row 0.</summary>
    [Fact]
    public void APositiveOffsetRightAlignsAgainstTheEndOfTheConditioning()
    {
        IBackend backend = new CpuBackend();
        using Tensor cond = Ramp(4, 1);
        float[] weights = [2f, 3f];
        using Tensor? scaled = CondTokenWeights.ScaleRightAligned(backend, cond, weights);
        Assert.NotNull(scaled);
        ReadOnlySpan<float> got = scaled!.AsReadOnlySpan<float>();
        Assert.Equal(1f, got[0], 5);
        Assert.Equal(2f, got[1], 5);
        Assert.Equal(6f, got[2], 5);
        Assert.Equal(12f, got[3], 5);
    }

    [Fact]
    public void WeightsFallingEntirelyOutsideTheConditioningChangeNothing()
    {
        IBackend backend = new CpuBackend();
        using Tensor cond = Ramp(2, 2);
        float[] weights = [2f, 3f, 1f, 1f];
        Assert.Null(CondTokenWeights.ScaleRightAligned(backend, cond, weights));
    }

    [Fact]
    public void ABatchedConditioningScalesTheSameRowInEverySlot()
    {
        IBackend backend = new CpuBackend();
        using Tensor cond = Ramp(2, 1, batch: 2);
        float[] weights = [1f, 2f];
        using Tensor? scaled = CondTokenWeights.ScaleRightAligned(backend, cond, weights);
        Assert.NotNull(scaled);
        ReadOnlySpan<float> got = scaled!.AsReadOnlySpan<float>();
        Assert.Equal(1f, got[0], 5);
        Assert.Equal(4f, got[1], 5);
        Assert.Equal(3f, got[2], 5);
        Assert.Equal(8f, got[3], 5);
    }

    /// <summary>The uniform fallback is the only path that touches <c>pooled_output</c>; the per-token path never
    /// does, because ComfyUI takes the pooled vector before its weighting loop.</summary>
    [Fact]
    public void AUniformWeightScalesThePooledVectorToo()
    {
        IBackend backend = new CpuBackend();
        using Tensor cond = Ramp(2, 2);
        using Tensor pooled = Ramp(1, 3);
        CondScaleResult result = CondTokenWeights.ScaleUniform(backend, cond, pooled, 0.5f);
        using Tensor? scaledCond = result.Cond;
        using Tensor? scaledPooled = result.Pooled;
        Assert.NotNull(scaledCond);
        Assert.NotNull(scaledPooled);
        Assert.Equal(0.5f, scaledCond!.AsReadOnlySpan<float>()[0], 5);
        Assert.Equal(0.5f, scaledPooled!.AsReadOnlySpan<float>()[0], 5);
        Assert.Equal(1.0f, scaledPooled.AsReadOnlySpan<float>()[1], 5);
    }

    /// <summary>SwarmUI reaches the uniform branch only when the per-token pass applied nothing; with rows in range it
    /// is the per-token result that must win, pooled untouched.</summary>
    [Fact]
    public void ApplyPrefersPerTokenScalingOverTheUniformFallback()
    {
        IBackend backend = new CpuBackend();
        using Tensor cond = Ramp(2, 1);
        using Tensor pooled = Ramp(1, 1);
        WeightedTokenSequence sequence = new WeightedTokenSequence([1, 2], [2f, 2f]) { UniformWeight = 2f };
        CondScaleResult result = CondTokenWeights.Apply(backend, cond, pooled, sequence);
        using Tensor? scaledCond = result.Cond;
        Assert.NotNull(scaledCond);
        Assert.Null(result.Pooled);
        Assert.Equal(2f, scaledCond!.AsReadOnlySpan<float>()[0], 5);
    }

    [Fact]
    public void ApplyFallsBackToTheUniformScaleWhenNoTokenPositionSurvived()
    {
        IBackend backend = new CpuBackend();
        using Tensor cond = Ramp(1, 1);
        using Tensor pooled = Ramp(1, 1);
        // Three weighted tokenizer positions, one surviving cond row whose own weight is 1: nothing lands in range.
        WeightedTokenSequence sequence = new WeightedTokenSequence([1, 2, 3], [2f, 2f, 1f]) { UniformWeight = 2f };
        CondScaleResult result = CondTokenWeights.Apply(backend, cond, pooled, sequence);
        using Tensor? scaledCond = result.Cond;
        using Tensor? scaledPooled = result.Pooled;
        Assert.NotNull(scaledCond);
        Assert.NotNull(scaledPooled);
        Assert.Equal(2f, scaledCond!.AsReadOnlySpan<float>()[0], 5);
        Assert.Equal(2f, scaledPooled!.AsReadOnlySpan<float>()[0], 5);
    }

    [Fact]
    public void ApplyDoesNothingForAnUnweightedSequence()
    {
        IBackend backend = new CpuBackend();
        using Tensor cond = Ramp(2, 2);
        using Tensor pooled = Ramp(1, 2);
        WeightedTokenSequence sequence = new WeightedTokenSequence([1, 2], [1f, 1f]) { UniformWeight = 1f };
        CondScaleResult result = CondTokenWeights.Apply(backend, cond, pooled, sequence);
        Assert.Null(result.Cond);
        Assert.Null(result.Pooled);
    }

    /// <summary>The input is never mutated: several pipelines cache conditioning under a token-id key that CondScale
    /// does not change, so an in-place scale would hand the next unweighted request the weighted tensor.</summary>
    [Fact]
    public void TheSourceConditioningIsLeftUnmodified()
    {
        IBackend backend = new CpuBackend();
        using Tensor cond = Ramp(2, 2);
        float[] before = cond.AsReadOnlySpan<float>().ToArray();
        float[] weights = [2f, 3f];
        using Tensor? scaled = CondTokenWeights.ScaleRightAligned(backend, cond, weights);
        Assert.NotNull(scaled);
        Assert.Equal(before, cond.AsReadOnlySpan<float>().ToArray());
    }

    /// <summary>The elementwise scale reads and writes <c>float*</c> for <c>ElementCount</c> elements whatever the
    /// dtype, so a half-precision tensor would have twice its own size written into it. That is a buffer overrun,
    /// not a wrong number, so it is refused by name rather than scaled.</summary>
    [Theory]
    [InlineData("cond")]
    [InlineData("pooled")]
    public void AHalfPrecisionTensorIsRefusedRatherThanOverrun(string which)
    {
        IBackend backend = new CpuBackend();
        bool condHalf = which == "cond";
        using Tensor cond = condHalf ? new Tensor(new TensorShape(2, 2), DType.F16) : Ramp(2, 2);
        using Tensor pooled = condHalf ? Ramp(1, 3) : new Tensor(new TensorShape(1, 3), DType.F16);
        ArgumentException ex = Assert.Throws<ArgumentException>(
            () => CondTokenWeights.ScaleUniform(backend, cond, pooled, 0.5f));
        Assert.Equal(which, ex.ParamName);
    }

    /// <summary>Why the guard above matters only to a direct caller. It is tempting to assume a uniform weight
    /// reaches <see cref="CondTokenWeights.ScaleUniform"/> through <c>Apply</c> and carries an unchecked pooled
    /// vector with it — it does not. <c>ScaleRightAligned</c> returns null only when no weight other than 1 lands
    /// inside the cond, and a uniform non-1 weight lands on every row, so the per-token path takes it and the
    /// pooled vector is never touched. This pins that routing, so a future change that makes the uniform fallback
    /// genuinely reachable fails here rather than silently widening what reaches the scale.</summary>
    [Fact]
    public void AUniformWeightTakesThePerTokenPathAndNeverSeesThePooledVector()
    {
        IBackend backend = new CpuBackend();
        using Tensor cond = Ramp(2, 2);
        using Tensor pooled = new Tensor(new TensorShape(1, 3), DType.F16);
        WeightedTokenSequence sequence = new WeightedTokenSequence([1, 2], [1.5f, 1.5f]) { UniformWeight = 1.5f };
        CondScaleResult result = CondTokenWeights.Apply(backend, cond, pooled, sequence);
        using Tensor? scaledCond = result.Cond;
        Assert.NotNull(scaledCond);
        Assert.Null(result.Pooled);
        Assert.Equal(DType.F16, pooled.DType);
    }
}
