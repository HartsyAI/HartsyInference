using Xunit;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>A tensor that does not fill a whole number of blocks.
/// <para>MiniMax-H3's <c>adaln_t_table</c> is <c>[8, 1025]</c> — 8200 elements, eight short of the 33rd Q4_K
/// block. Quantizing it sized the buffer by integer division (32 blocks) while the codec wrote 33, and the only
/// thing standing in the way was a <c>Debug.Assert</c>, which Release compiles out. The overrun corrupted the
/// heap and surfaced as an allocator abort in unrelated code much later.</para></summary>
public sealed class QuantBlockAlignmentTests
{
    /// <summary>H3's real shape, and the reason this is not hypothetical.</summary>
    [Fact]
    public void AMisalignedTensorIsKeptWideRatherThanQuantized()
    {
        using Tensor adaln = new Tensor(new TensorShape(8, 1025), DType.F32);
        Assert.NotEqual(0, adaln.Shape.ElementCount % DType.Q4_K.BlockElementCount);
        Assert.Equal(DType.F16, GgufQuantPolicy.Q4_K_S.ResolveTargetDType("adaln_t_table", adaln));
    }

    /// <summary>The aligned neighbour still quantizes, so the guard is not just refusing everything.</summary>
    [Fact]
    public void AnAlignedTensorStillTakesTheBackboneQuant()
    {
        using Tensor weight = new Tensor(new TensorShape(512, 2688), DType.F32);
        Assert.Equal(0, weight.Shape.ElementCount % DType.Q4_K.BlockElementCount);
        Assert.Equal(DType.Q4_K, GgufQuantPolicy.Q4_K_S.ResolveTargetDType("blocks.0.attn.qkv_proj.weight", weight));
    }

    /// <summary>The second line of defence, and the one that matters in Release: sizing a buffer for a misaligned
    /// element count used to truncate silently. It refuses now, so a caller that bypasses the policy gets an
    /// error rather than a heap overrun.</summary>
    [Fact]
    public void SizingABlockQuantBufferForAMisalignedCountIsRefused()
    {
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(
            () => DType.Q4_K.ComputeByteCount(8200));
        Assert.Contains("8 left over", ex.Message, StringComparison.Ordinal);
        // The aligned case is unchanged.
        Assert.Equal((8192 / 256) * DType.Q4_K.BlockByteSize, DType.Q4_K.ComputeByteCount(8192));
    }
}
