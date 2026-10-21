using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Where element <c>(b, h, q, k)</c> of an additive attention mask lives, in elements: <c>b·Batch + h·Head +
/// q·Row + k</c>, a stride of 0 on a broadcast axis. Every backend accepts the same shapes, F32 only:
/// <c>[1|Sq, Skv]</c>, <c>[1|H, 1|Sq, Skv]</c> and <c>[1|B, 1|H, 1|Sq, Skv]</c>.</summary>
/// <param name="BatchStride">Elements between batch items; 0 when one block serves every batch item.</param>
/// <param name="HeadStride">Elements between heads; 0 when one block serves every head.</param>
/// <param name="RowStride">Elements between query rows; 0 for a key-only mask (one row serves every query).</param>
public readonly record struct AttentionMaskLayout(long BatchStride, long HeadStride, long RowStride)
{
    /// <summary>Whether a mask of shape <paramref name="s"/> is one of the accepted shapes against a <c>[batch, heads, sq, skv]</c> score block.</summary>
    public static bool Broadcasts(TensorShape s, long batch, long heads, long sq, long skv) => s.Rank switch
    {
        2 => (s[0] == 1 || s[0] == sq) && s[1] == skv,
        3 => (s[0] == 1 || s[0] == heads) && (s[1] == 1 || s[1] == sq) && s[2] == skv,
        4 => (s[0] == 1 || s[0] == batch) && (s[1] == 1 || s[1] == heads) && (s[2] == 1 || s[2] == sq) && s[3] == skv,
        _ => false,
    };

    /// <summary>The layout of <paramref name="mask"/> against a <c>[batch, heads, sq, skv]</c> score block.</summary>
    /// <exception cref="NotSupportedException">The mask is not F32.</exception>
    /// <exception cref="ArgumentException">The mask does not broadcast, or it or a score block exceeds Int32.MaxValue elements.</exception>
    public static AttentionMaskLayout Of(Tensor mask, long batch, long heads, long sq, long skv)
    {
        if (mask.DType != DType.F32)
            throw new NotSupportedException($"SDPA additive masks must be F32; got {mask.DType}.");
        if (mask.ElementCount > int.MaxValue || sq * skv > int.MaxValue)
            throw new ArgumentException("SDPA mask blocks currently support at most Int32.MaxValue elements.");
        TensorShape s = mask.Shape;
        if (!Broadcasts(s, batch, heads, sq, skv))
            throw new ArgumentException(
                $"SDPA mask {s} is not broadcastable to [{batch},{heads},{sq},{skv}]; "
                + "expected [1|Sq,Skv], [1|H,1|Sq,Skv], or [1|B,1|H,1|Sq,Skv].");
        int r = s.Rank;
        long rows = s[r - 2], maskHeads = r >= 3 ? s[r - 3] : 1, maskBatch = r == 4 ? s[0] : 1;
        return new(maskBatch == 1 ? 0 : maskHeads * rows * skv, maskHeads == 1 ? 0 : rows * skv, rows == 1 ? 0 : skv);
    }
}
