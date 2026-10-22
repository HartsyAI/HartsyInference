using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>How an attention operand's heads and tokens sit in its backing tensor (docs/ATTENTION_CONTRACT.md §1).</summary>
public enum AttentionLayout
{
    /// <summary><c>[B, H, S, D]</c>; a KV-cache buffer is this with <c>S</c> = capacity.</summary>
    HeadMajor,

    /// <summary><c>[B, S, H, D]</c>, byte-identical to <c>[B·S, H·D]</c>: what a Linear emits.</summary>
    TokenMajor,

    /// <summary>Token-major heads at a column offset inside wider rows: one part of a fused projection.</summary>
    PackedRows,
}

/// <summary>One attention operand: a backing tensor seen through element strides, head dim contiguous.</summary>
/// <remarks>Carries the original <see cref="Tensor"/>: the activation cache and the Q8 sidecars are keyed on the object.</remarks>
public readonly record struct AttentionOperand(Tensor Storage, AttentionLayout Layout, int Batch, int Heads, int Tokens,
    int HeadDim, long Offset, long BatchStride, long HeadStride, long TokenStride)
{
    /// <summary>A contiguous <c>[B, H, S, D]</c> tensor.</summary>
    public static AttentionOperand HeadMajor(Tensor t)
    {
        int b = (int)t.Shape[0], h = (int)t.Shape[1], s = (int)t.Shape[2], d = (int)t.Shape[3];
        return new(t, AttentionLayout.HeadMajor, b, h, s, d, 0, (long)h * s * d, (long)s * d, d);
    }

    /// <summary>A contiguous token-major tensor, <c>[S, heads·headDim]</c> or <c>[B, S, heads, headDim]</c>.</summary>
    public static AttentionOperand TokenMajor(Tensor t, int heads, int headDim) =>
        TokenMajor(t, heads, headDim, t.Shape.Rank == 4 ? (int)t.Shape[0] : 1);

    /// <summary>A contiguous token-major tensor of <paramref name="batch"/> sequences, whatever its rank: <c>[B, S, heads·headDim]</c>
    /// or <c>[B, S, heads, headDim]</c>.</summary>
    public static AttentionOperand TokenMajor(Tensor t, int heads, int headDim, int batch)
    {
        long perToken = (long)batch * heads * headDim;
        if (batch <= 0 || perToken <= 0 || t.ElementCount % perToken != 0)
            throw new ArgumentException($"A token-major {t.Shape} tensor does not hold whole rows of {heads}x{headDim} per batch item.");
        int tokens = (int)(t.ElementCount / perToken);
        long row = (long)heads * headDim;
        return new(t, AttentionLayout.TokenMajor, batch, heads, tokens, headDim, 0, tokens * row, headDim, row);
    }

    /// <summary>Heads <c>[columnOffset, columnOffset + heads·headDim)</c> of each <paramref name="rowWidth"/>-wide row;
    /// <paramref name="rows"/> holds <paramref name="batch"/> × <paramref name="tokens"/> rows.</summary>
    public static AttentionOperand Packed(Tensor rows, int tokens, int rowWidth, int columnOffset, int heads, int headDim, int batch = 1)
        => new(rows, AttentionLayout.PackedRows, batch, heads, tokens, headDim, columnOffset, (long)tokens * rowWidth, headDim, rowWidth);

    /// <summary>The storage dtype.</summary>
    public DType DType => Storage.DType;

    /// <summary>One past the highest element index the view addresses.</summary>
    public long Extent => Offset + (Batch - 1L) * BatchStride + (Heads - 1L) * HeadStride + (Tokens - 1L) * TokenStride + HeadDim;

    /// <summary>True when the view is its whole storage in that layout's natural strides.</summary>
    public bool IsPlain => Offset == 0 && Layout != AttentionLayout.PackedRows
        && (long)Batch * Heads * Tokens * HeadDim == Storage.ElementCount
        && (Layout == AttentionLayout.HeadMajor
            ? BatchStride == (long)Heads * Tokens * HeadDim && HeadStride == (long)Tokens * HeadDim && TokenStride == HeadDim
            : BatchStride == (long)Tokens * Heads * HeadDim && HeadStride == HeadDim && TokenStride == (long)Heads * HeadDim);

    /// <summary>True when the view reaches every element of its storage exactly once, in any axis order.</summary>
    public bool CoversStorage
    {
        get
        {
            if (Offset != 0 || (long)Batch * Heads * Tokens * HeadDim != Storage.ElementCount) return false;
            // Axes of extent 1 carry no stride; the rest must tile the storage with no gap or overlap once ordered by stride.
            Span<(long Extent, long Stride)> axes = stackalloc (long, long)[3];
            int n = 0;
            if (Batch > 1) axes[n++] = (Batch, BatchStride);
            if (Heads > 1) axes[n++] = (Heads, HeadStride);
            if (Tokens > 1) axes[n++] = (Tokens, TokenStride);
            axes = axes[..n];
            axes.Sort(static (a, b) => a.Stride.CompareTo(b.Stride));
            long expected = HeadDim;
            foreach ((long extent, long stride) in axes)
            {
                if (stride != expected) return false;
                expected *= extent;
            }
            return true;
        }
    }
}
