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
    public static AttentionOperand TokenMajor(Tensor t, int heads, int headDim)
    {
        int batch = t.Shape.Rank == 4 ? (int)t.Shape[0] : 1;
        int tokens = (int)(t.ElementCount / ((long)batch * heads * headDim));
        long row = (long)heads * headDim;
        return new(t, AttentionLayout.TokenMajor, batch, heads, tokens, headDim, 0, tokens * row, headDim, row);
    }

    /// <summary>Heads <c>[columnOffset, columnOffset + heads·headDim)</c> of each <paramref name="rowWidth"/>-wide row.</summary>
    public static AttentionOperand Packed(Tensor rows, int tokens, int rowWidth, int columnOffset, int heads, int headDim)
        => new(rows, AttentionLayout.PackedRows, 1, heads, tokens, headDim, columnOffset, (long)tokens * rowWidth, headDim, rowWidth);

    /// <summary>The storage dtype.</summary>
    public DType DType => Storage.DType;

    /// <summary>One past the highest element index the view addresses.</summary>
    public long Extent => Offset + (Batch - 1L) * BatchStride + (Heads - 1L) * HeadStride + (Tokens - 1L) * TokenStride + HeadDim;

    /// <summary>True when the view is its whole storage in that layout's natural strides.</summary>
    public bool IsPlain => Offset == 0 && Layout != AttentionLayout.PackedRows
        && (long)Batch * Heads * Tokens * HeadDim == Storage.ElementCount;
}
