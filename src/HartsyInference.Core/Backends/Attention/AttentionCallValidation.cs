using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>Validates the common attention-call boundary before a backend plans or writes output.</summary>
/// <remarks>Every malformed-call check runs before any unsupported-format check, so a call that is both malformed and
/// refused reports the malformation. Per-head sinks and ALiBi slopes follow the documented <c>[Hq]</c> F32 contract;
/// a different dtype or count is malformed, as the CUDA flash contract has always treated it.</remarks>
internal static class AttentionCallValidation
{
    /// <summary>Checks geometry, addressability, logit terms, mask windows, output ownership and numerical refusals.</summary>
    /// <exception cref="ArgumentException">The call is malformed.</exception>
    /// <exception cref="AttentionUnsupportedException">The call is well formed but stores a refused or unread format.</exception>
    public static void Validate(in AttentionCall call)
    {
        if (!float.IsFinite(call.Scale))
            throw new ArgumentOutOfRangeException(nameof(call), "Attention scale must be finite.");
        if (!float.IsFinite(call.Logits.Softcap) || call.Logits.Softcap < 0f)
            throw new ArgumentOutOfRangeException(nameof(call), "Attention softcap must be finite and non-negative.");
        AttentionMask mask = call.Mask;
        if (mask.WindowLeft < 0 || mask.WindowRight < 0 || mask.BlockSize < 0)
            throw new ArgumentOutOfRangeException(nameof(call),
                $"Attention windows and block size must be non-negative; got left={mask.WindowLeft}, right={mask.WindowRight}, block={mask.BlockSize}.");
        if (!call.Positions.OnDevice && call.Positions.QueryOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(call), $"Attention query offset {call.Positions.QueryOffset} is negative.");

        AttentionOperand q = call.Query, k = call.Key, v = call.Value, o = call.Output;
        if (q.Batch <= 0 || q.Heads <= 0 || q.Tokens <= 0 || q.HeadDim <= 0 || k.Heads <= 0 || k.Tokens <= 0 || v.HeadDim <= 0)
            throw new ArgumentException($"Attention operands need positive extents; got Q={Describe(q)}, K={Describe(k)}, V={Describe(v)}.");
        if (q.Heads % k.Heads != 0)
            throw new ArgumentException($"Query heads ({q.Heads}) must be a multiple of key/value heads ({k.Heads}).");
        if (k.HeadDim != q.HeadDim || v.Heads != k.Heads || v.Tokens != k.Tokens || o.HeadDim != v.HeadDim
            || o.Heads != q.Heads || o.Tokens != q.Tokens || k.Batch != q.Batch || v.Batch != q.Batch || o.Batch != q.Batch)
            throw new ArgumentException(
                $"Attention operands disagree: Q={Describe(q)}, K={Describe(k)}, V={Describe(v)}, O={Describe(o)}.");
        if (call.ValidKvLength < 0 || call.ValidKvLength > k.Tokens)
            throw new ArgumentException($"Valid key length {call.ValidKvLength} is outside the key operand's {k.Tokens} rows.");
        foreach (AttentionOperand x in (ReadOnlySpan<AttentionOperand>)[q, k, v, o])
            if (x.Offset < 0 || x.BatchStride < 0 || x.HeadStride < 0 || x.TokenStride < 0 || x.Extent > x.Storage.ElementCount)
                throw new ArgumentException($"Operand {Describe(x)} addresses past its {x.Storage.ElementCount}-element tensor.");
        if (ReferenceEquals(o.Storage, q.Storage) || ReferenceEquals(o.Storage, k.Storage) || ReferenceEquals(o.Storage, v.Storage))
            throw new ArgumentException("Attention output must not alias an input.", nameof(call));
        if (mask.Bias is { } bias && !BiasBroadcasts(bias.Shape, q, k))
            throw new ArgumentException($"Bias {bias.Shape} does not broadcast to [{q.Batch},{q.Heads},{q.Tokens},{k.Tokens}].");
        ValidatePerHead(call.Logits.Sinks, q.Heads, "Sinks");
        ValidatePerHead(call.Logits.AlibiSlopes, q.Heads, "ALiBi slopes");

        if (mask.Bias is { } unread && unread.DType != DType.F32)
            throw new AttentionUnsupportedException($"Attention reads F32 additive bias, not {unread.DType}.");
        AttentionFormats refused = call.Numerics.Refused & AttentionNumerics.OperandFormats(in call);
        if (refused != AttentionFormats.None)
            throw new AttentionUnsupportedException($"The call refuses {refused}, and an operand or the output is stored in it.");
    }

    /// <summary>The host reference's broadcast rule: rank 2–4, every axis either 1 or the full extent. The legacy GPU kernels
    /// accept a narrower set (the key axis must be <c>Skv</c>); their rows decline the rest after this boundary.</summary>
    private static bool BiasBroadcasts(TensorShape s, AttentionOperand query, AttentionOperand key)
    {
        int rank = s.Rank;
        return rank is >= 2 and <= 4 && (s[rank - 1] == 1 || s[rank - 1] == key.Tokens)
            && (s[rank - 2] == 1 || s[rank - 2] == query.Tokens)
            && (rank < 3 || s[rank - 3] == 1 || s[rank - 3] == query.Heads)
            && (rank < 4 || s[0] == 1 || s[0] == query.Batch);
    }

    private static void ValidatePerHead(Tensor? tensor, int heads, string name)
    {
        if (tensor is not null && (tensor.DType != DType.F32 || tensor.ElementCount != heads))
            throw new ArgumentException($"{name} must be F32 with exactly {heads} values; got {tensor.DType} {tensor.Shape}.");
    }

    private static string Describe(in AttentionOperand x) =>
        $"{x.Layout}[b{x.Batch} h{x.Heads} s{x.Tokens} d{x.HeadDim}] of {x.Storage.Shape}";
}
