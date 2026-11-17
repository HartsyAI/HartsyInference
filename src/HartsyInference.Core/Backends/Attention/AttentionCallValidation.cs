using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>Validates the common attention-call boundary before a backend plans or writes output.</summary>
internal static class AttentionCallValidation
{
    /// <summary>Checks geometry, addressability, numerical refusals and output ownership.</summary>
    /// <exception cref="ArgumentException">The call is malformed.</exception>
    /// <exception cref="AttentionUnsupportedException">The call stores an explicitly refused format.</exception>
    public static void Validate(in AttentionCall call)
    {
        AttentionFormats refused = call.Numerics.Refused & AttentionNumerics.OperandFormats(in call);
        if (refused != AttentionFormats.None)
            throw new AttentionUnsupportedException($"The call refuses {refused}, and an operand or the output is stored in it.");
        if (!float.IsFinite(call.Scale))
            throw new ArgumentOutOfRangeException(nameof(call), "Attention scale must be finite.");

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

        ValidateBias(call.Mask.Bias, q, k);
        ValidatePerHead(call.Logits.Sinks, q.Heads, "Sinks");
        ValidatePerHead(call.Logits.AlibiSlopes, q.Heads, "ALiBi slopes");
    }

    private static void ValidateBias(Tensor? bias, AttentionOperand query, AttentionOperand key)
    {
        if (bias is null) return;
        if (bias.DType != DType.F32)
            throw new AttentionUnsupportedException($"Attention reads F32 additive bias, not {bias.DType}.");
        TensorShape s = bias.Shape;
        int rank = s.Rank;
        bool fits = rank is >= 2 and <= 4 && (s[rank - 1] == 1 || s[rank - 1] == key.Tokens)
            && (s[rank - 2] == 1 || s[rank - 2] == query.Tokens)
            && (rank < 3 || s[rank - 3] == 1 || s[rank - 3] == query.Heads)
            && (rank < 4 || s[0] == 1 || s[0] == query.Batch);
        if (!fits)
            throw new ArgumentException($"Bias {s} does not broadcast to [{query.Batch},{query.Heads},{query.Tokens},{key.Tokens}].");
    }

    private static void ValidatePerHead(Tensor? tensor, int heads, string name)
    {
        if (tensor is not null && tensor.DType != DType.F32)
            throw new AttentionUnsupportedException($"{name} are F32, not {tensor.DType}.");
        if (tensor is { ElementCount: var count } && count < heads)
            throw new ArgumentException($"{name} hold {count} values for {heads} heads.");
    }

    private static string Describe(in AttentionOperand x) =>
        $"{x.Layout}[b{x.Batch} h{x.Heads} s{x.Tokens} d{x.HeadDim}] of {x.Storage.Shape}";
}
