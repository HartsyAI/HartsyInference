using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>Runs an <see cref="AttentionRoute.Auto"/> call whose operands no legacy entry point carries through the backend's
/// <see cref="IBackend.FlashAttention"/> entry, with the staging copies docs/ATTENTION_CONTRACT.md §6 allows: each strided Q, K and V
/// view gathered into a contiguous <c>[B, H, S, D]</c> tensor, a value narrower than the query/key width (MLA) zero-padded to it, and
/// the value columns of the output copied back into the output view.</summary>
/// <remarks>Covers unmasked calls with host positions, an optional soft-cap and optional sinks, F32 throughout: the single-query MQA
/// calls of the HCA and DSA layers. The padded value columns contribute zeros to output columns that are discarded, so the kept
/// columns are the unpadded attention. A GPU backend uses it; the CPU runs these calls on the contract reference.</remarks>
public static class AttentionFlashLowering
{
    /// <summary>Why <paramref name="call"/> cannot be lowered, or null when <see cref="Run"/> takes it.</summary>
    public static string? Gap(in AttentionCall call)
    {
        if (call.Route != AttentionRoute.Auto) return "a pinned route";
        if (call.Mask.Any) return "a mask";
        if (call.Positions.OnDevice) return "device positions";
        if (call.Logits.AlibiSlopes is not null) return "ALiBi";
        if (call.Query.DType != DType.F32 || call.Key.DType != DType.F32 || call.Value.DType != DType.F32 || call.Output.DType != DType.F32)
            return "a non-F32 operand";
        if (call.KvHeads <= 0 || call.Heads % call.KvHeads != 0) return "query heads that are not a multiple of the key/value heads";
        if (call.Key.HeadDim != call.QkHeadDim || call.ValueHeadDim > call.QkHeadDim) return "a value wider than the query/key";
        if (call.Value.Tokens != call.Key.Tokens || call.Value.Heads != call.KvHeads || call.Value.Batch != call.Batch || call.Key.Batch != call.Batch)
            return "key and value operands of different extents";
        if (call.Output.Heads != call.Heads || call.Output.Tokens != call.QueryTokens || call.Output.HeadDim != call.ValueHeadDim
            || call.Output.Batch != call.Batch)
            return "an output that does not match the query and value";
        if (call.ValidKvLength < 1) return "no valid key";
        return null;
    }

    /// <summary>Runs <paramref name="call"/> through <paramref name="backend"/>'s flash entry; <see cref="Gap"/> must be null.</summary>
    public static void Run(IBackend backend, in AttentionCall call)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (Gap(in call) is { } gap) throw new AttentionUnsupportedException($"The flash lowering cannot carry a call with {gap}.");
        int b = call.Batch, hq = call.Heads, sq = call.QueryTokens, hkv = call.KvHeads, skv = call.KeyTokens;
        int d = call.QkHeadDim, dv = call.ValueHeadDim;
        using Tensor q = Gather(backend, call.Query);
        using Tensor k = Gather(backend, call.Key);
        using Tensor vNarrow = Gather(backend, call.Value);
        using Tensor? vWide = dv < d ? new Tensor(new TensorShape(b, hkv, skv, d), DType.F32) : null;
        if (vWide is not null) backend.Pad(vWide, vNarrow, PadSpec.Axis(PadMode.Zero, 4, 3, 0, d - dv));
        using Tensor o = new(new TensorShape(b, hq, sq, d), DType.F32);
        backend.FlashAttention(o, q, k, vWide ?? vNarrow, call.ValidKvLength, hq / hkv, causal: false, call.Positions.QueryOffset, call.Scale,
            call.Logits.Softcap, call.Logits.Sinks);
        StridedView kept = new([b, hq, sq, dv], [(long)hq * sq * d, (long)sq * d, d, 1]);
        backend.CopyStrided(call.Output.Storage, o, new StridedCopySpec(View(call.Output), kept));
    }

    // The operand as a contiguous [B, H, S, D] tensor the caller disposes.
    private static Tensor Gather(IBackend backend, in AttentionOperand op)
    {
        Tensor t = new(new TensorShape(op.Batch, op.Heads, op.Tokens, op.HeadDim), DType.F32);
        backend.CopyStrided(t, op.Storage, StridedCopySpec.Gather(View(op)));
        return t;
    }

    private static StridedView View(in AttentionOperand op) =>
        new([op.Batch, op.Heads, op.Tokens, op.HeadDim], [op.BatchStride, op.HeadStride, op.TokenStride, 1], op.Offset);
}
