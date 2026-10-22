using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>The legacy entry point a call came through. A pinned call is planned over that entry's rows only, so a
/// migrated call keeps the kernel it had until its own bounded A/B moves it.</summary>
public enum AttentionRoute
{
    /// <summary>The planner chooses among every row.</summary>
    Auto,

    /// <summary><c>ScaledDotProductAttention</c>, head-major.</summary>
    Dense,

    /// <summary><c>ScaledDotProductAttentionTokenMajor</c>, equal heads.</summary>
    TokenMajor,

    /// <summary><c>ScaledDotProductAttentionTokenMajor</c> with fewer K/V heads.</summary>
    TokenMajorGqa,

    /// <summary><c>FlashAttention</c>, host positions.</summary>
    Flash,

    /// <summary><c>FlashAttentionDev</c>, device positions.</summary>
    FlashDevice,
}

/// <summary>One attention launch (docs/ATTENTION_CONTRACT.md): operands, mask, positions, logit terms and permissions.</summary>
public readonly record struct AttentionCall
{
    /// <summary>Queries.</summary>
    public required AttentionOperand Query { get; init; }

    /// <summary>Keys; for a cache view, <see cref="AttentionOperand.Tokens"/> is the capacity.</summary>
    public required AttentionOperand Key { get; init; }

    /// <summary>Values.</summary>
    public required AttentionOperand Value { get; init; }

    /// <summary>Output; must not alias an input.</summary>
    public required AttentionOperand Output { get; init; }

    /// <summary>Score multiplier α; padding a head dim never changes it.</summary>
    public required float Scale { get; init; }

    /// <summary>Exclusions and additive terms.</summary>
    public AttentionMask Mask { get; init; }

    /// <summary>Valid key length and query position.</summary>
    public AttentionPositions Positions { get; init; }

    /// <summary>Soft-cap, sinks, ALiBi.</summary>
    public AttentionLogitTerms Logits { get; init; }

    /// <summary>Permitted approximations and their bounds.</summary>
    public AttentionNumerics Numerics { get; init; }

    /// <summary>The legacy entry point the call is pinned to; <see cref="AttentionRoute.Auto"/> plans freely.</summary>
    public AttentionRoute Route { get; init; }

    /// <summary>Batch.</summary>
    public int Batch => Query.Batch;

    /// <summary>Query heads.</summary>
    public int Heads => Query.Heads;

    /// <summary>Key/value heads.</summary>
    public int KvHeads => Key.Heads;

    /// <summary>Query heads per key/value head.</summary>
    public int KvGroup => Key.Heads == 0 ? 0 : Query.Heads / Key.Heads;

    /// <summary>Query rows present.</summary>
    public int QueryTokens => Query.Tokens;

    /// <summary>Key rows present (a cache's capacity).</summary>
    public int KeyTokens => Key.Tokens;

    /// <summary>Head dim of Q and K.</summary>
    public int QkHeadDim => Query.HeadDim;

    /// <summary>Head dim of V and the output.</summary>
    public int ValueHeadDim => Value.HeadDim;

    /// <summary>Keys actually attended.</summary>
    public int ValidKvLength => Positions.Bounded ? Positions.KvLength : Key.Tokens;

    /// <summary>What the legacy entry point for <see cref="Route"/> cannot carry, or null when it can carry the whole call.
    /// A backend that serves <see cref="IBackend.Attention"/> by re-entering that entry point refuses the rest rather than
    /// dropping it.</summary>
    public string? LegacyGap() => Gap(operands: true);

    /// <summary>What a row of the pinned route cannot carry once the operand views are set aside: the mask terms, logit terms
    /// and positions of a legacy entry. A backend whose rows take strided views plans over its rows when this is null.</summary>
    public string? TermsGap() => Gap(operands: false);

    private string? Gap(bool operands)
    {
        AttentionMask m = Mask;
        AttentionLayout layout = Route is AttentionRoute.TokenMajor or AttentionRoute.TokenMajorGqa
            ? AttentionLayout.TokenMajor : AttentionLayout.HeadMajor;
        if (operands)
        {
            if (Query.Layout != layout || Key.Layout != layout || Value.Layout != layout || Output.Layout != layout)
                return $"{Query.Layout} operands on the {Route} route";
            if (!Query.IsPlain || !Key.IsPlain || !Value.IsPlain || !Output.IsPlain) return "strided operands";
        }
        if (m.WindowRight is not null || m.BlockSize > 0) return "a right window or a block-causal mask";
        switch (Route)
        {
            case AttentionRoute.Dense or AttentionRoute.TokenMajor or AttentionRoute.TokenMajorGqa:
                if (operands && layout == AttentionLayout.TokenMajor && Batch > 1)
                    return "a batch above 1 (the token-major entry reads one sequence)";
                if (m.Causal || m.WindowLeft is not null) return "a causal or windowed mask (it takes an additive bias)";
                if (Logits.Any) return "a softcap, sinks or ALiBi";
                if (Positions.OnDevice || (Positions.Bounded && (Positions.KvLength != Key.Tokens || Positions.QueryOffset != 0)))
                    return "key or query positions";
                return null;
            case AttentionRoute.Flash:
                if (m.Bias is not null) return "an additive bias";
                return Positions.OnDevice ? "device positions (the FlashDevice route reads them)" : null;
            case AttentionRoute.FlashDevice:
                if (m.Bias is not null) return "an additive bias";
                return Logits.Sinks is not null || Logits.AlibiSlopes is not null ? "sinks or ALiBi" : null;
            default:
                return "an unpinned route";
        }
    }

    /// <summary>A <c>ScaledDotProductAttention</c> call: head-major operands, an optional additive bias.</summary>
    public static AttentionCall FromScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask,
        float scale, bool allowF16) => new()
        {
            Query = AttentionOperand.HeadMajor(query), Key = AttentionOperand.HeadMajor(key),
            Value = AttentionOperand.HeadMajor(value), Output = AttentionOperand.HeadMajor(output),
            Scale = scale, Mask = AttentionMask.Additive(mask), Numerics = AttentionNumerics.FromLegacy(allowF16),
            Route = AttentionRoute.Dense,
        };

    /// <summary>A <c>ScaledDotProductAttentionTokenMajor</c> call; <paramref name="kvHeads"/> below
    /// <paramref name="heads"/> is the grouped-query overload.</summary>
    public static AttentionCall FromTokenMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, int heads,
        int kvHeads, int headDim, float scale, bool allowF16)
    {
        ValidateHeads(heads, kvHeads, headDim);
        return Token(AttentionOperand.TokenMajor(output, heads, headDim), AttentionOperand.TokenMajor(query, heads, headDim),
            AttentionOperand.TokenMajor(key, kvHeads, headDim), AttentionOperand.TokenMajor(value, kvHeads, headDim), mask, heads, kvHeads,
            scale, allowF16);
    }

    /// <summary>A token-major call over <paramref name="batch"/> sequences whatever each tensor's rank: <c>[B, S, heads·D]</c> rows
    /// read as B sequences rather than one of B·S tokens.</summary>
    public static AttentionCall FromTokenMajor(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, int heads,
        int kvHeads, int headDim, float scale, bool allowF16, int batch)
    {
        ValidateHeads(heads, kvHeads, headDim);
        return Token(AttentionOperand.TokenMajor(output, heads, headDim, batch), AttentionOperand.TokenMajor(query, heads, headDim, batch),
            AttentionOperand.TokenMajor(key, kvHeads, headDim, batch), AttentionOperand.TokenMajor(value, kvHeads, headDim, batch), mask,
            heads, kvHeads, scale, allowF16);
    }

    private static void ValidateHeads(int heads, int kvHeads, int headDim)
    {
        if (heads <= 0 || kvHeads <= 0 || headDim <= 0)
            throw new ArgumentException($"Token-major attention needs positive heads, kvHeads and headDim; got {heads}, {kvHeads}, {headDim}.");
    }

    private static AttentionCall Token(AttentionOperand output, AttentionOperand query, AttentionOperand key, AttentionOperand value,
        Tensor? mask, int heads, int kvHeads, float scale, bool allowF16) => new()
        {
            Query = query, Key = key, Value = value, Output = output,
            Scale = scale, Mask = AttentionMask.Additive(mask), Numerics = AttentionNumerics.FromLegacy(allowF16),
            Route = kvHeads == heads ? AttentionRoute.TokenMajor : AttentionRoute.TokenMajorGqa,
        };

    /// <summary>A <c>FlashAttention</c> call: head-major Q/O, a head-major K/V cache whose first <paramref name="kvLen"/>
    /// rows are valid, query row 0 at <paramref name="qOffset"/>.</summary>
    public static AttentionCall FromFlashAttention(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, bool causal,
        int qOffset, float scale, float softcap, Tensor? sink, int slidingWindow, Tensor? alibiSlopes) => new()
        {
            Query = AttentionOperand.HeadMajor(query), Key = AttentionOperand.HeadMajor(key),
            Value = AttentionOperand.HeadMajor(value), Output = AttentionOperand.HeadMajor(output),
            Scale = scale, Mask = AttentionMask.FromFlash(causal, slidingWindow), Positions = AttentionPositions.Host(kvLen, qOffset),
            Logits = new(softcap, sink, alibiSlopes), Route = AttentionRoute.Flash,
        };

    /// <summary>A <c>FlashAttentionDev</c> call: as <see cref="FromFlashAttention"/>, positions read from
    /// <paramref name="devicePos"/> (the host values are the eager fallback's).</summary>
    public static AttentionCall FromFlashAttentionDev(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, bool causal,
        int qOffset, float scale, ulong devicePos, float softcap, int slidingWindow) => new()
        {
            Query = AttentionOperand.HeadMajor(query), Key = AttentionOperand.HeadMajor(key),
            Value = AttentionOperand.HeadMajor(value), Output = AttentionOperand.HeadMajor(output),
            Scale = scale, Mask = AttentionMask.FromFlash(causal, slidingWindow),
            Positions = AttentionPositions.Device(devicePos, kvLen, qOffset), Logits = new(softcap, null, null),
            Route = AttentionRoute.FlashDevice,
        };
}

/// <summary>No kernel row covers a call within its permissions and budget; the message names what was unmet.</summary>
public sealed class AttentionUnsupportedException(string message) : NotSupportedException(message);
