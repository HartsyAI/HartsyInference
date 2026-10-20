using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Which keys each query sees and the finite terms added to its scores; the components compose
/// (docs/ATTENTION_CONTRACT.md §4). Positions are logical: a query's is <see cref="AttentionPositions.QueryOffset"/> plus
/// its row, a key's is its row.</summary>
public readonly record struct AttentionMask
{
    /// <summary>A key is visible only at a position at or before the query's.</summary>
    public bool Causal { get; init; }

    /// <summary>Keys more than this many positions before the query are excluded; null is unbounded. Inclusive.</summary>
    public int? WindowLeft { get; init; }

    /// <summary>Keys more than this many positions after the query are excluded; null is unbounded. Inclusive.</summary>
    public int? WindowRight { get; init; }

    /// <summary>Block-causal granularity: with &gt; 0 the causal and window rules count blocks of this many tokens.</summary>
    public int BlockSize { get; init; }

    /// <summary>Additive F32 bias broadcastable to <c>[B, H, Sq, Skv]</c>. Finite entries stay finite; only −∞ excludes.</summary>
    public Tensor? Bias { get; init; }

    /// <summary>No mask.</summary>
    public static AttentionMask None => default;

    /// <summary>An additive bias and nothing else.</summary>
    public static AttentionMask Additive(Tensor? bias) => new() { Bias = bias };

    /// <summary>The FlashAttention rule: causal, and with <paramref name="slidingWindow"/> &gt; 0 the query sees its own key
    /// and the <c>slidingWindow − 1</c> before it. A negative window is invalid; it is kept so the legacy entry that
    /// rejects it still sees it.</summary>
    public static AttentionMask FromFlash(bool causal, int slidingWindow) => new()
    {
        Causal = causal,
        WindowLeft = slidingWindow != 0 ? slidingWindow - 1 : null,
    };

    /// <summary>The legacy FlashAttention <c>slidingWindow</c> argument this mask was built from (0 = none).</summary>
    public int LegacySlidingWindow => WindowLeft is { } left ? left + 1 : 0;

    /// <summary>True when any exclusion or bias is present.</summary>
    public bool Any => Causal || WindowLeft is not null || WindowRight is not null || BlockSize > 0 || Bias is not null;
}

/// <summary>Valid key length and query position, on the host or in a device buffer (docs/ATTENTION_CONTRACT.md §2).</summary>
/// <param name="KvLength">Keys of the key operand that are valid when <paramref name="Bounded"/>; otherwise all are.</param>
/// <param name="QueryOffset">Logical position of query row 0; keys start at position 0.</param>
/// <param name="DeviceHandle">An <see cref="IBackend.AllocDevicePos"/> buffer holding <c>{kvLen, qOffset}</c>, or 0.</param>
/// <param name="Bounded">True when <paramref name="KvLength"/> was given, so 0 means no keys rather than all of them.</param>
public readonly record struct AttentionPositions(int KvLength, int QueryOffset, ulong DeviceHandle, bool Bounded = false)
{
    /// <summary>Every key is valid and query row 0 sits at position 0.</summary>
    public static AttentionPositions Dense => default;

    /// <summary>Host positions.</summary>
    public static AttentionPositions Host(int kvLength, int queryOffset) => new(kvLength, queryOffset, 0, true);

    /// <summary>Positions read on the device; the host values are placeholders.</summary>
    public static AttentionPositions Device(ulong handle, int kvLength, int queryOffset) => new(kvLength, queryOffset, handle, true);

    /// <summary>True when the kernel reads positions from the device.</summary>
    public bool OnDevice => DeviceHandle != 0;
}

/// <summary>Per-logit terms (docs/ATTENTION_CONTRACT.md §3): soft-cap, per-head sink logits, per-head ALiBi slopes.</summary>
public readonly record struct AttentionLogitTerms(float Softcap, Tensor? Sinks, Tensor? AlibiSlopes)
{
    /// <summary>True when any term is set.</summary>
    public bool Any => Softcap != 0f || Sinks is not null || AlibiSlopes is not null;
}
