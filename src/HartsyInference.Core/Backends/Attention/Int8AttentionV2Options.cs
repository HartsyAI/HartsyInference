namespace HartsyInference.Core.Backends;

/// <summary>How INT8 attention v2 groups its K scales.</summary>
public enum Int8KeyScaleGrouping
{
    /// <summary>Four scales per key tile, one per class <c>(key mod 8) / 2</c>: the keys one lane's mma columns hold
    /// (Comfy Kitchen's and SageAttention's per-thread grouping). Layout <c>[B, Hkv, tiles, 4]</c>.</summary>
    ThreadClass,

    /// <summary>One scale per key. Layout <c>[B, Hkv, kvLen]</c>.</summary>
    PerKey,
}

/// <summary>The formats P and V take in the PV product.</summary>
public enum Int8ProbabilityFormat
{
    /// <summary>u8 P against its key tile's own maximum, rounded to nearest even, times s8 V with per-channel scales;
    /// the INT32 partial drains into F32 at every key tile.</summary>
    U8,

    /// <summary><see cref="U8"/> with P rounded up or down by a position hash: unbiased, bitwise reproducible.</summary>
    U8Stochastic,

    /// <summary>E4M3 P against the running maximum times E4M3 V with per-channel scales: the sm_89 fallback.</summary>
    Fp8,

    /// <summary>F16 P in [0, 1] against the running maximum times F16 V: the sm_86 fallback.</summary>
    F16,
}

/// <summary>What a (query tile, key tile) pair holds under a call's exclusions.</summary>
public enum AttentionTileClass
{
    /// <summary>No query row of the tile sees any key of it; the tile is not visited.</summary>
    Skip,

    /// <summary>Every query row sees every key slot; no element mask is applied.</summary>
    Full,

    /// <summary>Some pairs are excluded or past the valid keys; the element mask applies.</summary>
    Partial,
}

/// <summary>One INT8 attention v2 variant: tile sizes, K scales, K smoothing, V scales and the PV formats.</summary>
public sealed record Int8AttentionV2Options
{
    /// <summary>Query rows per tile (BR), 64 or 128. It only classifies tiles; no row's result depends on it.</summary>
    public int BlockRows { get; init; } = 64;

    /// <summary>Keys per tile (BC), 64 or 128: the P quantization block, INT32 drain interval and K scale tile.</summary>
    public int BlockCols { get; init; } = 128;

    /// <summary>K scale grouping.</summary>
    public Int8KeyScaleGrouping KeyScales { get; init; } = Int8KeyScaleGrouping.ThreadClass;

    /// <summary>Subtract each key segment's channel mean before K is quantized.</summary>
    public bool SmoothKeys { get; init; } = true;

    /// <summary>Starts of the key segments after the first, ascending; each is smoothed with its own mean.</summary>
    public IReadOnlyList<int> KeySegmentStarts { get; init; } = [];

    /// <summary>One V channel scale per key tile instead of one per sequence, applied at each tile's drain.</summary>
    public bool ValueScalesPerTile { get; init; }

    /// <summary>P and V formats.</summary>
    public Int8ProbabilityFormat Probabilities { get; init; } = Int8ProbabilityFormat.U8;

    /// <summary>Seed of the stochastic rounding hash.</summary>
    public uint RoundingSeed { get; init; }

    /// <summary>The recurrence's exp2, null for double exp2 narrowed to F32; a kernel test replays the GPU's.</summary>
    public Func<float, float>? Exp2 { get; init; }

    /// <summary>Key segments: one, or one plus <see cref="KeySegmentStarts"/> when keys are smoothed.</summary>
    public int Segments => SmoothKeys ? 1 + KeySegmentStarts.Count : 1;
}
