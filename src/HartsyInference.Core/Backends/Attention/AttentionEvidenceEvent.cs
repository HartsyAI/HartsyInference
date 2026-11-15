namespace HartsyInference.Core.Backends.Attention;

/// <summary>A tensor-free observation; byte counts describe the named action, not total device traffic.</summary>
public readonly record struct AttentionEvidenceEvent
{
    public AttentionEvidenceKind Kind { get; init; }
    public string? Site { get; init; }
    public string? Component { get; init; }
    public int Layer { get; init; }
    public string? Implementation { get; init; }
    public string? Branch { get; init; }
    public AttentionRoute Route { get; init; }
    public int Batch { get; init; }
    public int Heads { get; init; }
    public int KvHeads { get; init; }
    public int QueryTokens { get; init; }
    public int KeyCapacity { get; init; }
    public int ValidKeys { get; init; }
    public int QkHeadDim { get; init; }
    public int ValueHeadDim { get; init; }
    public string? QueryType { get; init; }
    public string? KvType { get; init; }
    public AttentionLayout QueryLayout { get; init; }
    public AttentionLayout KvLayout { get; init; }
    public long QueryOffset { get; init; }
    public long QueryHeadStride { get; init; }
    public long QueryTokenStride { get; init; }
    public long KeyOffset { get; init; }
    public long KeyHeadStride { get; init; }
    public long KeyTokenStride { get; init; }
    public long Bytes { get; init; }
    public long Count { get; init; }
    public AttentionNumerics Numerics { get; init; }
    public AttentionLowering DeclaredLowering { get; init; }
}
