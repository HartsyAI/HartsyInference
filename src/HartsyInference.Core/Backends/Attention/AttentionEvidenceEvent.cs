namespace HartsyInference.Core.Backends.Attention;

/// <summary>A tensor-free observation; byte counts describe the named action, not total device traffic.</summary>
public readonly record struct AttentionEvidenceEvent
{
    /// <summary>Completed action category.</summary>
    public AttentionEvidenceKind Kind { get; init; }
    /// <summary>Stable source-site identifier.</summary>
    public string? Site { get; init; }
    /// <summary>Caller component, when a synchronous scope supplied it.</summary>
    public string? Component { get; init; }
    /// <summary>Caller layer index, or minus one when unknown.</summary>
    public int Layer { get; init; }
    /// <summary>Executed candidate identifier.</summary>
    public string? Implementation { get; init; }
    /// <summary>Existing backend executed-branch identifier.</summary>
    public string? Branch { get; init; }
    /// <summary>Entry family requested by the call.</summary>
    public AttentionRoute Route { get; init; }
    /// <summary>Batch size.</summary>
    public int Batch { get; init; }
    /// <summary>Query head count.</summary>
    public int Heads { get; init; }
    /// <summary>Key/value head count.</summary>
    public int KvHeads { get; init; }
    /// <summary>Query rows.</summary>
    public int QueryTokens { get; init; }
    /// <summary>Key storage capacity in rows.</summary>
    public int KeyCapacity { get; init; }
    /// <summary>Host-declared valid key rows.</summary>
    public int ValidKeys { get; init; }
    /// <summary>Query/key channels per head.</summary>
    public int QkHeadDim { get; init; }
    /// <summary>Value channels per head.</summary>
    public int ValueHeadDim { get; init; }
    /// <summary>Query storage dtype.</summary>
    public string? QueryType { get; init; }
    /// <summary>Key storage dtype.</summary>
    public string? KvType { get; init; }
    /// <summary>Query layout declaration.</summary>
    public AttentionLayout QueryLayout { get; init; }
    /// <summary>Key layout declaration.</summary>
    public AttentionLayout KvLayout { get; init; }
    /// <summary>Query storage offset in elements.</summary>
    public long QueryOffset { get; init; }
    /// <summary>Query head stride in elements.</summary>
    public long QueryHeadStride { get; init; }
    /// <summary>Query token stride in elements.</summary>
    public long QueryTokenStride { get; init; }
    /// <summary>Key storage offset in elements.</summary>
    public long KeyOffset { get; init; }
    /// <summary>Key head stride in elements.</summary>
    public long KeyHeadStride { get; init; }
    /// <summary>Key token stride in elements.</summary>
    public long KeyTokenStride { get; init; }
    /// <summary>Logical payload bytes of the named preparation action.</summary>
    public long Bytes { get; init; }
    /// <summary>Action count, or admitted sequence count for a scheduler batch.</summary>
    public long Count { get; init; }
    /// <summary>Declared numerical permissions, not a new admission decision.</summary>
    public AttentionNumerics Numerics { get; init; }
    /// <summary>Candidate lowering declaration; separate preparation events prove actual actions.</summary>
    public AttentionLowering DeclaredLowering { get; init; }
}
