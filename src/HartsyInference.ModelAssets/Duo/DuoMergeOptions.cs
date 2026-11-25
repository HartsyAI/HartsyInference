namespace HartsyInference.ModelAssets.Duo;

/// <summary>How <see cref="DuoRepacker.MergeBackbone"/> merges the LoRA into the backbone shards.</summary>
public sealed record DuoMergeOptions
{
    /// <summary>LoRA alpha; the scale is <c>Alpha / rank</c> per module. Null means alpha equals the rank, a scale of 1, which is what the released checkpoints record (rank 128, alpha 128).</summary>
    public double? Alpha { get; init; }

    public int Heads { get; init; } = 56;

    public int HeadDim { get; init; } = 128;

    public DuoQkvInput InputQkv { get; init; } = DuoQkvInput.HeadInterleaved;

    /// <summary>Leave the 50 per-block <c>adaln_proj</c> layers (26 GB in BF16) out of the output, for use with <see cref="DuoRepacker.WriteAdalnTables"/>. The final layer's <c>adaln_proj</c> stays.</summary>
    public bool DropBlockAdalnProj { get; init; }
}
