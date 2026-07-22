namespace HartsyInference.Engine.Requests;

/// <summary>An ordered stack of LoRA adapters to fuse before generation.</summary>
public sealed record LoraStack
{
    /// <summary>The LoRAs to apply, in application order.</summary>
    public required IReadOnlyList<LoraEntry> Entries { get; init; }
}
