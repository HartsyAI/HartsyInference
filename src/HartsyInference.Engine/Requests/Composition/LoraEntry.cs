namespace HartsyInference.Engine.Requests;

/// <summary>One LoRA in a <see cref="LoraStack"/>: the adapter model plus its blend weights and optional section scope.</summary>
public sealed record LoraEntry
{
    /// <summary>LoRA model id or local path.</summary>
    public required string Model { get; init; }

    /// <summary>Blend weight applied to the diffusion (UNet/transformer) side.</summary>
    public double Weight { get; init; } = 1.0;

    /// <summary>Blend weight applied to the text-encoder side; defaults to <see cref="Weight"/> when null.</summary>
    public double? TextEncoderWeight { get; init; }

    /// <summary>Optional section-confinement expression restricting where the LoRA applies; null for global.</summary>
    public string? SectionConfinement { get; init; }
}
