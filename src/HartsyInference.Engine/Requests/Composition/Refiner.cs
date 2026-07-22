namespace HartsyInference.Engine.Requests;

/// <summary>Second-pass refiner configuration: a distinct model (and optional VAE) run after the base pass, with its
/// own step/cfg budget and an optional latent upscale.</summary>
public sealed record Refiner
{
    /// <summary>Refiner model id or local path.</summary>
    public required string Model { get; init; }

    /// <summary>Refiner VAE override; null reuses the base VAE.</summary>
    public string? Vae { get; init; }

    /// <summary>Refiner hand-off method (e.g. how latents are transferred from the base pass).</summary>
    public string? Method { get; init; }

    /// <summary>Refiner control fraction (schedule split between base and refiner).</summary>
    public double? Control { get; init; }

    /// <summary>Refiner step count; null reuses the base steps.</summary>
    public int? Steps { get; init; }

    /// <summary>Refiner classifier-free guidance scale; null reuses the base CFG.</summary>
    public double? CfgScale { get; init; }

    /// <summary>Latent upscale factor applied before the refiner pass.</summary>
    public double? Upscale { get; init; }
}
