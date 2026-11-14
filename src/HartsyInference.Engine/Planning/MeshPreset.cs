namespace HartsyInference.Engine.Planning;

/// <summary>What a <see cref="Requests.MeshQuality"/> sets for one 3D model: each value fills the request field of that name when the request leaves it at 0. A 0 here leaves the model's own default.</summary>
public sealed record MeshPreset
{
    /// <summary>Sampler steps, or 0.</summary>
    public int Steps { get; init; }

    /// <summary>The model's resolution knob (<see cref="Requests.MeshRequest.GridResolution"/>: a grid, or a texture edge for the Paint models), or 0.</summary>
    public int GridResolution { get; init; }

    /// <summary>Baked texture edge (<see cref="Requests.MeshRequest.TextureSize"/>), used only when the request also sets <see cref="Requests.MeshRequest.Texture"/>; or 0.</summary>
    public int TextureSize { get; init; }

    /// <summary>Where the values come from and what is measured about them on the 3060.</summary>
    public required string Basis { get; init; }
}
