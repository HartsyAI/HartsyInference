using HartsyInference.Core.Configuration;
using HartsyInference.Core.Memory.Vram;

namespace HartsyInference.Engine.Requests;

/// <summary>Native 3D-mesh request: text- or image-to-3D.</summary>
public sealed record MeshRequest
{
    /// <summary>The text prompt; may be empty for pure image-to-3D.</summary>
    public string Prompt { get; init; } = "";

    /// <summary>Input image for image-to-3D; null for text-to-3D.</summary>
    public ImageData? Image { get; init; }

    /// <summary>Number of generation steps; 0 uses the model default.</summary>
    public int Steps { get; init; }

    /// <summary>Grid / voxel resolution; 0 uses the model default.</summary>
    public int GridResolution { get; init; }

    /// <summary>RNG seed; negative means a random seed.</summary>
    public long Seed { get; init; } = -1;

    /// <summary>Number of samples (seeds Seed, Seed + 1, ...). Only HairDeck reads it; other models reject any value but 1.</summary>
    public int Samples { get; init; } = 1;

    /// <summary>HairDeck: colour the cards with the colour model. Null keeps the model default; other models reject a value.</summary>
    public bool? Colors { get; init; }

    /// <summary>HairDeck: also generate spring-bone chains. Null keeps the model default; other models reject a value.</summary>
    public bool? Chains { get; init; }

    /// <summary>HairDeck: rigged output, skinned to the spring chains with head-motion clips. Not implemented yet, so a request for it fails. Other models reject it.</summary>
    public bool Rigged { get; init; }

    /// <summary>Per-request VRAM lever overrides; null follows the backend's policy.</summary>
    public VramOverrides? Vram { get; init; }

    /// <summary>Per-request engine settings (profile + individual overrides); null keeps the machine's configuration.</summary>
    public RequestSettings? Settings { get; init; }
}
