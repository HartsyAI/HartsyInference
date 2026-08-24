using HartsyInference.Core.MemoryManagement;

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

    /// <summary>Per-request VRAM lever overrides; null follows the backend's policy.</summary>
    public VramOverrides? Vram { get; init; }
}
