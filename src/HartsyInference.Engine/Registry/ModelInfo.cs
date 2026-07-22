using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Registry;

/// <summary>Metadata about a known model — its name, location, format, and architecture.</summary>
public sealed class ModelInfo
{
    /// <summary>Unique identifier for this model in the registry.</summary>
    public required string Id { get; init; }

    /// <summary>Human-readable model name.</summary>
    public required string Name { get; init; }

    /// <summary>Model format on disk.</summary>
    public required ModelFormat Format { get; init; }

    /// <summary>Architecture identifier (e.g., "stable-diffusion-v1-5", "flux-dev").</summary>
    public string? Architecture { get; init; }

    /// <summary>Data type of the model weights.</summary>
    public DType WeightDType { get; init; } = DType.F16;

    /// <summary>Local file path to the model.</summary>
    public required string LocalPath { get; init; }

    /// <summary>File size in bytes.</summary>
    public long FileSize { get; init; }

    /// <summary>HuggingFace repository ID (e.g., "runwayml/stable-diffusion-v1-5"), if applicable.</summary>
    public string? HuggingFaceRepo { get; init; }

    /// <summary>When this model was added or last accessed.</summary>
    public DateTime LastAccessed { get; set; } = DateTime.UtcNow;
}
