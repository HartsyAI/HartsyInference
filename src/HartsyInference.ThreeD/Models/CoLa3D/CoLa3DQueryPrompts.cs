using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>Prompts for a prompt-driven decomposer: one query per (sample, prompt slot). Masks are <c>[Batch, Queries, H, W]</c>
/// for <see cref="CoLa3DConfig.QuerySourceKind.PromptMask"/>; points are <c>Batch·Queries·3</c> latent voxel coordinates for
/// <see cref="CoLa3DConfig.QuerySourceKind.PromptPoint"/>. <see cref="Valid"/> marks the slots that take part (null = all).</summary>
public sealed class CoLa3DQueryPrompts
{
    /// <summary>Samples in the batch.</summary>
    public required int Batch { get; init; }

    /// <summary>Prompt slots per sample.</summary>
    public required int Queries { get; init; }

    /// <summary>Prompt masks, F32 <c>[Batch, Queries, H, W]</c>, borrowed for the call.</summary>
    public Tensor? Masks { get; init; }

    /// <summary>Prompt voxel coordinates <c>(x, y, z)</c>, flat <c>Batch·Queries·3</c>.</summary>
    public int[]? Points { get; init; }

    /// <summary>Per-slot validity, flat <c>Batch·Queries</c>; null means every slot is valid.</summary>
    public bool[]? Valid { get; init; }
}
