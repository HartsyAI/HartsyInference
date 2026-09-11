using System.Collections.Generic;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Maps a denoise step to one of several pre-encoded conditioning tensors, the runtime form of the
/// <c>&lt;alternate:&gt;</c>/<c>&lt;fromto[N]:&gt;</c> prompt tags. Pipelines encode each distinct variant once before the
/// loop, then call <see cref="Resolve"/> per step.</summary>
public sealed record ConditioningSchedule
{
    /// <summary>The distinct conditioning tensors, one per scheduled variant.</summary>
    public required IReadOnlyList<Tensor> Variants { get; init; }

    /// <summary>Pooled/ADM conditioning for each variant, indexed identically to <see cref="Variants"/>, for the
    /// architectures that carry one (SDXL's CLIP-G pooled vector). Null means the pipeline keeps using its own
    /// single pooled encode — correct whenever the schedule has one variant, and the only option for an
    /// architecture with no pooled conditioning at all (SD 1.5). When a multi-variant schedule leaves this null,
    /// the hidden states switch per step while the pooled vector stays frozen at variant 0, which pairs (say)
    /// "dog" hidden states with "cat" ADM conditioning.</summary>
    public IReadOnlyList<Tensor>? PooledVariants { get; init; }

    /// <summary>Selector mapping <c>(step, totalSteps)</c> to an index into <see cref="Variants"/>.</summary>
    public required Func<int, int, int> IndexForStep { get; init; }

    /// <summary>Resolves the variant index for <paramref name="step"/> and validates it is in range.</summary>
    public int Resolve(int step, int totalSteps)
    {
        int index = IndexForStep(step, totalSteps);
        if (index < 0 || index >= Variants.Count)
        {
            throw new HartsyInferenceException($"ConditioningSchedule index {index} out of range [0, {Variants.Count}).");
        }
        return index;
    }

    /// <summary>Builds a schedule from a resolved <see cref="PromptSchedule"/> and its already-encoded variant tensors (same order as <see cref="PromptSchedule.Variants"/>).</summary>
    public static ConditioningSchedule FromPromptSchedule(PromptSchedule schedule, IReadOnlyList<Tensor> encodedVariants)
    {
        if (encodedVariants.Count != schedule.Variants.Count)
        {
            throw new HartsyInferenceException($"Encoded variant count {encodedVariants.Count} must equal prompt variant count {schedule.Variants.Count}.");
        }
        int[] map = schedule.StepToVariant;
        return new ConditioningSchedule
        {
            Variants = encodedVariants,
            IndexForStep = (step, total) => map[Math.Clamp(step, 0, map.Length - 1)],
        };
    }
}
