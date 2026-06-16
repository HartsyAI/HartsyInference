using System.Collections.Generic;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Maps a denoise step to one of several pre-encoded conditioning tensors, the runtime form of prompt alternation <c>[a|b]</c> and scheduling <c>[a:b:when]</c>. Pipelines encode each distinct variant once before the loop, then call <see cref="Resolve"/> per step.</summary>
public sealed record ConditioningSchedule
{
    /// <summary>The distinct conditioning tensors, one per scheduled variant.</summary>
    public required IReadOnlyList<Tensor> Variants { get; init; }

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
