using System.Collections.Generic;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>A prompt whose text changes across the denoise loop — SwarmUI's <c>&lt;alternate:a, b&gt;</c> and
/// <c>&lt;fromto[N]:a, b&gt;</c> — tokenized once per DISTINCT variant rather than once per step.
/// <para>This is the token-side half of <see cref="ConditioningSchedule"/>: a pipeline builds one of these before
/// its loop, encodes <see cref="Variants"/> once each, and then selects by step. SwarmUI does the same thing in a
/// different shape — <c>SwarmTextEncodeAdvanced</c> resolves the prompt per step, groups contiguous runs that
/// resolve alike, and hands ComfyUI one conditioning per run tagged with a timestep range
/// (<c>SwarmText.py</c>, the <c>parsed.has_steps()</c> branch). Indexing by step instead of by timestep range is
/// equivalent and dedupes globally rather than only across neighbours, so <c>&lt;alternate:a, b&gt;</c> over 30
/// steps encodes twice here where a per-run grouping would encode thirty times (its own cache notwithstanding).
/// </para></summary>
public sealed record ScheduledPrompt
{
    /// <summary>The distinct tokenized variants, in first-occurrence order.</summary>
    public required IReadOnlyList<WeightedTokenSequence> Variants { get; init; }

    /// <summary>Index into <see cref="Variants"/> for each step of the loop this was built for.</summary>
    public required int[] StepToVariant { get; init; }

    /// <summary>Whether the prompt actually changes across steps. A prompt with a scheduling tag whose branches
    /// resolve to the same text — or one resolved for a single step — has one variant and needs no schedule.</summary>
    public bool IsScheduled => Variants.Count > 1;

    /// <summary>The variant for <paramref name="step"/>, clamped so a sampler that runs past the step count it
    /// planned for reuses the last entry rather than throwing.</summary>
    public int IndexForStep(int step) => StepToVariant[Math.Clamp(step, 0, StepToVariant.Length - 1)];

    /// <summary>Resolves <paramref name="prompt"/>'s scheduling tags and tokenizes each distinct variant, or
    /// returns null when the prompt carries no scheduling — which keeps an unscheduled request on its pipeline's
    /// existing single-encode path rather than routing it through a one-variant schedule.</summary>
    /// <param name="tokenize">The caller's own tokenizer, including its instruction template. Called once per
    /// DISTINCT variant, so the weight grammar inside each branch is resolved per branch — <c>&lt;fromto[0.5]:
    /// (a:1.2), b&gt;</c> tokenizes its first variant per-span and its second in one call.</param>
    public static ScheduledPrompt? TryBuild(string? prompt, int totalSteps,
        Func<string, WeightedTokenSequence> tokenize)
    {
        ArgumentNullException.ThrowIfNull(tokenize);
        if (prompt is null || !PromptTagScheduling.HasScheduling(prompt))
        {
            return null;
        }
        PromptSchedule schedule = PromptTagScheduling.Resolve(prompt, totalSteps);
        List<WeightedTokenSequence> variants = new List<WeightedTokenSequence>(schedule.Variants.Count);
        foreach (string variant in schedule.Variants)
        {
            variants.Add(tokenize(variant));
        }
        return new ScheduledPrompt { Variants = variants, StepToVariant = schedule.StepToVariant };
    }
}
