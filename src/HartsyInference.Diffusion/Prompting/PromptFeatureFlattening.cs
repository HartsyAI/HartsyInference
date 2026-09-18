namespace HartsyInference.Diffusion.Prompting;

/// <summary>The one place a service decides what a recipe's prompt string may still contain. SwarmUI's 2026-09-01
/// parser hands every backend <c>&lt;weight[N]:…&gt;</c>/<c>&lt;alternate:…&gt;</c>/<c>&lt;fromto[N]:…&gt;</c> as literal
/// tags, so they are resolved before any pipeline sees them: a family that can weight gets the <c>(text:N)</c> grammar
/// back, and a family that cannot gets the inner text alone — exactly what SwarmUI's own <c>join_text(leaves, False)</c>
/// does for an encoder it cannot weight, since leaving the digits in would feed them to the encoder as prose.</summary>
public static class PromptFeatureFlattening
{
    /// <summary>Resolves the tags <paramref name="mode"/> and <paramref name="preserveScheduling"/> say this family
    /// can still act on.</summary>
    /// <param name="preserveScheduling">True only for a recipe whose denoise loop consumes a multi-variant
    /// <see cref="ConditioningSchedule"/>; everything else collapses the tag to its step-0 value.</param>
    public static string Prepare(string? prompt, PromptWeightingMode mode, bool preserveScheduling = false) =>
        PromptTagFlattening.Flatten(prompt, !preserveScheduling, mode != PromptWeightingMode.None);

    /// <summary>Re-resolves a prompt already prepared for one family so a SECOND family can consume it — the generic
    /// refiner hand-off, where a weighting base's <c>(text:N)</c> grammar would otherwise reach a refiner whose
    /// encoder reads the parens and digits as prose. Tag flattening is not enough here: the tags are long gone and it
    /// is the parens that have to go.</summary>
    public static string Rebind(string? prompt, PromptWeightingMode mode) =>
        mode == PromptWeightingMode.None
            ? PromptWeighting.Join(PromptWeighting.Parse(prompt ?? "")) : prompt ?? "";
}
