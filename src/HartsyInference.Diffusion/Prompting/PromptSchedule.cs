using System.Collections.Generic;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>The result of resolving a scheduled prompt: the distinct prompt strings that occur across the denoise steps, plus a per-step index into that list. Pipelines encode each variant once and select by step.</summary>
public readonly record struct PromptSchedule(IReadOnlyList<string> Variants, int[] StepToVariant);
