namespace HartsyInference.Diffusion.Prompting.MagicPrompt;

/// <summary>Optionally expands a plain-text idea into a full <see cref="StructuredPrompt"/> via an LLM ("magic prompt"). The core builder + dialects work without one; concrete expanders (Claude API, dotLLM, local Gemma) live behind this interface so the prompt package carries ZERO LLM dependency. Upstream caveat: the open magic-prompt is not the same as Ideogram.ai's production one, so results differ.</summary>
public interface IMagicPromptExpander
{
    /// <summary>Expands a plain-text prompt into a structured prompt.</summary>
    Task<StructuredPrompt> ExpandAsync(string plainPrompt, CancellationToken cancellationToken = default);
}
