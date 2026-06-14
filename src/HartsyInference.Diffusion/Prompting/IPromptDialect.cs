namespace HartsyInference.Diffusion.Prompting;

/// <summary>Serializes a model-agnostic <see cref="StructuredPrompt"/> into a specific model's conditioning format (JSON for Ideogram 4, prose for plain-text models, per-region masks for regional-prompting models).</summary>
public interface IPromptDialect
{
    /// <summary>Dialect name (e.g. <c>"ideogram4"</c>).</summary>
    string Name { get; }

    /// <summary>Renders the prompt to the model's conditioning string.</summary>
    string Serialize(StructuredPrompt prompt);

    /// <summary>Optional per-region prompts for attention-coupled regional prompting. Default: none (the model takes the serialized string only).</summary>
    IReadOnlyList<RegionPrompt> SerializeRegions(StructuredPrompt prompt, int width, int height) => [];
}
