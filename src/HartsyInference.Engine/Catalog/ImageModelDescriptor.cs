using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Recipes;

namespace HartsyInference.Engine.Catalog;

/// <summary>An image family's descriptor: its recommended sampling defaults, the composition features and input limits it
/// applies, and how it weights prompts.</summary>
public sealed record ImageModelDescriptor : ModelDescriptor
{
    /// <summary>The values a request left unset resolves to.</summary>
    public required ImageDefaults Defaults { get; init; }

    /// <summary>The composition features the family applies for this variant.</summary>
    public required ImageFeatures Features { get; init; }

    /// <summary>How many input images the family reads, and whether references need an init image.</summary>
    public required ImageInputLimits InputLimits { get; init; }

    /// <summary>Which prompt-weighting mechanism the family's pipeline applies.</summary>
    public required PromptWeightingMode PromptWeighting { get; init; }
}
