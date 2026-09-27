namespace HartsyInference.Engine.Variants;

/// <summary>Which kind of evidence selected a <see cref="ResolvedModelVariant"/>, strongest first. The order is the
/// resolution order, so a weaker source is only ever consulted when every stronger one had nothing to say.</summary>
public enum ModelVariantSource
{
    /// <summary>The weights themselves: a marker tensor or structural rule the variant declares.</summary>
    Structure,

    /// <summary>The caller named the variant: <c>ModelSpec.Variant</c> (SwarmUI's model class), a <c>family:variant</c> selector, or a legacy family id.</summary>
    CallerHint,

    /// <summary>The file's own metadata: <c>modelspec.architecture</c>, the engine's <c>hartsy.model_id</c> stamp, or a family rule.</summary>
    Metadata,

    /// <summary>Whole-token filename match, a last resort that is always logged as a guess.</summary>
    Filename,

    /// <summary>Nothing matched; the family's declared default.</summary>
    Default,
}
