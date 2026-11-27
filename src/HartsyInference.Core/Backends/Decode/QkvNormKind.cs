namespace HartsyInference.Core.Backends.Decode;

/// <summary>Which RMS normalization a <see cref="QkvDecodeCall"/> applies to q and k before rotating them.</summary>
public enum QkvNormKind
{
    /// <summary>The sections are rotated as projected.</summary>
    None,

    /// <summary>Each head is normalized over its <c>headDim</c> elements with a <c>[headDim]</c> weight (Qwen3, Gemma 3).</summary>
    PerHead,

    /// <summary>All q heads are normalized as one row with a <c>[heads·headDim]</c> weight, and all k heads likewise (OLMoE).</summary>
    FullWidth,
}
