namespace HartsyInference.Engine.Features;

/// <summary>What a reference-image policy does with more images than the model addresses.</summary>
public enum ReferenceOverflow
{
    /// <summary>Keep the first ones and log the rest as dropped.</summary>
    Drop,

    /// <summary>Refuse the request.</summary>
    Refuse,
}
