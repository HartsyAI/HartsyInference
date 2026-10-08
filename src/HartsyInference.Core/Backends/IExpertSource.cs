namespace HartsyInference.Core.Backends;

/// <summary>
/// Supplies one expert's weights on demand. A model describes where its experts come from; the bank and cache decide when
/// they are resolved and where they are copied. Implementations must return weights whose <see cref="ExpertWeights.Key"/>
/// equals the requested key.
/// </summary>
public interface IExpertSource
{
    /// <summary>Where the authoritative bytes of these experts live.</summary>
    ExpertBacking Backing { get; }

    /// <summary>Resolves one expert.</summary>
    ExpertWeights Resolve(ExpertKey key);
}
