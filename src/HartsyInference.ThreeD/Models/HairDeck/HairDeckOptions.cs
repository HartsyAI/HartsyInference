namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>Generation options of one HairDeck request.</summary>
public sealed record HairDeckOptions
{
    /// <summary>RNG seed of the first sample.</summary>
    public long Seed { get; init; }

    /// <summary>Euler steps of the image model (the reference uses 50).</summary>
    public int Steps { get; init; } = 50;

    /// <summary>Classifier-free guidance scale (the reference uses 2).</summary>
    public double Cfg { get; init; } = 2.0;

    /// <summary>Number of samples (variations with seeds Seed, Seed + 1, ...); this build generates exactly one.</summary>
    public int Samples { get; init; } = 1;

    /// <summary>Colour the cards with the colour model (one hair colour from the input when off).</summary>
    public bool Colors { get; init; } = true;

    /// <summary>Also generate spring-bone chains with the chain model.</summary>
    public bool Chains { get; init; } = true;

    /// <summary>Rigged output: the cards and chain tubes skinned to the spring chains, with four head-motion clips. Needs
    /// <see cref="Chains"/>; it is a glTF rig, not VRM.</summary>
    public bool Rigged { get; init; }

    /// <summary>Checks the options against what this build implements. Unsupported requests fail here rather than being changed silently.</summary>
    /// <exception cref="NotSupportedException">More than one sample, or rigged output without chains.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Steps below 1 or a guidance scale that is not positive.</exception>
    public HairDeckOptions Validate()
    {
        if (Samples != 1)
        {
            throw new NotSupportedException($"HairDeck generates one sample per request in this build; {Samples} samples were requested.");
        }

        if (Rigged && !Chains)
        {
            throw new NotSupportedException("HairDeck rigged output needs the spring chains; do not turn chains off with rigged output.");
        }

        if (Steps < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(Steps), Steps, "HairDeck needs at least one step.");
        }

        if (!(Cfg > 0.0) || double.IsNaN(Cfg) || double.IsInfinity(Cfg))
        {
            throw new ArgumentOutOfRangeException(nameof(Cfg), Cfg, "The guidance scale must be a positive finite number.");
        }

        return this;
    }
}
