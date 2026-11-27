namespace HartsyInference.Audio.Models.Haru;

/// <summary>How the language model's big matrices are held. Everything else (the embedding table, the head, the projectors, the codec) is
/// used as the checkpoint stores it.</summary>
public enum HaruPrecision
{
    /// <summary>As stored: BF16, 10.5 GB for the whole checkpoint.</summary>
    Native,

    /// <summary>Q8_0 for every matrix.</summary>
    Q8_0,

    /// <summary>Q6_K for every matrix.</summary>
    Q6_K,

    /// <summary>Q5_K for the feed-forward matrices, Q6_K for the attention and delta-rule projections.</summary>
    Q5_K,

    /// <summary>Q4_K for the feed-forward matrices, Q6_K for the attention and delta-rule projections.</summary>
    Q4_K,
}
