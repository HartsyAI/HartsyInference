namespace HartsyInference.LLM.Kimi;

/// <summary>The part of a Kimi K3.1 decoder one tensor-parallel rank holds: rank <see cref="Rank"/> of <see cref="Ranks"/>.</summary>
public readonly record struct KimiK31Shard
{
    /// <summary>Validates the rank against the rank count.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="ranks"/> is below 1, or <paramref name="rank"/> is outside [0, ranks).</exception>
    public KimiK31Shard(int rank, int ranks)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(ranks, 1);
        if ((uint)rank >= (uint)ranks) throw new ArgumentOutOfRangeException(nameof(rank), rank, $"Rank must lie in [0, {ranks}).");
        Rank = rank;
        Ranks = ranks;
    }

    /// <summary>This rank's index.</summary>
    public int Rank { get; }

    /// <summary>Ranks the decoder is split over.</summary>
    public int Ranks { get; }

    /// <summary>The whole decoder on one rank.</summary>
    public static KimiK31Shard Whole => new(0, 1);
}
