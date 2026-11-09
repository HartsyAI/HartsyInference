namespace HartsyInference.Core.Distributed;

/// <summary>Where and how one process joins a multi-node job's rendezvous. Rank 0 listens on <see cref="MasterPort"/>; every other rank connects to <see cref="MasterAddress"/>.</summary>
public sealed record DistributedRendezvousOptions
{
    /// <summary>Total process count across all nodes; at least 2.</summary>
    public required int WorldSize { get; init; }

    /// <summary>This process's rank in [0, <see cref="WorldSize"/>). Rank 0 hosts the rendezvous.</summary>
    public required int Rank { get; init; }

    /// <summary>Host name or address that rank 0 is reachable at from the other ranks; rank 0 binds all interfaces.</summary>
    public required string MasterAddress { get; init; }

    /// <summary>TCP port of the rendezvous on rank 0.</summary>
    public required int MasterPort { get; init; }

    /// <summary>Shared secret every rank must present. Compared as a SHA-256 digest; it is not transport encryption.</summary>
    public required string JobToken { get; init; }

    /// <summary>Deadline for every rank to join. Covers the whole exchange, including retries while rank 0 is not yet listening.</summary>
    // TODO(unsure): 5 min default is a guess; real job start-up skew across nodes is unmeasured.
    public TimeSpan Timeout { get; init; } = TimeSpan.FromMinutes(5);

    /// <summary>Throws when a member is outside its contract.</summary>
    public void Validate()
    {
        if (WorldSize < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(WorldSize), WorldSize, "A distributed job needs at least 2 ranks.");
        }
        if (Rank < 0 || Rank >= WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(Rank), Rank, $"Rank must be in [0, {WorldSize}).");
        }
        if (MasterPort is < 1 or > 65535)
        {
            throw new ArgumentOutOfRangeException(nameof(MasterPort), MasterPort, "Master port must be in [1, 65535].");
        }
        if (string.IsNullOrWhiteSpace(MasterAddress))
        {
            throw new ArgumentException("Master address is required.", nameof(MasterAddress));
        }
        if (string.IsNullOrEmpty(JobToken))
        {
            throw new ArgumentException("Job token is required; every rank must present the same one.", nameof(JobToken));
        }
        if (Timeout <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(Timeout), Timeout, "Timeout must be positive.");
        }
    }
}
