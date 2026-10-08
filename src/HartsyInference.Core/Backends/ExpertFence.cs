namespace HartsyInference.Core.Backends;

/// <summary>A backend event recorded when a lease is released, shared by the entries that lease pinned. Pooled by the cache.</summary>
internal sealed class ExpertFence
{
    public object Handle { get; set; } = null!;

    public int References { get; set; }
}
