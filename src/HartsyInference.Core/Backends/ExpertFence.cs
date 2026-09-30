namespace HartsyInference.Core.Backends;

/// <summary>A backend event recorded when a lease is released, shared by the entries that lease pinned.</summary>
internal sealed class ExpertFence
{
    public ExpertFence(object handle, int references)
    {
        Handle = handle;
        References = references;
    }

    public object Handle { get; }

    public int References { get; set; }
}
