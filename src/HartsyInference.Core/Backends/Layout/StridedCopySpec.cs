namespace HartsyInference.Core.Backends.Layout;

/// <summary>The geometry of one <see cref="IBackend.CopyStrided"/> or <see cref="IBackend.AccumulateStrided"/> call: a
/// source view and a destination view with the same extents, so element <c>i</c> of the destination reads element
/// <c>i</c> of the source. Build one from a view with <see cref="Gather"/> (packed destination) or
/// <see cref="Scatter"/> (packed source), or pair two views for a strided-to-strided copy.</summary>
public readonly struct StridedCopySpec
{
    /// <summary>Pairs a destination view with a source view of the same extents.</summary>
    public StridedCopySpec(StridedView destination, StridedView source)
    {
        if (!destination.HasSameDims(source))
            throw new ArgumentException($"Destination {destination} and source {source} must have the same extents.", nameof(destination));
        Destination = destination;
        Source = source;
    }

    /// <summary>Where the elements are written.</summary>
    public StridedView Destination { get; }

    /// <summary>Where the elements are read.</summary>
    public StridedView Source { get; }

    /// <summary>Elements moved.</summary>
    public long ElementCount => Source.ElementCount;

    /// <summary>Reads <paramref name="source"/> into a row-major output of the same extents starting at element 0.</summary>
    public static StridedCopySpec Gather(StridedView source) => new(source.Packed(), source);

    /// <summary>Writes a row-major input of the same extents (from element 0) into <paramref name="destination"/>.</summary>
    public static StridedCopySpec Scatter(StridedView destination) => new(destination, destination.Packed());

    public override string ToString() => $"source {Source} -> destination {Destination}";
}
