namespace HartsyInference.API.Endpoints;

/// <summary>One operation of <see cref="NativeMeshOperateRequest"/>.</summary>
public sealed class NativeMeshOperationStep
{
    /// <summary>The operation id (<c>decimate</c>, <c>fill-holes</c>, <c>remove-small-components</c>, <c>smooth</c>, <c>recompute-normals</c>, <c>recenter</c>, <c>rescale</c>).</summary>
    public string? Operation { get; set; }

    /// <summary>The operation's numbers by name; a missing one takes its default.</summary>
    public Dictionary<string, double>? Parameters { get; set; }
}
