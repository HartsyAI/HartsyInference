namespace HartsyInference.Engine.Planning;

/// <summary>Which files a 3D model needs, which are present, which are missing and which the engine cannot fetch.</summary>
public sealed record MeshAssetStatus
{
    /// <summary>The model's catalog id.</summary>
    public required string ModelId { get; init; }

    /// <summary>Every file the catalog lists, plus those placed by hand; empty for a model with no catalog files.</summary>
    public required IReadOnlyList<MeshAssetInfo> Assets { get; init; }

    /// <summary>Set when the model as a whole has no downloadable files, with what to do.</summary>
    public MeshAssetBlocker? ModelBlocker { get; init; }

    /// <summary>Every required file is present. False for a model with no catalog files: nothing says it is installed.</summary>
    public bool Complete => Assets.Count > 0 && Assets.All(asset => !asset.Required || asset.Present);

    /// <summary>The absent files, required first.</summary>
    public IEnumerable<MeshAssetInfo> Missing => Assets.Where(asset => !asset.Present).OrderByDescending(asset => asset.Required);
}
