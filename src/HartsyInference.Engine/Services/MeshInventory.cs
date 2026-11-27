using HartsyInference.Engine.Planning;

namespace HartsyInference.Engine.Services;

/// <summary>A model's <see cref="MeshAssetStatus"/> with, for each catalog row, the catalog asset behind it and where it was found, so a prefetch needs no assumption about list order.</summary>
/// <param name="Status">The inventory a host sees.</param>
/// <param name="Catalog">One entry per row of <paramref name="Status"/>: its catalog asset, or null for a file placed by hand.</param>
/// <param name="Paths">One entry per row of <paramref name="Status"/>: the absolute path it was found at, or null when it is absent.</param>
internal sealed record MeshInventory(MeshAssetStatus Status, IReadOnlyList<ModelAsset?> Catalog, IReadOnlyList<string?> Paths);
