namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The extents of a validated sum of products: the groups kept, the elements summed into each, and the sums written per group.</summary>
public readonly record struct SumProductsGeometry(long Groups, long ElementsPerGroup, int Terms);
