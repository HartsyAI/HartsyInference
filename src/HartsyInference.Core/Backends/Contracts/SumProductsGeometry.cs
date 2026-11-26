namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The extents of a validated sum of products: the groups kept, the elements summed into each, the sums written per group, and the views
/// with their axes collapsed (the first operand is the source, the second the destination). Row-major element <c>i</c> of <see cref="Axes"/> belongs
/// to group <c>i / ElementsPerGroup</c>, also where a collapse crossed from the kept axes into the summed ones.</summary>
public readonly record struct SumProductsGeometry(long Groups, long ElementsPerGroup, int Terms, StridedCopyGeometry Axes);
