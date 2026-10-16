using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>One logical weight of an <see cref="HfQuantCheckpoint"/>.</summary>
/// <param name="Name">Logical name, ending in <c>.weight</c>.</param>
/// <param name="Shape">Logical <c>[out, in]</c> (or stored) shape.</param>
/// <param name="Storage">How the checkpoint stores it.</param>
/// <param name="StoredDType">Stored dtype of the main tensor (I32 for a GPTQ <c>qweight</c>).</param>
/// <param name="SourceKey">The stored tensor (or GPTQ module prefix) it reads.</param>
/// <param name="StackedSource">The fused expert stack it is a slice of, or null.</param>
/// <param name="StackedExpert">Expert index within <paramref name="StackedSource"/>.</param>
/// <param name="StackedRowOffset">First row within the expert's slab (the up half of a gate/up stack starts past the gate rows).</param>
public sealed record HfQuantWeight(string Name, TensorShape Shape, HfWeightStorage Storage, DType StoredDType, string SourceKey,
    string? StackedSource = null, int StackedExpert = 0, long StackedRowOffset = 0);
