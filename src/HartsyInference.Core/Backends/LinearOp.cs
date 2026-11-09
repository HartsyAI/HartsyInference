using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>One projection in an <see cref="IBackend.LinearMulti"/> group: <c>Output = input × Weight^T + Bias</c>.
/// The input is the group's, not the op's — that shared input is the whole reason the grouping exists.</summary>
public readonly record struct LinearOp(Tensor Output, Tensor Weight, Tensor? Bias);
