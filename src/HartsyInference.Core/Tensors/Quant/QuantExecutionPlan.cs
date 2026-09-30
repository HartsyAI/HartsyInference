namespace HartsyInference.Core.Tensors.Quant;

/// <summary>The execution path a backend picked for a recipe and the scratch memory it needs.</summary>
public readonly record struct QuantExecutionPlan(QuantExecutionKind Kind, long WorkspaceBytes);
