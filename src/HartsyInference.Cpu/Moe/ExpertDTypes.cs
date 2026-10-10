using HartsyInference.Core.Tensors;

namespace HartsyInference.Cpu.Moe;

/// <summary>The packed format of each of an expert's three projections. A GGUF K-quant mix can store them differently: Q4_K_M
/// keeps gate and up in Q4_K and some layers' down projection in Q6_K.</summary>
/// <param name="Gate">Gate projection, <c>[I, H]</c>.</param>
/// <param name="Up">Up projection, <c>[I, H]</c>.</param>
/// <param name="Down">Down projection, <c>[H, I]</c>.</param>
public readonly record struct ExpertDTypes(DType Gate, DType Up, DType Down);
