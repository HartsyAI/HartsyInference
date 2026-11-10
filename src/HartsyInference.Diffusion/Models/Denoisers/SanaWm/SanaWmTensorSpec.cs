using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>One tensor the SANA-WM DiT expects: its upstream key, the dtype the converter stores it as, and its shape.</summary>
public readonly record struct SanaWmTensorSpec(string Name, DType DType, TensorShape Shape);
