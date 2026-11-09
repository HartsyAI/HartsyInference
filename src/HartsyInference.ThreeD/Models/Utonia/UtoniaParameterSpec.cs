namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>One checkpoint parameter a config implies: its PyTorch state-dict key, its C# name and its shape.</summary>
public readonly record struct UtoniaParameterSpec(string TorchKey, string CSharpName, long[] Shape);
