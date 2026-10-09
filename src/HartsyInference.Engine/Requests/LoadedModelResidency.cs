using HartsyInference.Engine.Placement;

namespace HartsyInference.Engine.Requests;

/// <summary>A loaded model's residency plan, for the memory view: the device key it is loaded on, its checkpoint path, and the plan it was admitted under.</summary>
public sealed record LoadedModelResidency(string DeviceKey, string ModelPath, ResidencyPlan Plan);
