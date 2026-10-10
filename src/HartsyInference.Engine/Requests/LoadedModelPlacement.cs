using HartsyInference.Engine.Placement;

namespace HartsyInference.Engine.Requests;

/// <summary>A loaded text model and the placement the planner gave it.</summary>
/// <param name="DeviceKey">The slot's device key.</param>
/// <param name="ModelPath">The checkpoint.</param>
/// <param name="Placement">The plan the load followed.</param>
public sealed record LoadedModelPlacement(string DeviceKey, string ModelPath, TextPlacement Placement);
