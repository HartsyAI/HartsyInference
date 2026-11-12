namespace HartsyInference.ThreeD.Operations;

/// <summary>One number an operation takes, described so a host can build a form and the engine can validate a request without knowing the operation.</summary>
/// <param name="Name">The parameter name in a request.</param>
/// <param name="Description">What it does, in one sentence.</param>
/// <param name="Default">The value used when the request leaves it out, or null when there is none (the operation then says what it needs).</param>
/// <param name="Min">The smallest accepted value.</param>
/// <param name="Max">The largest accepted value.</param>
/// <param name="Integer">Whether the value must be a whole number.</param>
public sealed record MeshOperationParameter(string Name, string Description, double? Default, double Min, double Max, bool Integer = false);
