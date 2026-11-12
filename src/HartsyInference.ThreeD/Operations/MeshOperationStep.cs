namespace HartsyInference.ThreeD.Operations;

/// <summary>One operation of a request and its numbers.</summary>
/// <param name="Operation">The operation id.</param>
/// <param name="Parameters">The numbers by name; a missing one takes the operation's default.</param>
public sealed record MeshOperationStep(string Operation, IReadOnlyDictionary<string, double>? Parameters = null);
