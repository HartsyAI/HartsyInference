using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe;

/// <summary>One routed expert in a layer's plan: where it runs, and how many (token, slot) pairs it serves.</summary>
/// <param name="Key">The expert.</param>
/// <param name="Placement">Where its rows run.</param>
/// <param name="Rows">Token-slot pairs routed to it in this batch; always at least 1.</param>
public readonly record struct ExpertAssignment(ExpertKey Key, ExpertPlacement Placement, int Rows);
