namespace HartsyInference.Core.Backends.Attention;

/// <summary>The <c>Depth × Height × Width</c> window each query of <see cref="IBackend.NeighborhoodAttention"/> attends. The
/// window slides inward at the borders (NATTEN's rule) so every query sees exactly this many keys, and an extent longer than
/// its axis collapses to the whole axis.</summary>
public readonly record struct NeighborhoodWindow(int Depth, int Height, int Width);
