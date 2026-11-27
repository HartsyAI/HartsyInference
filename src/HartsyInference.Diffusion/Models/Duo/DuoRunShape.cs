namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>The Intent row counts a run folder holds: the shared rows, and the body rows of each block in order.</summary>
internal sealed record DuoRunShape(int SharedRows, IReadOnlyList<int> BodyRows);
