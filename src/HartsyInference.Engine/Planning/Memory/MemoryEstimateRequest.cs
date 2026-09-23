using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Planning.Memory;

/// <summary>The geometry and memory overrides a VRAM estimate is taken for — the subset of an image or video request
/// that moves the memory peak, so a host can ask before it builds (or routes) the real request.</summary>
/// <param name="Width">Output width in pixels.</param>
/// <param name="Height">Output height in pixels.</param>
/// <param name="Frames">Output frame count; null takes the video family's own default (and 1 for a still image), so
/// a host that leaves the count to the recipe is estimated at the length the recipe will actually generate.</param>
/// <param name="Batch">Images or clips generated per forward.</param>
/// <param name="Vram">The same per-request overrides the generate request will carry, so the estimate is judged under
/// the policy the generation actually runs with. Null inherits the backend's policy.</param>
public readonly record struct MemoryEstimateRequest(int Width, int Height, int? Frames = null, int Batch = 1,
    VramOverrides? Vram = null)
{
    /// <summary>Pixels processed per generation (width x height x frames x batch): a single monotone size measure, so
    /// two requests for the same model compare by one number — a larger workload never needs less memory. An unset
    /// frame count counts as one; compare workloads of requests resolved the same way.</summary>
    public long Workload =>
        (long)Math.Max(1, Width) * Math.Max(1, Height) * Math.Max(1, Frames ?? 1) * Math.Max(1, Batch);
}
