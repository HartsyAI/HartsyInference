using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning.Memory;

namespace HartsyInference.Engine.Services;

/// <summary>Header-only VRAM estimation: what a generation will need, and whether it fits this engine's device, decided
/// before any weight is loaded.</summary>
/// <remarks>Built for routing. Checkpoint headers are read once per process and cached, so after the first call for a
/// model every answer is arithmetic — cheap enough to ask for every queued request against every device.</remarks>
public interface IMemoryEstimationService
{
    /// <summary>The per-phase memory <paramref name="spec"/> needs at <paramref name="request"/>'s geometry, sized as the
    /// checkpoint stores it (no device-specific widening).</summary>
    Task<MemoryEstimate> EstimateAsync(ModelSpec spec, MemoryEstimateRequest request, CancellationToken cancel = default);

    /// <summary>Whether <paramref name="spec"/> fits this engine's device under the effective VRAM policy (the backend's
    /// policy with <see cref="MemoryEstimateRequest.Vram"/> applied), its placement, and the memory levers the model
    /// actually wires.</summary>
    Task<MemoryFit> AssessAsync(ModelSpec spec, MemoryEstimateRequest request, CancellationToken cancel = default);
}
