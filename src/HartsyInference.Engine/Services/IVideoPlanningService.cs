using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Header-first video preflight surface shared by generation, CLI inspection, HTTP planning, and hosts.</summary>
public interface IVideoPlanningService
{
    /// <summary>Resolves checkpoint identity, effective settings, capabilities, and blocking issues without constructing model weights.</summary>
    Task<VideoPlan> PlanAsync(ModelSpec spec, VideoRequest request, CancellationToken cancel = default);
}
