using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Header-first video preflight surface shared by generation, CLI inspection, HTTP planning, and hosts.</summary>
public interface IVideoPlanningService
{
    /// <summary>Resolves checkpoint identity, effective settings, capabilities, and blocking issues without constructing model weights.</summary>
    /// <exception cref="Recipes.RecipeNotRegisteredException">The model's family has no video recipe: there are no
    /// defaults to plan with, so no plan is returned rather than one built on invented values.</exception>
    Task<VideoPlan> PlanAsync(ModelSpec spec, VideoRequest request, CancellationToken cancel = default);
}
