using HartsyInference.Engine.Dispatch;

namespace HartsyInference.Engine.Services;

/// <summary>Typed decisions: a state and a schema of choice / score / true-false questions answered in one forward pass, in the
/// Jev / SystemOne request and response shape. Backed by Cloudflare Clef (<see cref="DecisionService"/>).</summary>
public interface IDecisionService
{
    /// <summary>Answers one <c>/v1/systemone</c> request body and returns the response body, both as JSON text.</summary>
    Task<string> DecideAsync(ModelSpec spec, string requestJson, CancellationToken cancel = default);
}
