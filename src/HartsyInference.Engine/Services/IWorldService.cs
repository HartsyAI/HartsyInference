using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed interactive-world surface: opens a stateful session that streams frames in response to actions.</summary>
public interface IWorldService
{
    /// <summary>Opens an interactive session seeded by <paramref name="request"/>.</summary>
    IWorldSession Open(ModelSpec spec, WorldRequest request);
}
