using HartsyInference.Engine;
using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Dispatch;

/// <summary>Encapsulates loading and running one modality, wrapping the engine's per-pipeline wiring behind a
/// uniform surface that both the subcommands and the REPL drive.</summary>
public interface IModalityHandler
{
    /// <summary>The modality this handler serves.</summary>
    Modality Modality { get; }

    /// <summary>Loads the model named by <paramref name="spec"/> onto <paramref name="backend"/>, returning a runner.</summary>
    IModalityRunner Load(ModelSpec spec, IBackend backend, IProgressSink progress);

    /// <summary>Runs one generation with <paramref name="prompt"/> and the tunables in <paramref name="parameters"/>.</summary>
    GeneratedArtifact Run(IModalityRunner runner, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct);
}
