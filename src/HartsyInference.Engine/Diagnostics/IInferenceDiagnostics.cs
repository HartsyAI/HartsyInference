using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Diagnostics;
/// <summary>Optional synchronous observer. Callbacks must be short; do not dispose or mutate the borrowed backend. An observer that throws is disabled.</summary>
public interface IInferenceDiagnostics
{
    /// <summary>Receives an event on the execution thread. Concurrent requests have distinct ids.</summary>
    void OnEvent(in InferenceDiagnosticEvent diagnostic);
    /// <summary>Reports the backend actually used by this request, after construction, on its execution thread.</summary>
    void OnBackendReady(long requestId, IBackend backend);
}
