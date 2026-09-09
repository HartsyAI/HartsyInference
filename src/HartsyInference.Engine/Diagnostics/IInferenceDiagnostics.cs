using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Diagnostics;
/// <summary>Optional synchronous observer. Keep callbacks short; throwing observers are disabled.</summary>
/// <remarks>Text and image services emit request boundaries. Detailed prefill/token events depend on the text pipeline.</remarks>
public interface IInferenceDiagnostics
{
    /// <summary>Receives an event on the execution thread. Concurrent requests have distinct ids.</summary>
    void OnEvent(in InferenceDiagnosticEvent diagnostic);
    /// <summary>Reports the borrowed backend on the execution thread. Do not dispose or mutate it.</summary>
    void OnBackendReady(long requestId, IBackend backend);
}
