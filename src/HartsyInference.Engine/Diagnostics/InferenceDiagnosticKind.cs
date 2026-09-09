namespace HartsyInference.Engine.Diagnostics;
/// <summary>Boundaries measured on the generation thread, not inferred from transport chunks.</summary>
public enum InferenceDiagnosticKind
{
    RequestStarted,
    ModelReady,
    PrefillCompleted,
    TokenGenerated,
    RequestCompleted,
}
