namespace SharpInference.Core.Logging;

/// <summary>Log severity levels. Ordering matches SwarmUI's convention so the extension
/// can forward 1:1 without remapping: Verbose is the chattiest (per-step progress, every
/// tile, every cache decision), Debug is developer detail, Info is user-visible milestones,
/// Warning is non-fatal anomalies (e.g. recovered OOM retries), Error is generation failure.</summary>
public enum LogLevel : byte
{
    Verbose = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
}
