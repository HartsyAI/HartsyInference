namespace HartsyInference.Core.Logging;

/// <summary>Log severity levels. Ordering matches SwarmUI's convention so the extension can forward 1:1 without remapping.</summary>
public enum LogLevel : byte
{
    Verbose = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
}
