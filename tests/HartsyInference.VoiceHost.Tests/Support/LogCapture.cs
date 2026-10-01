using HartsyInference.Core.Logging;

namespace HartsyInference.VoiceHost.Tests.Support;

/// <summary>Captures every log line (at Debug and above) until disposed, then restores the default logger and level.</summary>
internal sealed class LogCapture : IDisposable
{
    private readonly List<string> _lines = [];
    private readonly LogLevel _previousLevel = Logs.MinLevel;

    public LogCapture()
    {
        Logs.MinLevel = LogLevel.Debug;
        Logs.SetLogger((level, message) =>
        {
            lock (_lines)
            {
                _lines.Add($"{level}: {message}");
            }
        });
    }

    public string[] Lines
    {
        get
        {
            lock (_lines)
            {
                return [.. _lines];
            }
        }
    }

    public string All => string.Join('\n', Lines);

    public void Dispose()
    {
        Logs.SetLogger(null!);
        Logs.MinLevel = _previousLevel;
    }
}
