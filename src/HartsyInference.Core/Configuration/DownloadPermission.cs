using HartsyInference.Core.Exceptions;

namespace HartsyInference.Core.Configuration;

/// <summary>Whether this process may fetch model files. Every download asks here first, so a "no" at a prompt holds for the files a model load would
/// fetch on its own too.</summary>
/// <remarks>A library host and the API server are unrestricted unless <c>paths.allowDownloads</c> is off. The CLI refuses for a run that cannot ask, or
/// whose user said no. <see cref="Refuse"/> and <see cref="Grant"/> are for a process that serves one run at a time, as the CLI does: the state is
/// process-wide, because a load fetches from whichever thread runs it, and the scopes must be disposed in reverse order.</remarks>
public static class DownloadPermission
{
    private static readonly object _gate = new();
    private static string? _refusal;

    /// <summary>The reason a set <c>paths.allowDownloads=false</c> gives; see <see cref="Refuse"/>.</summary>
    public const string SettingOffReason = "paths.allowDownloads is off";

    /// <summary>True when the setting was given a value, true or false, by the settings file, the host or the request: that answer needs no prompt.</summary>
    public static bool IsDecided => KnobStore.SourceOf(EngineKnobs.AllowDownloads.Id) != "default";

    /// <summary>Throws unless downloading <paramref name="what"/> is allowed.</summary>
    /// <exception cref="DownloadRefusedException">The setting is off, or a <see cref="Refuse"/> is in force and the setting does not say yes.</exception>
    public static void Demand(string what)
    {
        string? why;
        lock (_gate)
        {
            why = IsDecided ? (EngineKnobs.AllowDownloads.Value ? null : SettingOffReason) : _refusal;
        }
        if (why is not null)
            throw new DownloadRefusedException($"'{what}' is not on disk and was not downloaded: {why}.");
    }

    /// <summary>Refuses downloads until the result is disposed. <paramref name="why"/> completes "was not downloaded: ".</summary>
    public static IDisposable Refuse(string why) => new Scope(why);

    /// <summary>Allows downloads until the result is disposed: the user asked for exactly these files.</summary>
    public static IDisposable Grant() => new Scope(null);

    private sealed class Scope : IDisposable
    {
        private readonly string? _previous;
        private bool _disposed;

        internal Scope(string? refusal)
        {
            lock (_gate)
            {
                _previous = _refusal;
                _refusal = refusal;
            }
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed)
                    return;
                _disposed = true;
                _refusal = _previous;
            }
        }
    }
}
