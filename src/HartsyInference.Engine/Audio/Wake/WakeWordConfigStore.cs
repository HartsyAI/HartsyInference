using System.Text.Json;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio.Wake;

/// <summary>Per-word settings persisted beside the heads, so a wake word keeps its threshold, route and speaker
/// restriction across restarts instead of falling back to a global default.
///
/// <para>This matters most for trained words: <c>wake-train</c> measures a threshold for the head it just
/// produced, and without somewhere to record it that measurement is lost the moment the command exits and every
/// word ends up sharing one guessed number.</para>
///
/// <para>Uses <see cref="JsonSerializer"/> directly, matching <c>ModelCacheStore</c> and the rest of the engine;
/// there is no source-generated context anywhere in this codebase to follow.</para></summary>
public sealed class WakeWordConfigStore
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true,
    };

    private readonly string _path;
    private readonly object _lock = new();
    private Dictionary<string, WakeWordConfig> _entries = [];

    /// <summary>Config file path, <c>{modelRoot}/wake-words.json</c> by default.</summary>
    public string Path => _path;

    public WakeWordConfigStore(string modelRoot, string? fileName = null)
        => _path = System.IO.Path.Combine(modelRoot, fileName ?? "wake-words.json");

    /// <summary>Reads the file if present. A malformed file is logged and ignored rather than thrown: the
    /// listener starting with default settings is better than it refusing to start at all.</summary>
    public IReadOnlyDictionary<string, WakeWordConfig> Load()
    {
        lock (_lock)
        {
            if (!File.Exists(_path)) return _entries;
            try
            {
                string json = File.ReadAllText(_path);
                _entries = JsonSerializer.Deserialize<Dictionary<string, WakeWordConfig>>(json, JsonOptions) ?? [];
                Logs.Info($"[Audio][Wake] Loaded settings for {_entries.Count} wake word(s) from {_path}.");
            }
            catch (Exception ex)
            {
                Logs.Error($"[Audio][Wake] Could not read {_path}; using defaults.", ex);
                _entries = [];
            }
            return _entries;
        }
    }

    /// <summary>Adds or replaces one word's settings and rewrites the file.</summary>
    public void Set(string name, WakeWordConfig config)
    {
        lock (_lock)
        {
            _entries[name] = config;
            Save();
        }
    }

    /// <summary>Removes a word's settings. Returns false when it was not configured.</summary>
    public bool Remove(string name)
    {
        lock (_lock)
        {
            if (!_entries.Remove(name)) return false;
            Save();
            return true;
        }
    }

    /// <summary>Current settings, keyed by wake word.</summary>
    public IReadOnlyDictionary<string, WakeWordConfig> Entries
    {
        get { lock (_lock) return new Dictionary<string, WakeWordConfig>(_entries); }
    }

    private void Save()
    {
        try
        {
            string? directory = System.IO.Path.GetDirectoryName(_path);
            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
            // Write-then-move so a crash mid-write cannot leave a truncated file that fails to parse on the
            // next start.
            string temporary = _path + ".tmp";
            File.WriteAllText(temporary, JsonSerializer.Serialize(_entries, JsonOptions));
            File.Move(temporary, _path, overwrite: true);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][Wake] Could not write {_path}; settings will not survive a restart.", ex);
        }
    }
}
