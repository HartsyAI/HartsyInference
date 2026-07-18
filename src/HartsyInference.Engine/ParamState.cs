using System.Globalization;

namespace HartsyInference.Engine;

/// <summary>Mutable, printable bag of generation parameters shared by the subcommands and the interactive REPL.
/// Holds the global selectors (backend, model, output) plus per-modality tunables seeded with sensible defaults.</summary>
public sealed class ParamState
{
    private readonly Dictionary<string, string> _values = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Creates a param state seeded with the defaults for <paramref name="modality"/>.</summary>
    public ParamState(Modality modality)
    {
        Modality = modality;
        ApplyDefaults(modality);
    }

    /// <summary>The modality whose defaults and tunables are currently loaded.</summary>
    public Modality Modality { get; private set; }

    /// <summary>Backend selector: auto, cpu, cuda, or vulkan.</summary>
    public string Backend { get; set; } = "auto";

    /// <summary>Selected model id (catalog id) or a local/HF path; null means "use the modality default".</summary>
    public string? Model { get; set; }

    /// <summary>Directory generated artifacts are written to; null means the repo/cwd default.</summary>
    public string? OutputDir { get; set; }

    /// <summary>The per-modality tunable keys and their current string values.</summary>
    public IReadOnlyDictionary<string, string> Values => _values;

    /// <summary>Re-seeds the tunables for a new modality (used when the REPL switches command family).</summary>
    public void SwitchModality(Modality modality)
    {
        Modality = modality;
        ApplyDefaults(modality);
    }

    /// <summary>Restores the tunables to the defaults for the current modality, leaving backend/model/output intact.</summary>
    public void Reset() => ApplyDefaults(Modality);

    /// <summary>Sets a tunable. Returns false when the key is not a recognized tunable for the current modality.</summary>
    public bool TrySet(string key, string value)
    {
        if (!_values.ContainsKey(key))
            return false;
        _values[key] = value;
        return true;
    }

    /// <summary>Adds or overwrites a tunable regardless of whether it was seeded (used by command binders).</summary>
    public void Put(string key, string value) => _values[key] = value;

    /// <summary>Gets a tunable's raw string value, or null when unset.</summary>
    public string? Get(string key) => _values.TryGetValue(key, out string? v) ? v : null;

    /// <summary>Gets a tunable as an int, falling back when unset or unparseable.</summary>
    public int GetInt(string key, int fallback) =>
        _values.TryGetValue(key, out string? v) && int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out int i) ? i : fallback;

    /// <summary>Gets a tunable as a float, falling back when unset or unparseable.</summary>
    public float GetFloat(string key, float fallback) =>
        _values.TryGetValue(key, out string? v) && float.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out float f) ? f : fallback;

    /// <summary>Gets a tunable as a bool, falling back when unset or unparseable.</summary>
    public bool GetBool(string key, bool fallback) =>
        _values.TryGetValue(key, out string? v) && bool.TryParse(v, out bool b) ? b : fallback;

    private void ApplyDefaults(Modality modality)
    {
        _values.Clear();
        switch (modality)
        {
            case Modality.Image:
                _values["width"] = "1024";
                _values["height"] = "1024";
                _values["steps"] = "20";
                _values["cfg"] = "7.5";
                _values["seed"] = "-1";
                _values["negative"] = "";
                break;
            case Modality.Text:
                _values["max-tokens"] = "256";
                _values["temperature"] = "0.7";
                _values["top-p"] = "0.95";
                _values["seed"] = "-1";
                _values["graph-decode"] = "false";
                break;
            case Modality.Speech:
                _values["voice"] = "default";
                _values["speed"] = "1.0";
                break;
            case Modality.Music:
                _values["duration"] = "10";
                _values["seed"] = "-1";
                break;
            case Modality.Transcribe:
                _values["language"] = "en";
                _values["translate"] = "false";
                _values["timestamps"] = "false";
                break;
            case Modality.Vision:
                _values["mode"] = "embed";
                _values["confidence"] = "0.25";
                break;
            case Modality.Video:
                _values["width"] = "704";
                _values["height"] = "480";
                _values["steps"] = "30";
                _values["frames"] = "25";
                _values["fps"] = "25";
                _values["negative"] = "";
                _values["seed"] = "-1";
                break;
            case Modality.ThreeD:
                _values["grid"] = "0";
                _values["steps"] = "0";
                _values["seed"] = "-1";
                break;
            case Modality.Interactive:
                _values["frames"] = "16";
                _values["steps"] = "10";
                _values["seed"] = "-1";
                break;
        }
    }
}
