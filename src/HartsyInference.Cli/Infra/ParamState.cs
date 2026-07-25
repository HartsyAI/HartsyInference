using System.Globalization;

namespace HartsyInference.Cli.Infra;

/// <summary>Mutable, printable bag of generation parameters shared by the subcommands and the interactive REPL.</summary>
/// <remarks>Holds the global selectors (backend, model, output) plus the per-modality tunable keys. A key seeded empty
/// means "not chosen" — it is sent to the engine as null so the model family's own official default applies.</remarks>
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

    /// <summary>Gets a tunable as an int, or null when the user never set it — the engine then applies the model family's own default.</summary>
    public int? GetIntOrNull(string key) =>
        _values.TryGetValue(key, out string? v) && int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out int i) ? i : null;

    /// <summary>Gets a tunable as a float, or null when the user never set it — the engine then applies the model family's own default.</summary>
    public float? GetFloatOrNull(string key) =>
        _values.TryGetValue(key, out string? v) && float.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out float f) ? f : null;

    /// <summary>Gets a tunable as a double, or null when the user never set it — the engine then applies the model family's own default.</summary>
    public double? GetDoubleOrNull(string key) =>
        _values.TryGetValue(key, out string? v) && double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) ? d : null;

    /// <summary>Gets a tunable as a non-empty string, or null when the user never set it.</summary>
    public string? GetStringOrNull(string key) =>
        _values.TryGetValue(key, out string? v) && v.Length > 0 ? v : null;

    /// <summary>Gets a tunable as a bool, falling back when unset or unparseable.</summary>
    public bool GetBool(string key, bool fallback) =>
        _values.TryGetValue(key, out string? v) && bool.TryParse(v, out bool b) ? b : fallback;

    private void ApplyDefaults(Modality modality)
    {
        _values.Clear();
        switch (modality)
        {
            case Modality.Image:
                _values["width"] = "";
                _values["height"] = "";
                _values["steps"] = "";
                _values["cfg"] = "";
                _values["sampler"] = "";
                _values["scheduler"] = "";
                _values["sigma-shift"] = "";
                _values["seed"] = "-1";
                _values["negative"] = "";
                break;
            case Modality.Text:
                _values["max-tokens"] = "256";
                _values["temperature"] = "0.7";
                _values["top-p"] = "0.95";
                _values["seed"] = "-1";
                _values["graph-decode"] = "false";
                _values["system"] = "";
                _values["image"] = "";
                _values["top-k"] = "";
                _values["min-p"] = "";
                _values["repetition-penalty"] = "";
                _values["thinking"] = "";
                _values["low-vram-quant"] = "false";
                _values["always-free-memory"] = "false";
                break;
            case Modality.Speech:
                _values["voice"] = "default";
                _values["speed"] = "1.0";
                _values["reference"] = "";
                _values["ref-text"] = "";
                _values["exaggeration"] = "";
                _values["nfe-step"] = "";
                _values["cfg-scale"] = "";
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
                _values["width"] = "";
                _values["height"] = "";
                _values["steps"] = "";
                _values["cfg"] = "";
                _values["frames"] = "";
                _values["fps"] = "";
                _values["negative"] = "";
                _values["seed"] = "-1";
                break;
            case Modality.Mesh:
                _values["grid"] = "0";
                _values["steps"] = "0";
                _values["seed"] = "-1";
                break;
            case Modality.World:
                _values["frames"] = "16";
                _values["steps"] = "10";
                _values["seed"] = "-1";
                break;
            case Modality.VoiceConvert:
                _values["target-path"] = "";
                _values["pitch-shift"] = "0";
                break;
            case Modality.Fx:
                _values["mode"] = "separate";
                _values["lambda"] = "";
                _values["tau"] = "";
                _values["seed"] = "-1";
                break;
        }
    }
}
