using System.Text.Json;

namespace HartsyInference.Core.Configuration;

/// <summary>Loads engine settings from a JSON file — the replacement for the environment variables the engine used to read.</summary>
/// <remarks>Discovered rather than passed, because the engine has no single entry point: the CLI has a
/// <c>Main</c>, but the SwarmUI extension is loaded as a library and never gets one, so anything requiring an
/// explicit startup call would silently do nothing there. Loading happens once, on the first knob resolution.
/// <para>Search order, first hit wins:</para>
/// <list type="number">
/// <item>the path in <see cref="ExplicitPath"/>, if a host set one;</item>
/// <item><c>hartsyinference.settings.json</c> in the current directory;</item>
/// <item><c>hartsyinference.settings.json</c> beside the entry assembly;</item>
/// <item><c>~/.config/hartsyinference/settings.json</c>.</item>
/// </list>
/// <para>File shape — a profile applied first, then individual settings on top, matching the CLI and the API:</para>
/// <code>
/// {
///   "profile": "reference",
///   "settings": { "numerics.ltx2TwoStage": true, "vram.int8RowBudgetMb": 256 }
/// }
/// </code>
/// <para>A malformed file THROWS rather than being skipped. A silently-ignored settings file is how an operator
/// ends up benchmarking a configuration they never actually applied.</para></remarks>
public static class KnobFile
{
    private const string FileName = "hartsyinference.settings.json";

    private static readonly object _gate = new();
    private static bool _loaded;

    /// <summary>Set by a host that knows where its settings live; overrides discovery. Must be set before the first knob is read.</summary>
    public static string? ExplicitPath { get; set; }

    /// <summary>The file actually loaded, or null when none was found. For diagnostics and the CLI banner.</summary>
    public static string? LoadedFrom { get; private set; }

    /// <summary>How many settings the file applied.</summary>
    public static int LoadedCount { get; private set; }

    /// <summary>Loads the settings file once. Safe to call repeatedly and from multiple threads.</summary>
    internal static void EnsureLoaded()
    {
        if (Volatile.Read(ref _loaded))
        {
            return;
        }
        lock (_gate)
        {
            if (_loaded)
            {
                return;
            }
            // Set BEFORE applying, so the Apply path's own knob reads cannot recurse into a second load.
            _loaded = true;
            string? path = Discover();
            if (path is not null)
            {
                Apply(File.ReadAllText(path), path);
                LoadedFrom = path;
            }
        }
    }

    /// <summary>Re-reads the settings file, discarding what a previous load applied. For tests and for a host that rewrote its file.</summary>
    public static void Reload()
    {
        lock (_gate)
        {
            _loaded = false;
            LoadedFrom = null;
            LoadedCount = 0;
            KnobStore.ResetOverrides();
        }
        EnsureLoaded();
    }

    private static string? Discover()
    {
        if (!string.IsNullOrWhiteSpace(ExplicitPath))
        {
            return File.Exists(ExplicitPath)
                ? ExplicitPath
                : throw new FileNotFoundException($"Engine settings file not found: '{ExplicitPath}'.", ExplicitPath);
        }
        string cwd = Path.Combine(Directory.GetCurrentDirectory(), FileName);
        if (File.Exists(cwd))
        {
            return cwd;
        }
        string beside = Path.Combine(AppContext.BaseDirectory, FileName);
        if (File.Exists(beside))
        {
            return beside;
        }
        string home = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".config", "hartsyinference", "settings.json");
        return File.Exists(home) ? home : null;
    }

    /// <summary>Parses and applies one settings document. Public so a host can supply settings it holds in memory.</summary>
    public static void Apply(string json, string origin = "(inline)")
    {
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(json, new JsonDocumentOptions
            {
                CommentHandling = JsonCommentHandling.Skip,
                AllowTrailingCommas = true,
            });
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException($"Engine settings file '{origin}' is not valid JSON: {ex.Message}", ex);
        }
        using (doc)
        {
            JsonElement root = doc.RootElement;
            int applied = 0;
            if (root.TryGetProperty("profile", out JsonElement profileElement)
                && profileElement.ValueKind == JsonValueKind.String)
            {
                string name = profileElement.GetString()!;
                KnobProfile profile = KnobProfiles.ByName(name)
                    ?? throw new InvalidOperationException(
                        $"Engine settings file '{origin}' names unknown profile '{name}'. "
                        + $"Known profiles: {string.Join(", ", KnobProfiles.Names)}.");
                foreach ((string id, object? value) in profile.Values)
                {
                    KnobStore.SetByIdRaw(id, value);
                    applied++;
                }
            }
            if (root.TryGetProperty("settings", out JsonElement settings)
                && settings.ValueKind == JsonValueKind.Object)
            {
                foreach (JsonProperty entry in settings.EnumerateObject())
                {
                    ApplyOne(entry, origin);
                    applied++;
                }
            }
            LoadedCount = applied;
        }
    }

    private static void ApplyOne(JsonProperty entry, string origin)
    {
        object? knob = KnobRegistry.Find(entry.Name)
            ?? throw new InvalidOperationException(
                $"Engine settings file '{origin}' sets unknown setting '{entry.Name}'. Run the CLI with --list-settings.");
        Type declared = knob.GetType().GetGenericArguments()[0];
        Type t = Nullable.GetUnderlyingType(declared) ?? declared;
        object? value = entry.Value.ValueKind switch
        {
            JsonValueKind.Null => null,
            JsonValueKind.True or JsonValueKind.False when t == typeof(bool) => entry.Value.GetBoolean(),
            JsonValueKind.Number when t == typeof(int) => entry.Value.GetInt32(),
            JsonValueKind.Number when t == typeof(long) => entry.Value.GetInt64(),
            JsonValueKind.Number when t == typeof(float) => entry.Value.GetSingle(),
            JsonValueKind.String when t == typeof(string) => entry.Value.GetString(),
            // Strings are accepted for every type so a value can be quoted, matching CLI --set.
            JsonValueKind.String => ParseString(entry.Value.GetString()!, t, entry.Name, origin),
            _ => throw new InvalidOperationException(
                $"Engine settings file '{origin}': setting '{entry.Name}' expects {t.Name}, got {entry.Value.ValueKind}."),
        };
        KnobStore.SetByIdRaw(entry.Name, value);
    }

    private static object ParseString(string raw, Type t, string id, string origin)
    {
        if (t == typeof(bool) && (raw is "1" or "0" || bool.TryParse(raw, out _)))
        {
            return raw is "1" || (bool.TryParse(raw, out bool b) && b);
        }
        if (t == typeof(int) && int.TryParse(raw, out int i)) { return i; }
        if (t == typeof(long) && long.TryParse(raw, out long l)) { return l; }
        if (t == typeof(float) && float.TryParse(raw, System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out float f)) { return f; }
        throw new InvalidOperationException(
            $"Engine settings file '{origin}': setting '{id}' expects {t.Name}, got '{raw}'.");
    }
}
