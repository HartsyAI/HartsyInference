using System.Text.Json;

namespace HartsyInference.Core.Configuration;

/// <summary>Loads engine settings from a JSON file — the replacement for the environment variables the engine used to read.</summary>
/// <remarks>Discovered rather than passed, because the engine has no single entry point: the CLI has a
/// <c>Main</c>, but the SwarmUI extension is loaded as a library and never gets one, so anything requiring an
/// explicit startup call would silently do nothing there. Loading happens once, on the first knob resolution.
/// <para>There is exactly ONE file: <c>~/.config/hartsyinference/settings.json</c> (see <see cref="Path"/>),
/// unless a host points <see cref="ExplicitPath"/> somewhere else. It used to be searched for in the working
/// directory and beside the entry assembly too, which meant the settings that applied depended on where the
/// process happened to be started from — the same engine could read a different file per host, which is the
/// opposite of what a settings file is for.</para>
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
    private static readonly object _gate = new();
    private static bool _loaded;

    /// <summary>Set by a host that keeps its settings elsewhere; overrides <see cref="Path"/>. Must be set before the first knob is read.</summary>
    public static string? ExplicitPath { get; set; }

    /// <summary>The settings file this process reads and writes, whether or not it exists yet.</summary>
    public static string Path => string.IsNullOrWhiteSpace(ExplicitPath)
        ? System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".config", "hartsyinference", "settings.json")
        : ExplicitPath;

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
        if (!string.IsNullOrWhiteSpace(ExplicitPath) && !File.Exists(ExplicitPath))
        {
            throw new FileNotFoundException($"Engine settings file not found: '{ExplicitPath}'.", ExplicitPath);
        }
        return File.Exists(Path) ? Path : null;
    }

    /// <summary>Writes one setting to <see cref="Path"/> and applies it, so it survives a restart.</summary>
    /// <remarks>Validates through the same parse the file load uses, so an unknown id, a wrong type or a value
    /// outside a clamped range fails here rather than at the next startup. The rest of the document is preserved
    /// — the <c>profile</c> line and any other settings — because a user editing one value must not silently drop
    /// the others. Written to a temporary file and renamed, so a crash mid-write cannot leave a truncated file
    /// that the next load would reject outright.
    /// <para>Returns the parsed value that was stored. A <see cref="KnobScope.Construction"/> setting is written
    /// but does not take effect until the process restarts; the caller is expected to say so.</para></remarks>
    public static object? Save(string id, string rawValue)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        object knob = KnobRegistry.Find(id)
            ?? throw new InvalidOperationException($"Unknown setting '{id}'. Run 'hartsy settings list' to see them all.");

        // Round-trip through the loader's own parse so "true"/"1"/"256" are read exactly as the file would read them.
        string probe = "{\"settings\":{" + JsonSerializer.Serialize(id) + ":"
            + JsonSerializer.Serialize(rawValue) + "}}";
        using (JsonDocument parsed = JsonDocument.Parse(probe))
        {
            ApplyOne(parsed.RootElement.GetProperty("settings").EnumerateObject().First(), "(set)");
        }
        // The effective value, not the raw one: two knobs CLAMP rather than reject, and a file holding
        // a number the engine would quietly narrow is exactly the kind of lie this rewrite is removing.
        object? stored = KnobRegistry.ValueOf(knob);
        KnobStore.SetByIdRaw(id, stored, "settings file");

        Dictionary<string, JsonElement> settings = new(StringComparer.Ordinal);
        string? profile = null;
        if (File.Exists(Path))
        {
            using JsonDocument existing = JsonDocument.Parse(File.ReadAllText(Path), new JsonDocumentOptions
            {
                CommentHandling = JsonCommentHandling.Skip,
                AllowTrailingCommas = true,
            });
            if (existing.RootElement.TryGetProperty("profile", out JsonElement p) && p.ValueKind == JsonValueKind.String)
            {
                profile = p.GetString();
            }
            if (existing.RootElement.TryGetProperty("settings", out JsonElement s) && s.ValueKind == JsonValueKind.Object)
            {
                foreach (JsonProperty entry in s.EnumerateObject())
                {
                    settings[entry.Name] = entry.Value.Clone();
                }
            }
        }

        using (MemoryStream buffer = new())
        {
            using (Utf8JsonWriter writer = new(buffer, new JsonWriterOptions { Indented = true }))
            {
                writer.WriteStartObject();
                if (profile is not null)
                {
                    writer.WriteString("profile", profile);
                }
                writer.WriteStartObject("settings");
                foreach ((string key, JsonElement value) in settings.Where(kv => !string.Equals(kv.Key, id, StringComparison.Ordinal))
                             .OrderBy(kv => kv.Key, StringComparer.Ordinal))
                {
                    writer.WritePropertyName(key);
                    value.WriteTo(writer);
                }
                writer.WritePropertyName(id);
                WriteValue(writer, stored);
                writer.WriteEndObject();
                writer.WriteEndObject();
            }
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(Path)!);
            string temp = Path + ".tmp";
            File.WriteAllBytes(temp, buffer.ToArray());
            File.Move(temp, Path, overwrite: true);
        }
        LoadedFrom = Path;
        return stored;
    }

    private static void WriteValue(Utf8JsonWriter writer, object? value)
    {
        switch (value)
        {
            case null: writer.WriteNullValue(); break;
            case bool b: writer.WriteBooleanValue(b); break;
            case int i: writer.WriteNumberValue(i); break;
            case long l: writer.WriteNumberValue(l); break;
            case float f: writer.WriteNumberValue(f); break;
            default: writer.WriteStringValue(value.ToString()); break;
        }
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
                    KnobStore.SetByIdRaw(id, value, "settings file");
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
        KnobStore.SetByIdRaw(entry.Name, value, "settings file");
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
