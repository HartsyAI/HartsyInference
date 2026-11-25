using System.Security.Cryptography;
using System.Text.Json;

namespace HartsyInference.ModelAssets.Duo;

/// <summary>The sha256 manifest written beside a Duo repack: each output file with its size and hash, the source files it was made from, and free-form notes (licence, pins) the caller supplies.</summary>
public static class DuoRepackManifest
{
    public const string FileName = "duo-repack-manifest.json";

    /// <summary>Hashes <paramref name="relativePaths"/> under <paramref name="folder"/> and writes <see cref="FileName"/> there. Returns the hashed files.</summary>
    public static IReadOnlyList<DuoRepackFile> Write(string folder, IEnumerable<string> relativePaths,
        IReadOnlyDictionary<string, string> sources, IReadOnlyDictionary<string, string> notes)
    {
        ArgumentException.ThrowIfNullOrEmpty(folder);
        ArgumentNullException.ThrowIfNull(relativePaths);
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentNullException.ThrowIfNull(notes);

        List<DuoRepackFile> files = [];
        foreach (string relative in relativePaths.Order(StringComparer.Ordinal))
        {
            string full = Path.Combine(folder, relative);
            files.Add(new DuoRepackFile(relative.Replace('\\', '/'), new FileInfo(full).Length, HashFile(full)));
        }

        string path = Path.Combine(folder, FileName);
        using FileStream stream = File.Create(path);
        using Utf8JsonWriter json = new(stream, new JsonWriterOptions { Indented = true });
        json.WriteStartObject();
        json.WriteNumber("schema", 1);
        json.WriteStartObject("notes");
        foreach (KeyValuePair<string, string> note in notes.OrderBy(static n => n.Key, StringComparer.Ordinal)) json.WriteString(note.Key, note.Value);
        json.WriteEndObject();
        json.WriteStartObject("sources");
        foreach (KeyValuePair<string, string> source in sources.OrderBy(static s => s.Key, StringComparer.Ordinal)) json.WriteString(source.Key, source.Value);
        json.WriteEndObject();
        json.WriteStartArray("files");
        foreach (DuoRepackFile file in files)
        {
            json.WriteStartObject();
            json.WriteString("path", file.RelativePath);
            json.WriteNumber("bytes", file.Bytes);
            json.WriteString("sha256", file.Sha256);
            json.WriteEndObject();
        }
        json.WriteEndArray();
        json.WriteEndObject();
        return files;
    }

    /// <summary>Lower-case hex SHA-256 of a file.</summary>
    public static string HashFile(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }
}
