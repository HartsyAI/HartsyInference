using System.IO.Compression;
using System.Reflection;
using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>Opens the gzip-compressed data files the text frontends embed (dictionaries, tagger and segmenter tables,
/// small model weights) by their logical resource name.</summary>
internal static class EmbeddedTextResource
{
    private const string Prefix = "HartsyInference.Audio.Frontends.Resources.";

    /// <summary>Decompressing stream over the embedded <paramref name="name"/> (a <c>.gz</c> resource).</summary>
    public static Stream OpenGzip(string name)
    {
        Assembly asm = typeof(EmbeddedTextResource).Assembly;
        Stream raw = asm.GetManifestResourceStream(Prefix + name)
            ?? throw new InvalidOperationException($"Embedded resource '{Prefix + name}' is missing.");
        return new GZipStream(raw, CompressionMode.Decompress);
    }

    /// <summary>Every line of the embedded gzip text resource <paramref name="name"/> (UTF-8).</summary>
    public static IEnumerable<string> ReadLines(string name)
    {
        using Stream stream = OpenGzip(name);
        using StreamReader reader = new(stream, Encoding.UTF8);
        string? line;
        while ((line = reader.ReadLine()) is not null) yield return line;
    }
}
