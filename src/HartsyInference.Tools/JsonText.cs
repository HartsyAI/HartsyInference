using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace HartsyInference.Tools;

/// <summary>JSON text helpers for prompt-bound output: the writer keeps <c>'</c>, <c>+</c> and non-ASCII characters literal (the default encoder's <c>'</c>-style escapes are HTML hardening a model does not need to read through).</summary>
internal static class JsonText
{
    /// <summary>Writer options for every JSON this package emits.</summary>
    public static readonly JsonWriterOptions WriterOptions = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    /// <summary>Runs <paramref name="write"/> against a fresh writer and returns the JSON text.</summary>
    public static string Write(Action<Utf8JsonWriter> write)
    {
        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer, WriterOptions))
        {
            write(writer);
        }
        return Encoding.UTF8.GetString(buffer.GetBuffer(), 0, (int)buffer.Length);
    }
}
