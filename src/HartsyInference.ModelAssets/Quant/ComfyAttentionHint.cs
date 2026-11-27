using System.Text;
using System.Text.Json;

namespace HartsyInference.ModelAssets.Quant;

/// <summary>ComfyUI's per-block attention hint: a tiny U8 tensor named <c>{block}.comfy_attention.config</c> holding the UTF-8 JSON
/// <c>{"attention": "&lt;backend&gt;"}</c>. It names the attention backend ComfyUI runs that block with. It is not a weight, and a
/// loader that does not run that backend must say so rather than treat the blob as a tensor.</summary>
/// <remarks><para>Seen in the Comfy-Org Fibo 2.0 int8 builds (<c>layers.N.attention</c>, 35 bytes, <c>comfy_kitchen_int8</c>) and the
/// MiniMax-H3 image build (<c>blocks.N.attn</c>, 37 bytes). The value read from the Fibo 2.0 and RMBG 3.0 int8 files is the same on
/// every block; the bf16 builds carry no hint.</para></remarks>
public static class ComfyAttentionHint
{
    /// <summary>Key suffix carrying the hint.</summary>
    public const string Suffix = ".comfy_attention.config";

    /// <summary>Whether a checkpoint key is an attention hint rather than a tensor.</summary>
    public static bool IsHint(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return key.EndsWith(Suffix, StringComparison.Ordinal);
    }

    /// <summary>Parses the backend name from the blob's bytes, or returns null when the bytes are not the attention form. Trailing NUL
    /// padding is ignored, as for the other ComfyUI JSON blobs.</summary>
    public static string? TryParseBackend(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length is <= 0 or > 4096)
            return null;
        int end = bytes.Length;
        while (end > 0 && bytes[end - 1] == 0)
            end--;
        if (end == 0)
            return null;
        try
        {
            using JsonDocument doc = JsonDocument.Parse(Encoding.UTF8.GetString(bytes[..end]));
            JsonElement root = doc.RootElement;
            if (root.ValueKind == JsonValueKind.Object
                && root.TryGetProperty("attention", out JsonElement value)
                && value.ValueKind == JsonValueKind.String
                && value.GetString() is { Length: > 0 } backend)
            {
                return backend;
            }
        }
        catch (JsonException)
        {
            // Not JSON: the caller reports it with the key.
        }
        return null;
    }
}
