using System.Text;
using System.Text.Json;

namespace HartsyInference.ModelAssets.Quant;

/// <summary>A ComfyUI <c>{layer}.comfy_quant</c> blob: a tiny U8 tensor holding the UTF-8 JSON that declares how that
/// one layer was quantized.</summary>
/// <remarks><para>The per-layer blob is authoritative, not the file-level <c>__metadata__._quantization_metadata</c>
/// mirror: re-quants of the same model skip different layer sets (the official Lightricks LTX 2.5 build and
/// <c>DmitryDB/LTX-2.5-ComfyUI-Quants</c> disagree), and a loader that reads the file-level copy would apply one
/// layer's format to another's weight.</para>
/// <para>Observed shapes: <c>{"format":"float8_e4m3fn"}</c>,
/// <c>{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256,"per_row":true}</c>,
/// <c>{"format":"nvfp4"}</c>. ComfyUI also reads <c>convrot</c>/<c>convrot_groupsize</c> out of a nested
/// <c>params</c> object, so both spellings are accepted.</para></remarks>
public sealed record ComfyQuantDescriptor
{
    /// <summary>Key suffix carrying the descriptor.</summary>
    public const string Suffix = ".comfy_quant";

    /// <summary>ComfyUI's format name — the key into its own <c>QUANT_ALGOS</c> table.</summary>
    public required string Format { get; init; }

    /// <summary>Hadamard rotation group size, or 0 when this layer was quantized unrotated.</summary>
    public int ConvRotGroupSize { get; init; }

    /// <summary>The layer opts out of the quantized GEMM and must be dequantized first.</summary>
    /// <remarks>ComfyUI honours this per layer (<c>ops.py</c> <c>_full_precision_mm_config</c>); MiniMax-H3's
    /// <c>mlp.fc2</c> is the known case, and dropping it silently costs accuracy with no error.</remarks>
    public bool FullPrecisionMatMul { get; init; }

    /// <summary>Parses the blob, or returns null when it is not a descriptor tensor or its JSON is unreadable.</summary>
    public static ComfyQuantDescriptor? TryParse(ReadOnlySpan<byte> json)
    {
        if (json.Length is <= 0 or > 4096) return null;
        try
        {
            using JsonDocument doc = JsonDocument.Parse(Encoding.UTF8.GetString(json));
            JsonElement root = doc.RootElement;
            // ValueKind is checked before GetString: on a non-string `format` that accessor throws
            // InvalidOperationException, which is not a JsonException and would escape as a load crash.
            if (root.ValueKind != JsonValueKind.Object
                || !root.TryGetProperty("format", out JsonElement formatElement)
                || formatElement.ValueKind != JsonValueKind.String
                || formatElement.GetString() is not string format)
            {
                return null;
            }

            root.TryGetProperty("params", out JsonElement parameters);
            bool convrot = ReadBool(root, parameters, "convrot");
            return new ComfyQuantDescriptor
            {
                Format = format,
                ConvRotGroupSize = convrot ? ReadInt(root, parameters, "convrot_groupsize", 256) : 0,
                FullPrecisionMatMul = ReadBool(root, parameters, "full_precision_matrix_mult"),
            };
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static bool ReadBool(JsonElement root, JsonElement parameters, string name)
    {
        if (root.TryGetProperty(name, out JsonElement value) && value.ValueKind is JsonValueKind.True or JsonValueKind.False)
            return value.GetBoolean();
        if (parameters.ValueKind == JsonValueKind.Object && parameters.TryGetProperty(name, out JsonElement nested)
            && nested.ValueKind is JsonValueKind.True or JsonValueKind.False)
        {
            return nested.GetBoolean();
        }
        return false;
    }

    private static int ReadInt(JsonElement root, JsonElement parameters, string name, int fallback)
    {
        if (root.TryGetProperty(name, out JsonElement value) && value.TryGetInt32(out int parsed))
            return parsed;
        if (parameters.ValueKind == JsonValueKind.Object && parameters.TryGetProperty(name, out JsonElement nested)
            && nested.TryGetInt32(out int nestedParsed))
        {
            return nestedParsed;
        }
        return fallback;
    }
}
