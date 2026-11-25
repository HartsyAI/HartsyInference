using System.Text.Json;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>ComfyUI's per-layer attention hints in the Fibo 2.0 Comfy-Org builds (<c>layers.N.attention.comfy_attention.config</c>):
/// a 35-byte U8 JSON blob, <c>{"attention": "comfy_kitchen_int8"}</c>, on every block. It names the attention backend ComfyUI runs
/// those layers with and is not a weight. The engine has no int8 attention, so it runs float attention; the hint is accepted, logged
/// as a deviation, and dropped from the weights. A hint naming any other backend is refused, since the engine cannot run it.</summary>
internal static class Fibo2AttentionHints
{
    /// <summary>The key suffix every hint carries.</summary>
    public const string Suffix = ".comfy_attention.config";

    /// <summary>ComfyUI's int8 attention backend, the only one the Fibo 2.0 builds name.</summary>
    public const string ComfyKitchenInt8 = "comfy_kitchen_int8";

    /// <summary>Whether a checkpoint key is an attention hint rather than a tensor.</summary>
    public static bool IsHint(string key) => key.EndsWith(Suffix, StringComparison.Ordinal);

    /// <summary>Reads every hint in <paramref name="weights"/> and returns the distinct backends they name, after checking each one.</summary>
    /// <exception cref="InvalidDataException">A hint is not a U8 blob holding <c>{"attention": "&lt;name&gt;"}</c>.</exception>
    /// <exception cref="NotSupportedException">A hint names a backend the engine cannot run.</exception>
    public static IReadOnlyList<string> Requested(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        List<string> names = [];
        foreach (KeyValuePair<string, Tensor> pair in weights)
        {
            if (!IsHint(pair.Key))
                continue;
            string name = Read(pair.Value, pair.Key);
            if (name != ComfyKitchenInt8)
            {
                throw new NotSupportedException($"Fibo 2.0 tensor '{pair.Key}' names attention backend '{name}'. The engine runs only "
                    + $"'{ComfyKitchenInt8}' hints (as float attention); it cannot run '{name}'.");
            }
            if (!names.Contains(name))
                names.Add(name);
        }
        return names;
    }

    /// <summary>Decodes one hint blob to its backend name. Trailing NUL padding is tolerated, as in the other ComfyUI JSON blobs.</summary>
    internal static string Read(Tensor blob, string key)
    {
        ArgumentNullException.ThrowIfNull(blob);
        if (blob.DType != DType.U8 || blob.Shape.Rank != 1 || blob.ElementCount is <= 0 or > 4096)
        {
            throw new InvalidDataException($"Fibo 2.0 tensor '{key}' must be a one-dimensional U8 blob of 1 to 4096 bytes; got {blob.DType} {blob.Shape}.");
        }
        ReadOnlySpan<byte> bytes = blob.AsReadOnlySpan<byte>();
        int end = bytes.Length;
        while (end > 0 && bytes[end - 1] == 0)
            end--;
        try
        {
            using JsonDocument json = JsonDocument.Parse(bytes[..end].ToArray());
            if (json.RootElement.ValueKind == JsonValueKind.Object
                && json.RootElement.TryGetProperty("attention", out JsonElement value)
                && value.ValueKind == JsonValueKind.String
                && value.GetString() is { Length: > 0 } name)
            {
                return name;
            }
        }
        catch (JsonException)
        {
            // Reported below with the key.
        }
        throw new InvalidDataException($"Fibo 2.0 tensor '{key}' is not an attention hint of the form {{\"attention\": \"<backend>\"}}.");
    }
}
