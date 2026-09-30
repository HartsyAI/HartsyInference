using System.Text.Json;
using HartsyInference.ModelAssets.Quant;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>Maps a checkpoint's <c>quantization_config</c> to the producer naming its companion tensors follow.</summary>
public static class HfQuantFlavorDetector
{
    /// <summary>Returns the flavor <paramref name="config"/> declares, or null when it declares none this engine binds.</summary>
    /// <remarks>NVIDIA's ModelOpt export also says <c>fp8</c> (its non-expert weights are official), so the producer decides; MLX omits <c>quant_method</c> and says <c>mode: affine</c>.</remarks>
    public static QuantFlavor? FromConfig(JsonElement config)
    {
        if (config.ValueKind != JsonValueKind.Object)
            return null;
        if (!TryObject(config, "quantization_config", out JsonElement quant)
            && !TryObject(config, "quantization", out quant))
            return null;

        string? method = String(quant, "quant_method")?.ToLowerInvariant();
        switch (method)
        {
            case "fp8":
                return IsNvidiaModelOpt(quant) ? QuantFlavor.NvidiaNvfp4 : QuantFlavor.Official;
            case "modelopt":
                return QuantFlavor.NvidiaNvfp4;
            case "quark":
                return QuantFlavor.AmdQuark;
            case "exl3":
                return QuantFlavor.Exl3;
            case null when string.Equals(String(quant, "mode"), "affine", StringComparison.OrdinalIgnoreCase):
                return QuantFlavor.Mlx;
            default:
                return null;
        }
    }

    private static bool IsNvidiaModelOpt(JsonElement quant)
    {
        if (string.Equals(String(quant, "moe_quant_algo"), "NVFP4", StringComparison.OrdinalIgnoreCase))
            return true;
        return TryObject(quant, "producer", out JsonElement producer)
            && string.Equals(String(producer, "name"), "modelopt", StringComparison.OrdinalIgnoreCase);
    }

    private static bool TryObject(JsonElement parent, string name, out JsonElement value)
    {
        if (parent.TryGetProperty(name, out value) && value.ValueKind == JsonValueKind.Object)
            return true;
        value = default;
        return false;
    }

    private static string? String(JsonElement parent, string name) =>
        parent.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
}
