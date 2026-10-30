using System.Text.Json;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Engine.Features;

/// <summary>A reference adapter as its own metadata describes it: where its reference sits on the RoPE axes. Only the asymmetric,
/// single-slot layout that reads the reference through its LoRA is supported (the released v2 file); the joint-frame pilot, the
/// two-slot files and the non-LoRA variant are refused by name.</summary>
public sealed record ReferenceAdapterSpec(string Path, ReferencePlacement Placement)
{
    /// <summary>The metadata key the adapter stores its JSON under.</summary>
    public const string MetadataKey = "anima_ref_adapter";

    /// <summary>Reads the spec from the adapter's <paramref name="metadataJson"/>; <paramref name="path"/> is named in every refusal.</summary>
    public static ReferenceAdapterSpec FromMetadata(string path, string? metadataJson)
    {
        string name = System.IO.Path.GetFileName(path);
        if (string.IsNullOrWhiteSpace(metadataJson))
            throw new NotSupportedException($"Reference adapter '{name}' has no '{MetadataKey}' metadata, so its layout is unknown.");
        try
        {
            return Parse(path, metadataJson);
        }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException or FormatException or ArgumentOutOfRangeException)
        {
            throw new NotSupportedException($"Reference adapter '{name}' has malformed '{MetadataKey}' metadata: {ex.Message}", ex);
        }
    }

    private static ReferenceAdapterSpec Parse(string path, string metadataJson)
    {
        string name = System.IO.Path.GetFileName(path);
        using JsonDocument document = JsonDocument.Parse(metadataJson);
        JsonElement root = document.RootElement;
        string arch = root.TryGetProperty("arch", out JsonElement archElement) ? archElement.GetString() ?? "" : "";
        if (arch != "asym")
        {
            throw new NotSupportedException($"Reference adapter '{name}' has arch '{arch}'. Only the asymmetric "
                + "layout (arch 'asym', the released v2 files) is supported; the joint-frame pilot is not.");
        }
        if (root.TryGetProperty("ref_lora", out JsonElement lora) && !lora.GetBoolean())
        {
            throw new NotSupportedException($"Reference adapter '{name}' has ref_lora false. Only an adapter that reads the reference "
                + "through its LoRA (the released v2 files) is supported.");
        }
        if (!root.TryGetProperty("slots", out JsonElement slots) || slots.ValueKind != JsonValueKind.Object)
            throw new NotSupportedException($"Reference adapter '{name}' names no reference slots.");
        int slotCount = 0;
        foreach (JsonProperty _ in slots.EnumerateObject())
            slotCount++;
        if (slotCount != 1 || !slots.TryGetProperty("character", out JsonElement character))
        {
            throw new NotSupportedException($"Reference adapter '{name}' has {slotCount} reference slots. Only the single "
                + "'character' slot is supported; the two-slot character/style files are not.");
        }

        int frame = character.TryGetProperty("t", out JsonElement t) ? t.GetInt32() : 1;
        float rowOffset = character.TryGetProperty("offset_h", out JsonElement h) ? h.GetSingle() : 1f;
        int columnOffset = character.TryGetProperty("offset_w", out JsonElement w) ? w.GetInt32() : 0;
        return new ReferenceAdapterSpec(path, new ReferencePlacement(frame, rowOffset, columnOffset));
    }
}
