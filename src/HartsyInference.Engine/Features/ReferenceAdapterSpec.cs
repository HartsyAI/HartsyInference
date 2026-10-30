using System.Text.Json;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Engine.Features;

/// <summary>A reference adapter as its own metadata describes it: where its reference sits on the RoPE axes, and whether it reads the
/// reference through its LoRA. Only the asymmetric, single-slot layout is supported (the released v2 file); the joint-frame pilot and
/// two-slot files are refused by name.</summary>
public sealed record ReferenceAdapterSpec(string Path, ReferencePlacement Placement, bool ReferenceLora)
{
    /// <summary>The metadata key the adapter stores its JSON under.</summary>
    public const string MetadataKey = "anima_ref_adapter";

    /// <summary>Reads the spec from the adapter's <paramref name="metadataJson"/>; <paramref name="path"/> is named in every refusal.</summary>
    public static ReferenceAdapterSpec FromMetadata(string path, string? metadataJson)
    {
        if (string.IsNullOrWhiteSpace(metadataJson))
            throw new NotSupportedException($"Reference adapter '{System.IO.Path.GetFileName(path)}' has no '{MetadataKey}' metadata, so its layout is unknown.");

        using JsonDocument document = JsonDocument.Parse(metadataJson);
        JsonElement root = document.RootElement;
        string arch = root.TryGetProperty("arch", out JsonElement archElement) ? archElement.GetString() ?? "" : "";
        if (arch != "asym")
        {
            throw new NotSupportedException($"Reference adapter '{System.IO.Path.GetFileName(path)}' has arch '{arch}'. Only the asymmetric "
                + "layout (arch 'asym', the released v2 files) is supported; the joint-frame pilot is not.");
        }
        if (!root.TryGetProperty("slots", out JsonElement slots) || slots.ValueKind != JsonValueKind.Object)
            throw new NotSupportedException($"Reference adapter '{System.IO.Path.GetFileName(path)}' names no reference slots.");
        int slotCount = 0;
        foreach (JsonProperty _ in slots.EnumerateObject())
            slotCount++;
        if (slotCount != 1 || !slots.TryGetProperty("character", out JsonElement character))
        {
            throw new NotSupportedException($"Reference adapter '{System.IO.Path.GetFileName(path)}' has {slotCount} reference slots. Only the single "
                + "'character' slot is supported; the two-slot character/style files are not.");
        }

        int frame = character.TryGetProperty("t", out JsonElement t) ? t.GetInt32() : 1;
        float rowOffset = character.TryGetProperty("offset_h", out JsonElement h) ? h.GetSingle() : 1f;
        int columnOffset = character.TryGetProperty("offset_w", out JsonElement w) ? w.GetInt32() : 0;
        bool referenceLora = !root.TryGetProperty("ref_lora", out JsonElement lora) || lora.GetBoolean();
        return new ReferenceAdapterSpec(path, new ReferencePlacement(frame, rowOffset, columnOffset), referenceLora);
    }
}
