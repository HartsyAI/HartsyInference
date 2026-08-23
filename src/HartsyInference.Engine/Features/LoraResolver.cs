using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Turns the request's <see cref="LoraStack"/> into concrete file paths + per-component strengths that <see cref="LoraApplier"/> can merge. Mirrors ComfyUI's <c>LoadLorasForConfinement</c> resolution order (try the <c>.safetensors</c> suffix first, then the raw name) so a given selection lands on the same file across backends. Section-confined LoRAs are skipped with a warning — per-segment LoRA scopes are not modeled yet.</summary>
public static class LoraResolver
{
    /// <summary>One resolved LoRA: the on-disk file plus its diffusion-side and text-encoder-side strengths.</summary>
    public sealed class LoraSpec
    {
        /// <summary>Model id or path as requested, kept for logging; may differ from <see cref="FilePath"/>.</summary>
        public string? ModelId { get; init; }

        /// <summary>Resolved on-disk path of the LoRA weights.</summary>
        public required string FilePath { get; init; }

        /// <summary>Strength applied to UNet/Transformer/CLIP-G targets.</summary>
        public required float ModelStrength { get; init; }

        /// <summary>Strength applied to the CLIP-L text-encoder side; equals <see cref="ModelStrength"/> when unspecified.</summary>
        public required float TencStrength { get; init; }
    }

    /// <summary>Models-root-relative folders searched for LoRA weights.</summary>
    private static readonly string[] _folders = ["Lora", "loras", "LoRA", "lora"];

    /// <summary>Resolves every entry in <paramref name="loras"/>. Returns an empty list when the stack is null/empty or every entry is confined to a non-global section. Throws when a selected LoRA has no resolvable file.</summary>
    public static List<LoraSpec> Resolve(LoraStack? loras)
    {
        List<LoraSpec> result = [];
        if (loras is null || loras.Entries is null || loras.Entries.Count == 0)
        {
            return result;
        }
        foreach (LoraEntry entry in loras.Entries)
        {
            // Skip non-global confinement: per-segment LoRA scopes aren't modeled, so honoring the global slot only is safe.
            if (!string.IsNullOrWhiteSpace(entry.SectionConfinement)
                && int.TryParse(entry.SectionConfinement, out int confinementId)
                && confinementId > 0)
            {
                Logs.Debug($"[Features][LoRA] '{entry.Model}' has section confinement={confinementId}; skipping (not yet supported).");
                continue;
            }
            string path = ModelFileLocator.Find(entry.Model, _folders)
                ?? throw new InvalidOperationException($"LoRA model '{entry.Model}' was not found under '{RepoPaths.ModelsRoot()}'.");
            float modelStrength = (float)entry.Weight;
            float tencStrength = entry.TextEncoderWeight is null ? modelStrength : (float)entry.TextEncoderWeight.Value;
            result.Add(new LoraSpec
            {
                ModelId = entry.Model,
                FilePath = path,
                ModelStrength = modelStrength,
                TencStrength = tencStrength,
            });
        }
        return result;
    }
}
