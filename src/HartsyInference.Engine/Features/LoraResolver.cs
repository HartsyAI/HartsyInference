using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Turns the request's <see cref="LoraStack"/> into concrete file paths + per-component strengths that <see cref="RecipeLoraMerge"/> can apply. Mirrors ComfyUI's <c>LoadLorasForConfinement</c> resolution order (try the <c>.safetensors</c> suffix first, then the raw name) so a given selection lands on the same file across backends. Zero-strength entries are dropped, and section-confined LoRAs are refused by name, since per-segment LoRA scopes are not modeled yet.</summary>
public static class LoraResolver
{
    /// <summary>One resolved LoRA: the on-disk file plus its diffusion-side and text-encoder-side strengths.</summary>
    public sealed class LoraSpec
    {
        /// <summary>Model id or path as requested, kept for logging; may differ from <see cref="FilePath"/>.</summary>
        public string? ModelId { get; init; }

        /// <summary>Resolved on-disk path of the LoRA weights.</summary>
        public required string FilePath { get; init; }

        /// <summary>Strength applied to the diffusion body — the UNet or transformer.</summary>
        public required float ModelStrength { get; init; }

        /// <summary>Strength applied to EVERY text-encoder arm (CLIP-L, CLIP-G, T5/LLM), matching SwarmUI's <c>strength_clip</c>; equals <see cref="ModelStrength"/> when unspecified.</summary>
        public required float TencStrength { get; init; }
    }

    /// <summary>Models-root-relative folders searched for LoRA weights.</summary>
    private static readonly string[] _folders = ["Lora", "loras", "LoRA", "lora"];

    /// <summary>Whether <paramref name="entry"/> changes nothing: zero on the diffusion side and on the text-encoder side. Resolution
    /// drops such an entry and the cache key omits it, so zeroing a LoRA lands on the base pipeline instead of building a second copy.</summary>
    public static bool IsNoOp(LoraEntry entry) => entry.Weight == 0 && (entry.TextEncoderWeight ?? entry.Weight) == 0;

    /// <summary>Resolves every entry in <paramref name="loras"/>, dropping the no-op ones. Throws when an entry is confined to a section
    /// or when a selected LoRA has no resolvable file.</summary>
    public static List<LoraSpec> Resolve(LoraStack? loras)
    {
        List<LoraSpec> result = [];
        if (loras is null || loras.Entries is null || loras.Entries.Count == 0)
        {
            return result;
        }
        foreach (LoraEntry entry in loras.Entries)
        {
            // NaN and infinity pass every later check (they compare unequal to zero), and would merge straight into the weights.
            if (!double.IsFinite(entry.Weight) || (entry.TextEncoderWeight is { } encoderWeight && !double.IsFinite(encoderWeight)))
            {
                throw new NotSupportedException($"LoRA '{entry.Model}' has a strength that is not a finite number. Set a finite weight.");
            }
            if (IsNoOp(entry))
            {
                Logs.Info($"[Features][LoRA] '{entry.Model}' has zero strength on both sides; not loaded.");
                continue;
            }
            // Section confinement names a segment scope this engine does not model. Skipping it would generate an image that
            // silently lacks the LoRA the request asked for, so the request is refused and the LoRA is named.
            if (!string.IsNullOrWhiteSpace(entry.SectionConfinement)
                && int.TryParse(entry.SectionConfinement, out int confinementId) && confinementId > 0)
            {
                throw new NotSupportedException(
                    $"LoRA '{entry.Model}' is confined to section {confinementId}. Per-section LoRA scopes are not modeled yet, "
                    + "so this LoRA cannot be applied. Remove the section confinement to apply it to the whole image.");
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
