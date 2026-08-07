using System.Globalization;
using System.Text;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes;

/// <summary>Builds the part of a constructed-pipeline cache key that depends on the request rather than the checkpoint:
/// LoRA stacks and swappable-component overrides are merged into the loaded weights, so two requests that differ in
/// either must not share a cached pipeline.</summary>
public static class RecipeCacheKey
{
    /// <summary>A stable, order-sensitive description of the construction-affecting parts of <paramref name="request"/>;
    /// the empty string for a null request or one that changes nothing about construction.</summary>
    public static string Describe(ImageRequest? request) =>
        request is null ? "" : Describe(request.Loras, request.Components);

    /// <summary>The video counterpart — video recipes bake LoRAs into the loaded weights exactly as image ones do, so
    /// their cached pipelines must be keyed by the stack too. The Wan 2.2 expert pair is construction-affecting the
    /// same way: without the swap suffix a swap request would silently reuse a cached single-expert pipeline.</summary>
    public static string Describe(VideoRequest? request)
    {
        if (request is null)
        {
            return "";
        }
        string key = Describe(request.Loras, request.Components);
        if (!string.IsNullOrWhiteSpace(request.VideoSwapModel))
        {
            key += $"swap:{request.VideoSwapModel}@{(request.VideoSwapPercent?.ToString("R", CultureInfo.InvariantCulture) ?? "auto")};";
        }
        return key;
    }

    private static string Describe(LoraStack? loras, ComponentOverrides? components)
    {
        StringBuilder builder = new StringBuilder();
        if (loras is { Entries.Count: > 0 })
        {
            foreach (LoraEntry entry in loras.Entries)
            {
                builder.Append("lora:").Append(entry.Model).Append('@')
                    .Append(entry.Weight.ToString("R", CultureInfo.InvariantCulture)).Append('/')
                    .Append((entry.TextEncoderWeight ?? entry.Weight).ToString("R", CultureInfo.InvariantCulture)).Append('/')
                    .Append(entry.SectionConfinement ?? "-").Append(';');
            }
        }
        if (components is not null)
        {
            builder.Append("cmp:")
                .Append(components.Vae).Append('|').Append(components.T5Xxl).Append('|')
                .Append(components.ClipL).Append('|').Append(components.ClipG).Append('|')
                .Append(components.ClipVision).Append('|').Append(components.Qwen).Append('|')
                .Append(components.Llama).Append('|').Append(components.Gemma).Append(';');
        }
        return builder.ToString();
    }
}
