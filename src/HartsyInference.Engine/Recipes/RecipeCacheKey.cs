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
    public static string Describe(ImageRequest? request)
    {
        if (request is null)
        {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        if (request.Loras is { Entries.Count: > 0 })
        {
            foreach (LoraEntry entry in request.Loras.Entries)
            {
                builder.Append("lora:").Append(entry.Model).Append('@')
                    .Append(entry.Weight.ToString("R", CultureInfo.InvariantCulture)).Append('/')
                    .Append((entry.TextEncoderWeight ?? entry.Weight).ToString("R", CultureInfo.InvariantCulture)).Append('/')
                    .Append(entry.SectionConfinement ?? "-").Append(';');
            }
        }
        ComponentOverrides? components = request.Components;
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
