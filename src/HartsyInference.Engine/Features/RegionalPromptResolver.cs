using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Engine.Features;

/// <summary>Translates <c>&lt;region:x,y,w,h,strength&gt;</c> / <c>&lt;object:…&gt;</c> prompt parts into a
/// <see cref="RegionalPlan"/>: each part's fractional bbox becomes a pixel-space <see cref="RectMask"/> and its prompt is
/// encoded via the supplied delegate, which the engine turns into a per-step attention bias. Returns null when the prompt
/// carries no region/object parts, so callers keep the plain single-conditioning path byte-identical.</summary>
public static class RegionalPromptResolver
{
    /// <summary>Builds the plan from the RAW (un-stripped) prompt, or null when it has no region/object parts.
    /// <paramref name="encodeRegion"/> encodes one region's text into a <c>[1, L, featDim]</c> caption embedding using the
    /// same encoder path as the global prompt; the caller owns disposal via <see cref="DisposeRegions"/>.</summary>
    public static RegionalPlan? Resolve(
        string? rawPrompt, Tensor baseCond, int width, int height, int steps,
        Func<string, Tensor> encodeRegion)
    {
        ArgumentNullException.ThrowIfNull(encodeRegion);
        if (string.IsNullOrEmpty(rawPrompt) || !rawPrompt.Contains('<', StringComparison.Ordinal))
        {
            return null;
        }
        PromptRegionParser parsed = new PromptRegionParser(rawPrompt);
        List<RegionConditioning> regions = [];
        foreach (PromptRegionParser.Part part in parsed.Parts)
        {
            if (part.Type is not (PromptRegionParser.PartType.Region or PromptRegionParser.PartType.Object))
            {
                continue;
            }
            // Fractions → pixel rect, clamped inside the canvas (rounding can overshoot by a pixel).
            int rx = Math.Clamp((int)MathF.Round(part.X * width), 0, Math.Max(0, width - 1));
            int ry = Math.Clamp((int)MathF.Round(part.Y * height), 0, Math.Max(0, height - 1));
            int rw = Math.Clamp((int)MathF.Round(part.Width * width), 1, width - rx);
            int rh = Math.Clamp((int)MathF.Round(part.Height * height), 1, height - ry);
            RegionMask mask = RegionMask.FromRect(new RectMask(rx, ry, rw, rh), width, height);
            Tensor cond = encodeRegion(part.Prompt);
            regions.Add(new RegionConditioning(cond, mask, (float)part.Strength, 0, steps));
        }
        if (regions.Count == 0)
        {
            return null;
        }
        return new RegionalPlan { BaseCond = baseCond, Regions = regions };
    }

    /// <summary>Cheap pre-check for region/object parts, so callers can decide text-encoder weight residency before encoding.</summary>
    public static bool HasRegionParts(string? rawPrompt)
    {
        if (string.IsNullOrEmpty(rawPrompt) || !rawPrompt.Contains('<', StringComparison.Ordinal))
        {
            return false;
        }
        PromptRegionParser parsed = new PromptRegionParser(rawPrompt);
        foreach (PromptRegionParser.Part part in parsed.Parts)
        {
            if (part.Type is PromptRegionParser.PartType.Region or PromptRegionParser.PartType.Object)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Disposes the per-region conditioning tensors owned by a plan. Null-safe.</summary>
    public static void DisposeRegions(RegionalPlan? plan)
    {
        if (plan is null)
        {
            return;
        }
        foreach (RegionConditioning region in plan.Regions)
        {
            region.Cond?.Dispose();
        }
    }
}
