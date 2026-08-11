using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Vision;

namespace HartsyInference.Engine.Features;

/// <summary>Tier 3.2: <c>&lt;segment:X&gt;</c> post-hoc refinement — the mask-driven counterpart to
/// <see cref="InpaintOnlyMasked"/>'s bbox crop, run AFTER a base image already exists as pixels (a
/// <c>&lt;region:&gt;</c>/<c>&lt;object:&gt;</c> plan is a live per-step attention bias applied while the image still
/// denoises; a <c>&lt;segment:&gt;</c> plan needs pixels to segment, so it cannot run before the base image is
/// decoded — confirmed against SwarmUI core's own <c>RunSegmentationProcessing</c> reference implementation before
/// writing this, see <c>ROADMAP.md</c>'s design-pass entry).
///
/// <para>Reuses <see cref="InpaintOnlyMasked"/> verbatim for the crop→generate→composite cycle per segment — that
/// subroutine already does exactly "crop to the mask's bounding box (oversized), generate at the model's native
/// resolution over just the crop, composite back" for ordinary "inpaint only masked" requests. The only genuinely
/// new piece is producing the mask (CLIPSeg text-match on the CURRENT image) and building the per-segment synthetic
/// <see cref="ImageRequest"/> that mask feeds into. Segments are applied sequentially, each seeing the previous
/// segment's composited result, so overlapping segments compose left-to-right in prompt order.</para>
///
/// <para><b>Known limitation, not fixed here</b>: the BASE (full-canvas) generation still tokenizes the raw prompt
/// including every <c>&lt;segment:X&gt;...&lt;segment:end&gt;</c> substring verbatim — the same pre-existing behavior
/// <c>&lt;region:&gt;</c>/<c>&lt;object:&gt;</c> tags already have (no recipe pipeline strips region/segment tag text
/// from what it sends to its own text encoder; each only reads <see cref="Regional"/> field
/// data or thread its own <c>RegionalPlan</c> attention bias on top). A prompt with a large segment sub-prompt will
/// lightly influence the base image too, not just the masked region. Fixing that is a separate, smaller item (make
/// every regional-capable recipe pipeline tokenize <see cref="Engine.Features.PromptRegionParser.GlobalPrompt"/>
/// instead of the raw prompt) — out of scope for landing the segment mechanism itself.</para>
///
/// <para>Covers CLIPSeg free-text matching only (<c>X</c> not starting with <c>yolo-</c>) — the YOLO closed-vocab
/// detection path (<see cref="Vision.Detection"/>-shaped output, needs its own box→mask rasterization before this
/// same crop/composite cycle applies) is not wired. Single-mask-per-segment only; the pipe-separated
/// <c>X|Y</c> OR-composite syntax is not parsed. <c>&lt;clear:&gt;</c> (a separate alpha-cutout mechanism, no
/// denoise) is not wired either — see the class doc above for why it's a distinct feature.</para></summary>
public static class SegmentRefinement
{
    /// <summary>Oversize padding used when <see cref="Regional.MaskOversize"/> is unset (0) —
    /// <see cref="InpaintOnlyMasked.Prepare"/> requires a nonzero <c>ShrinkGrow</c> to trigger its crop path at all,
    /// and 0 padding would crop exactly to the mask's raw bounds with no margin for the blurred edge to blend into.
    /// Matches SwarmUI core's own <c>SegmentMaskOversize</c> default.</summary>
    private const int DefaultMaskOversize = 16;

    /// <summary>True when <paramref name="prompt"/> carries at least one <c>&lt;segment:&gt;</c> part — cheap
    /// pre-check so a request with no segments never pays for a <see cref="PromptRegionParser"/> parse plus a
    /// CLIPSeg weight-directory lookup.</summary>
    public static bool HasSegmentParts(string? prompt)
    {
        if (string.IsNullOrEmpty(prompt) || !prompt.Contains("<segment:", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }
        PromptRegionParser parsed = new PromptRegionParser(prompt);
        foreach (PromptRegionParser.Part part in parsed.Parts)
        {
            if (part.Type == PromptRegionParser.PartType.Segment)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Runs every <c>&lt;segment:X&gt;</c> part in <paramref name="resolved"/>'s prompt against
    /// <paramref name="generated"/> (the already-composited base result), returning the final image with every
    /// matched segment refined in place. Segments whose CLIPSeg match is empty (nothing scored above the part's
    /// <see cref="PromptRegionParser.Part.Strength"/> threshold) or whose grown/oversized mask covers the whole
    /// canvas are logged and skipped, not treated as an error — the same tolerance
    /// <see cref="InpaintOnlyMasked.Prepare"/> already has for a mask that selects nothing.</summary>
    public static ImageResult Apply(
        ImageResult generated, ImageRequest resolved, IRecipePipeline pipeline, IBackend backend,
        ClipSegSegmenter segmenter, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(generated);
        ArgumentNullException.ThrowIfNull(resolved);
        ArgumentNullException.ThrowIfNull(pipeline);
        if (!HasSegmentParts(resolved.Prompt))
        {
            return generated;
        }
        PromptRegionParser parsed = new PromptRegionParser(resolved.Prompt);
        List<PromptRegionParser.Part> segments = [];
        foreach (PromptRegionParser.Part part in parsed.Parts)
        {
            if (part.Type == PromptRegionParser.PartType.Segment)
            {
                segments.Add(part);
            }
        }
        if (segments.Count == 0)
        {
            return generated;
        }
        string? clipSegDir = VisionModelPaths.FindClipSegDirectory(null);
        if (clipSegDir is null)
        {
            Logs.Warning("[Features][SegmentRefinement] CLIPSeg model not found — segment refinement skipped, "
                + "base image returned unmodified. Place the 'clipseg-rd64-refined' folder under Models/clipseg/.");
            return generated;
        }
        Regional regional = resolved.Regional
            ?? new Regional { Plan = resolved.Prompt };
        int oversize = regional.MaskOversize == 0 ? DefaultMaskOversize : regional.MaskOversize;

        ImageResult current = generated;
        int refined = 0;
        foreach (PromptRegionParser.Part seg in segments)
        {
            cancel.ThrowIfCancellationRequested();
            if (string.IsNullOrWhiteSpace(seg.DataText))
            {
                Logs.Warning("[Features][SegmentRefinement] <segment:> part has no matcher text — skipped.");
                continue;
            }
            if (seg.DataText.StartsWith("yolo-", StringComparison.OrdinalIgnoreCase))
            {
                Logs.Warning($"[Features][SegmentRefinement] YOLO-detector segments ('{seg.DataText}') are not wired yet — skipped.");
                continue;
            }
            ImageData currentImage = new ImageData { Rgb = current.Rgb, Width = current.Width, Height = current.Height };
            float threshold = (float)Math.Clamp(seg.Strength <= 0 ? 0.4 : seg.Strength, 0.01, 0.99);
            byte[]? mask = segmenter.Segment(backend, clipSegDir, currentImage, seg.DataText, threshold);
            if (mask is null)
            {
                // ClipSegSegmenter already logs the threshold miss — nothing more to say here.
                continue;
            }
            ImageData maskImage = VisionMasks.ToImageData(mask, current.Width, current.Height);

            ImageRequest segRequest = resolved with
            {
                Prompt = seg.Prompt,
                Regional = null,
                Img2Img = new Img2Img { InitImage = currentImage, Creativity = seg.Strength2, Mode = Img2ImgMode.Denoise },
                Inpaint = new Inpaint { Mask = maskImage, Grow = regional.MaskGrow, Blur = regional.MaskBlur, ShrinkGrow = oversize },
                Steps = regional.Steps ?? resolved.Steps,
                CfgScale = regional.CfgScale.HasValue ? (float)regional.CfgScale.Value : resolved.CfgScale,
            };
            InpaintOnlyMasked.Plan? cropPlan = InpaintOnlyMasked.Prepare(segRequest);
            if (cropPlan is null)
            {
                Logs.Info($"[Features][SegmentRefinement] Segment '{seg.DataText}' mask selects nothing (or the whole canvas) after grow/oversize — skipped.");
                continue;
            }
            ImageRequest cropped = InpaintOnlyMasked.Apply(segRequest, cropPlan);
            ImageResult segGenerated = pipeline.Generate(cropped, progress, cancel);
            current = InpaintOnlyMasked.Composite(segGenerated, cropPlan);
            refined++;
        }
        if (refined > 0)
        {
            current = current with
            {
                Meta = new Dictionary<string, string>(current.Meta, StringComparer.OrdinalIgnoreCase)
                {
                    ["segments_refined"] = refined.ToString(System.Globalization.CultureInfo.InvariantCulture),
                },
            };
        }
        return current;
    }
}
