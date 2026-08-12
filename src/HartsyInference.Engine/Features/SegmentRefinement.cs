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
/// <para><b>Base-prompt tag-leak — fixed for segment/clear text (2026-08-11), see <see cref="StripSegmentText"/>.</b>
/// <c>&lt;region:&gt;</c>/<c>&lt;object:&gt;</c> still have the same class of leak (no recipe pipeline tokenizes
/// <see cref="Engine.Features.PromptRegionParser.GlobalPrompt"/> for its base pass, each re-parses
/// <c>request.Prompt</c> raw via <c>RegionalPromptResolver.HasRegionParts</c>) — that is a DIFFERENT fix, not
/// covered here: those pipelines need the raw prompt string (region tags intact) to re-parse themselves, so
/// stripping at <see cref="Services.ImagesService"/> the way this class does for segments would silently break
/// five already-verified architectures (1.4's Flux.1/Z-Image/Ideogram4, 3.7's Flux.2/Krea2). Left as a separate,
/// documented gap in <c>ROADMAP.md</c>.</para>
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

    /// <summary>Tag prefixes that <see cref="PromptRegionParser.Parse"/> treats as opening a recognized section
    /// other than segment/clear — mirrored here (not shared via the parser) so this stays a narrow, low-risk
    /// addition rather than a change to the shared parser class the five already-verified regional architectures
    /// depend on.</summary>
    private static readonly string[] _otherRecognizedPrefixes =
        ["region", "object", "extend", "base", "refiner", "pixeldecoder", "video", "videoswap"];

    /// <summary>Returns <paramref name="prompt"/> with every <c>&lt;segment:X&gt;</c>/<c>&lt;clear:X&gt;</c> tag AND
    /// the text that accumulates into it (up to whichever tag reopens a different recognized section, or the end
    /// of the prompt) removed — everything else, including <c>&lt;region:&gt;</c>/<c>&lt;object:&gt;</c> tags and
    /// their own content, is preserved byte-for-byte so a pipeline that re-parses the result on its own (e.g.
    /// <c>RegionalPromptResolver.HasRegionParts</c>) sees exactly what it would have seen from the untouched
    /// prompt. Mirrors <see cref="PromptRegionParser.Parse"/>'s own split/accumulate loop rather than a regex,
    /// because "what belongs to a segment" is defined by that accumulator rebinding, not by the tag's own span —
    /// text after <c>&lt;segment:X&gt;</c> up to the next tag is the segment's sub-prompt, not the base prompt.
    /// Case-sensitive prefix match, same as the parser itself (a stray-cased <c>&lt;Segment:&gt;</c> is untouched
    /// here for the same reason <see cref="PromptRegionParser"/> would fall through and treat it as ordinary
    /// prompt text, not a tag).</summary>
    public static string StripSegmentText(string? prompt)
    {
        string text = prompt ?? "";
        if (!text.Contains('<', StringComparison.Ordinal))
        {
            return text;
        }
        string[] pieces = text.Split('<');
        System.Text.StringBuilder result = new();
        bool skip = false; // true while accumulating inside a segment/clear section
        bool first = true;
        foreach (string piece in pieces)
        {
            if (first)
            {
                first = false;
                result.Append(piece);
                continue;
            }
            int end = piece.IndexOf('>', StringComparison.Ordinal);
            if (end == -1)
            {
                // Unterminated "<...": the parser appends it to whichever section is currently accumulating.
                if (!skip)
                {
                    result.Append('<').Append(piece);
                }
                continue;
            }
            string tag = piece[..end];
            int cidAt = tag.LastIndexOf("//cid=", StringComparison.Ordinal);
            if (cidAt >= 0)
            {
                tag = tag[..cidAt];
            }
            int colon = tag.IndexOf(':', StringComparison.Ordinal);
            string prefix = colon < 0 ? tag : tag[..colon];
            if (prefix is "segment" or "clear")
            {
                skip = true; // drop the tag itself and everything until the next recognized section
                continue;
            }
            if (Array.IndexOf(_otherRecognizedPrefixes, prefix) >= 0)
            {
                skip = false;
                result.Append('<').Append(piece); // preserve verbatim, including any //cid= suffix
                continue;
            }
            // Unrecognized tag (weighting, <break>, <embed:...>): belongs to whichever section is active.
            if (!skip)
            {
                result.Append('<').Append(piece);
            }
        }
        // A trailing space left where a segment tag was cut out would tokenize differently from a prompt that
        // never had one — trim so a segment-free vs. stripped-segment prompt encode identically.
        return result.ToString().TrimEnd();
    }

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
