using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Vision;

namespace HartsyInference.Engine.Features;

/// <summary>Tier 3.8: <c>&lt;refcrop:N,query&gt;</c> / <c>&lt;refcrop:N,query,threshold&gt;</c> — auto-crops a MiniMax-H3 reference IMAGE to a CLIPSeg-matched region before it reaches <c>MiniMaxH3RecipePipeline.EncodeReferenceImage</c>, instead of the whole attached asset. This is a convenience, not a capability unlock — a user can already crop the reference themselves before uploading it; the value is "point at what you mean in text" instead of opening an image editor. <para><b>Syntax design, matched to SwarmUI's own conventions</b> (researched against the real SwarmUI checkout before inventing anything — see <c>docs/Research/MINIMAX_H3.md</c>'s design-pass section): the argument grammar mirrors <c>&lt;segment:query,creativity,threshold&gt;</c>'s own right-to-left optional-trailing-float parsing (last comma-field = threshold if it parses as a float, else default), and the picture index is 1-based to match H3's own established <c>&lt;Picture N&gt;</c> convention (the same reference the model's OWN presentation layer uses — <c>MiniMaxH3TextEncoding</c>'s <c>"&lt;Picture i&gt;"</c> labels, 1-based). No index-omission default exists (unlike <c>&lt;segment:&gt;</c>'s all-optional-trailing grammar) — there's no sensible "which reference" default, so the index is a required LEADING argument instead of a trailing one.</para> <para><b>Not registered with SwarmUI core</b> (<c>PromptRegion.RegisterCustomPrefix</c>) — confirmed by reading core directly that an unregistered <c>&lt;...&gt;</c> tag passes through both of core's parsing layers byte-for-byte unchanged (exactly how <c>&lt;Picture N&gt;</c> itself already survives to the engine today), so registration buys nothing this feature needs (no LoRA section-confinement/<c>ContextID</c> interplay applies to a reference crop). This class does its own self-contained parse of the raw prompt text, entirely engine-side.</para> <para><b>Must strip before the prompt reaches the text encoder</b> — learned directly from the Tier 3.2 base-prompt tag-leak bug (fixed 2026-08-11, see <see cref="SegmentRefinement"/>'s own history): an un-stripped structural tag left in <c>request.Prompt</c> reaches the tokenizer as literal text and steers the WHOLE generation, not just the targeted crop. <see cref="Apply"/> always returns a request with the tags stripped, even when no image ends up cropped (e.g. CLIPSeg found nothing) — the tag itself must never leak either way.</para> <para>Scope, matching Tier 3.2's own precedent: reference IMAGES only (video/audio references need per-frame or per-clip mask consistency — a materially bigger problem, explicitly out of scope), CLIPSeg free-text matching only (no YOLO closed-vocab detection), an empty match warns and falls back to the whole uncropped reference (tolerance, not an error) — the same discipline <see cref="SegmentRefinement"/> already established.</para></summary>
public static class ReferenceCropResolver
{
    /// <summary>Default mask threshold when the tag omits one — matches <c>&lt;segment:&gt;</c>'s own default.</summary>
    private const float DefaultThreshold = 0.5f;

    /// <summary>Oversize padding around the matched region so the crop isn't razor-tight to the mask (which could clip meaningful context — e.g. a face crop losing hair at the exact segmentation boundary). Matches <see cref="SegmentRefinement"/>'s own default oversize.</summary>
    private const int CropOversize = 16;

    /// <summary>Mask values below this are treated as unselected when finding the bounding box — mirrors <see cref="InpaintOnlyMasked"/>'s own <c>BoundsThreshold</c> (SwarmUI applies the same 0.01-equivalent cutoff before its own mask-bounds step).</summary>
    private const byte BoundsThreshold = 3;

    /// <summary>One parsed <c>&lt;refcrop:&gt;</c> tag.</summary>
    private sealed record CropTag(int PictureIndex, string Query, float Threshold);

    /// <summary>True when <paramref name="prompt"/> carries at least one <c>&lt;refcrop:&gt;</c> tag — cheap pre-check so a request with no crop tags never pays for a CLIPSeg weight-directory lookup.</summary>
    public static bool HasCropTags(string? prompt) =>
        !string.IsNullOrEmpty(prompt) && prompt.Contains("<refcrop:", StringComparison.Ordinal);

    /// <summary>Runs every <c>&lt;refcrop:N,query[,threshold]&gt;</c> tag in <paramref name="request"/>'s prompt against the referenced image, returning a request with the matched images cropped in place and the tags removed from the prompt. Always strips the tags even when nothing ends up cropped. Returns <paramref name="request"/> unchanged when it carries no crop tags at all (identity path).</summary>
    public static VideoRequest Apply(VideoRequest request, IBackend backend, ClipSegSegmenter segmenter, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (!HasCropTags(request.Prompt))
        {
            return request;
        }
        (string strippedPrompt, List<CropTag> tags) = Extract(request.Prompt);
        if (tags.Count == 0)
        {
            return request with { Prompt = strippedPrompt };
        }
        IReadOnlyList<ImageData> images = request.ReferenceImages ?? [];
        string? clipSegDir = VisionModelPaths.FindClipSegDirectory(null);
        if (clipSegDir is null)
        {
            Logs.Warning("[Features][ReferenceCropResolver] CLIPSeg model not found — <refcrop:> tags ignored, "
                + "references used uncropped. Place the 'clipseg-rd64-refined' folder under Models/clipseg/.");
            return request with { Prompt = strippedPrompt };
        }

        List<ImageData> cropped = [.. images];
        foreach (CropTag tag in tags)
        {
            cancel.ThrowIfCancellationRequested();
            int index = tag.PictureIndex - 1;
            if (index < 0 || index >= cropped.Count)
            {
                Logs.Warning($"[Features][ReferenceCropResolver] <refcrop:{tag.PictureIndex},{tag.Query}> "
                    + $"references Picture {tag.PictureIndex}, but only {cropped.Count} reference image(s) are "
                    + "attached — ignored.");
                continue;
            }
            ImageData source = cropped[index];
            byte[]? mask = segmenter.Segment(backend, clipSegDir, source, tag.Query, tag.Threshold);
            if (mask is null)
            {
                // ClipSegSegmenter already logs the threshold miss — nothing more to say here.
                Logs.Warning($"[Features][ReferenceCropResolver] Reference {tag.PictureIndex} left uncropped "
                    + $"('{tag.Query}' matched nothing at threshold {tag.Threshold:F2}).");
                continue;
            }
            (int X, int Y, int Width, int Height)? bounds =
                FeatureImaging.MaskBounds(mask, source.Width, source.Height, CropOversize, BoundsThreshold);
            if (bounds is null)
            {
                Logs.Warning($"[Features][ReferenceCropResolver] Reference {tag.PictureIndex} left uncropped "
                    + $"('{tag.Query}' mask selected no pixels after all).");
                continue;
            }
            (int x, int y, int width, int height) = bounds.Value;
            if (width >= source.Width && height >= source.Height)
            {
                Logs.Info($"[Features][ReferenceCropResolver] Reference {tag.PictureIndex} left uncropped "
                    + $"(grown '{tag.Query}' match covers the whole image).");
                continue;
            }
            cropped[index] = FeatureImaging.CropRgb24(source, x, y, width, height);
            Logs.Info($"[Features][ReferenceCropResolver] Cropped reference {tag.PictureIndex} to "
                + $"{width}x{height} at ({x},{y}) of {source.Width}x{source.Height}, matching '{tag.Query}'.");
        }
        return request with { Prompt = strippedPrompt, ReferenceImages = cropped };
    }

    /// <summary>Parses every <c>&lt;refcrop:...&gt;</c> tag out of <paramref name="prompt"/>, returning the prompt with every tag span removed (nothing else touched — unlike <see cref="SegmentRefinement.StripSegmentText"/> this tag has no accumulated "sub-prompt" that follows it; everything after a <c>&lt;refcrop:&gt;</c> tag is ordinary continuing prompt text) plus the parsed tags in prompt order. An unterminated tag (no closing <c>&gt;</c>) is left as literal text, matching <see cref="PromptRegionParser"/>'s own tolerance for the same case. A tag whose data doesn't parse (missing/invalid index, empty query) is dropped with a warning — it cannot be left in place, since un-stripped tag text is exactly the base-prompt leak this class exists to avoid.</summary>
    private static (string StrippedPrompt, List<CropTag> Tags) Extract(string prompt)
    {
        const string prefix = "<refcrop:";
        List<CropTag> tags = [];
        StringBuilder result = new();
        int i = 0;
        while (i < prompt.Length)
        {
            int tagStart = prompt.IndexOf(prefix, i, StringComparison.Ordinal);
            if (tagStart < 0)
            {
                result.Append(prompt, i, prompt.Length - i);
                break;
            }
            result.Append(prompt, i, tagStart - i);
            int tagEnd = prompt.IndexOf('>', tagStart);
            if (tagEnd < 0)
            {
                result.Append(prompt, tagStart, prompt.Length - tagStart);
                break;
            }
            string data = prompt[(tagStart + prefix.Length)..tagEnd];
            if (TryParse(data, out CropTag? tag))
            {
                tags.Add(tag!);
            }
            else
            {
                Logs.Warning($"[Features][ReferenceCropResolver] <refcrop:{data}> could not be parsed — expected "
                    + "<refcrop:N,query> or <refcrop:N,query,threshold> (N is the 1-based Picture index). Tag dropped.");
            }
            i = tagEnd + 1;
        }
        return (result.ToString(), tags);
    }

    /// <summary>Same right-to-left optional-trailing-float convention <c>PromptRegionParser.ParseSegmentData</c> uses for <c>&lt;segment:&gt;</c>, minus the second trailing float (no creativity concept for a crop) and plus a REQUIRED leading index (no sensible default for "which reference").</summary>
    private static bool TryParse(string data, out CropTag? tag)
    {
        tag = null;
        int firstComma = data.IndexOf(',');
        if (firstComma < 0 || !int.TryParse(data[..firstComma], out int index) || index < 1)
        {
            return false;
        }
        string rest = data[(firstComma + 1)..];
        float threshold = DefaultThreshold;
        string query = rest;
        int lastComma = rest.LastIndexOf(',');
        if (lastComma >= 0 && float.TryParse(rest[(lastComma + 1)..], out float parsedThreshold))
        {
            threshold = Math.Clamp(parsedThreshold, 0.01f, 0.99f);
            query = rest[..lastComma];
        }
        query = query.Trim();
        if (query.Length == 0)
        {
            return false;
        }
        tag = new CropTag(index, query, threshold);
        return true;
    }
}
