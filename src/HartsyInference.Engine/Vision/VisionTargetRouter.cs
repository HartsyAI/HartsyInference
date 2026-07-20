using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Vision;

/// <summary>Decodes a vision target string into the detector to run, mirroring the SwarmUI extension's
/// <c>&lt;segment:...&gt;</c> routing: <c>yolo-</c> → YOLO, <c>dino-</c> → Grounding DINO, <c>rtdetr</c> → RT-DETR,
/// anything else → open-vocabulary (Grounding DINO for boxes, CLIPSeg for masks), empty → RT-DETR.</summary>
public static class VisionTargetRouter
{
    private const string YoloPrefix = "yolo-";
    private const string DinoPrefix = "dino-";
    private const string RtDetrPrefix = "rtdetr";

    /// <summary>True when the prompt-region part targets a named YOLO checkpoint.</summary>
    public static bool IsYoloTarget(PromptRegionParser.Part part) => IsYoloTarget(part?.DataText);

    /// <summary>True when the prompt-region part targets Grounding DINO open-vocabulary detection.</summary>
    public static bool IsDinoTarget(PromptRegionParser.Part part) => IsDinoTarget(part?.DataText);

    /// <summary>True when the prompt-region part targets RT-DETR closed-set COCO detection.</summary>
    public static bool IsRtDetrTarget(PromptRegionParser.Part part) => IsRtDetrTarget(part?.DataText);

    /// <summary>True when the raw target text carries the <c>yolo-</c> prefix.</summary>
    public static bool IsYoloTarget(string? target) =>
        target is not null && target.StartsWith(YoloPrefix, StringComparison.OrdinalIgnoreCase);

    /// <summary>True when the raw target text carries the <c>dino-</c> prefix.</summary>
    public static bool IsDinoTarget(string? target) =>
        target is not null && target.StartsWith(DinoPrefix, StringComparison.OrdinalIgnoreCase);

    /// <summary>True when the raw target text carries the <c>rtdetr</c> prefix.</summary>
    public static bool IsRtDetrTarget(string? target) =>
        target is not null && target.StartsWith(RtDetrPrefix, StringComparison.OrdinalIgnoreCase);

    /// <summary>Parses the request prompt into a routed target; <paramref name="mode"/> decides where an unprefixed
    /// free-text query goes (boxes → Grounding DINO, masks → CLIPSeg).</summary>
    public static VisionTarget Parse(string? prompt, VisionMode mode)
    {
        string target = (prompt ?? "").Trim();
        if (target.Length == 0)
        {
            return new VisionTarget { Kind = VisionTargetKind.RtDetr };
        }
        if (IsYoloTarget(target))
        {
            return ParseYolo(target[YoloPrefix.Length..]);
        }
        if (IsDinoTarget(target))
        {
            return ParseDino(target[DinoPrefix.Length..]);
        }
        if (IsRtDetrTarget(target))
        {
            return ParseRtDetr(target[RtDetrPrefix.Length..]);
        }
        if (mode == VisionMode.Segment)
        {
            return new VisionTarget { Kind = VisionTargetKind.ClipSeg, Query = target };
        }
        return new VisionTarget { Kind = VisionTargetKind.GroundingDino, Query = target };
    }

    /// <summary>Parses <c>MODEL[-INDEX][:CLASS]</c> for the YOLO path.</summary>
    private static VisionTarget ParseYolo(string spec)
    {
        string[] modelAndClass = spec.Split(':');
        string fullname = modelAndClass[0];
        string classFilter = modelAndClass.Length > 1 ? modelAndClass[1].Trim() : "";
        int index = -1;
        int lastDash = fullname.LastIndexOf('-');
        if (lastDash > 0 && int.TryParse(fullname[(lastDash + 1)..], out int parsedIndex))
        {
            index = parsedIndex;
            fullname = fullname[..lastDash];
        }
        return new VisionTarget
        {
            Kind = VisionTargetKind.Yolo,
            ModelName = fullname.Trim(),
            ClassFilter = classFilter,
            Index = index,
        };
    }

    /// <summary>Parses <c>QUERY[:INDEX]</c> for the Grounding DINO path.</summary>
    private static VisionTarget ParseDino(string spec)
    {
        (string body, int index) = SplitTrailingIndex(spec);
        return new VisionTarget
        {
            Kind = VisionTargetKind.GroundingDino,
            Query = body.Trim(),
            Index = index,
        };
    }

    /// <summary>Parses <c>[-CLASS][:INDEX]</c> for the RT-DETR path.</summary>
    private static VisionTarget ParseRtDetr(string spec)
    {
        if (spec.StartsWith('-'))
        {
            spec = spec[1..];
        }
        (string body, int index) = SplitTrailingIndex(spec);
        return new VisionTarget
        {
            Kind = VisionTargetKind.RtDetr,
            ClassFilter = body.Trim(),
            Index = index,
        };
    }

    /// <summary>Splits a trailing <c>:INDEX</c> off a target body; the index is -1 when absent.</summary>
    private static (string Body, int Index) SplitTrailingIndex(string spec)
    {
        int lastColon = spec.LastIndexOf(':');
        if (lastColon >= 0 && int.TryParse(spec[(lastColon + 1)..].Trim(), out int parsed))
        {
            return (spec[..lastColon], parsed);
        }
        return (spec, -1);
    }
}
