namespace HartsyInference.Engine.Features;

/// <summary>Engine-native parser for the structural prompt tag syntax (<c>&lt;region:x,y,w,h,strength&gt;</c>, <c>&lt;object:…&gt;</c>, <c>&lt;segment:…&gt;</c>, <c>&lt;clear:…&gt;</c>, <c>&lt;extend:…&gt;</c>, plus the stage tags <c>&lt;base&gt;</c>/<c>&lt;refiner&gt;</c>/<c>&lt;video&gt;</c>/<c>&lt;videoswap&gt;</c>/<c>&lt;pixeldecoder&gt;</c>). Text before any tag — and after a <c>&lt;region:end&gt;</c> — accumulates into <see cref="GlobalPrompt"/>; each other tag opens a section that subsequent text appends to. Unrecognized <c>&lt;…&gt;</c> runs (weighting, <c>&lt;break&gt;</c>, <c>&lt;embed:…&gt;</c>) are preserved verbatim so encoder-level syntax reaches the tokenizer untouched.</summary>
public sealed class PromptRegionParser
{
    /// <summary>Kind of a parsed prompt part.</summary>
    public enum PartType
    {
        /// <summary>A rectangular regional-conditioning area.</summary>
        Region,

        /// <summary>A rectangular area that additionally gets an inpaint-back pass.</summary>
        Object,

        /// <summary>A mask-driven segment refinement.</summary>
        Segment,

        /// <summary>A mask-driven segment erase.</summary>
        ClearSegment,

        /// <summary>An outpaint/extend directive.</summary>
        Extend,
    }

    /// <summary>One parsed prompt part: its text plus the geometry or data payload of its tag.</summary>
    public sealed class Part
    {
        /// <summary>The part's prompt text.</summary>
        public string Prompt { get; set; } = "";

        /// <summary>Left edge as a fraction of image width (region/object only).</summary>
        public float X { get; set; }

        /// <summary>Top edge as a fraction of image height (region/object only).</summary>
        public float Y { get; set; }

        /// <summary>Width as a fraction of image width (region/object only).</summary>
        public float Width { get; set; }

        /// <summary>Height as a fraction of image height (region/object only).</summary>
        public float Height { get; set; }

        /// <summary>Primary strength: conditioning weight for regions, mask threshold for segments.</summary>
        public double Strength { get; set; } = 1;

        /// <summary>Secondary strength: creativity for segments, region-2 weight for regions.</summary>
        public double Strength2 { get; set; } = 1;

        /// <summary>Raw tag payload for the non-rectangular part types (segment matcher, extend direction).</summary>
        public string? DataText { get; set; }

        /// <summary>Which tag produced this part.</summary>
        public PartType Type { get; set; }

        /// <summary>The tag prefix without angle brackets or colon (e.g. "region").</summary>
        public string Prefix { get; set; } = "";

        /// <summary>The <c>//cid=N</c> context id carried on the tag, or -1 when absent.</summary>
        public int ContextID { get; set; } = -1;
    }

    /// <summary>Tag prefixes that trigger the full parse; a prompt containing none is returned verbatim as the global prompt.</summary>
    private static readonly string[] _partPrefixes =
        ["<region:", "<object:", "<segment:", "<clear:", "<extend:", "<refiner", "<base", "<video", "<pixeldecoder"];

    /// <summary>The untagged text (and anything after a <c>&lt;region:end&gt;</c>).</summary>
    public string GlobalPrompt { get; private set; } = "";

    /// <summary>Text under <c>&lt;region:background&gt;</c>.</summary>
    public string BackgroundPrompt { get; private set; } = "";

    /// <summary>Text under <c>&lt;base&gt;</c>.</summary>
    public string BasePrompt { get; private set; } = "";

    /// <summary>Text under <c>&lt;refiner&gt;</c>.</summary>
    public string RefinerPrompt { get; private set; } = "";

    /// <summary>Text under <c>&lt;pixeldecoder&gt;</c>.</summary>
    public string PixelDecoderPrompt { get; private set; } = "";

    /// <summary>Text under <c>&lt;video&gt;</c>.</summary>
    public string VideoPrompt { get; private set; } = "";

    /// <summary>Text under <c>&lt;videoswap&gt;</c>.</summary>
    public string VideoSwapPrompt { get; private set; } = "";

    /// <summary>The parsed region/object/segment/clear/extend parts, in prompt order.</summary>
    public List<Part> Parts { get; } = [];

    /// <summary>Parses <paramref name="prompt"/>; a prompt with no structural tag becomes the global prompt verbatim.</summary>
    public PromptRegionParser(string? prompt)
    {
        string text = prompt ?? "";
        if (!Array.Exists(_partPrefixes, p => text.Contains(p, StringComparison.Ordinal)))
        {
            GlobalPrompt = text;
            return;
        }
        Parse(text);
    }

    private void Parse(string prompt)
    {
        string[] pieces = prompt.Split('<');
        bool first = true;
        // Which section trailing text flows into; rebound whenever a tag opens a new section.
        Action<string> addMore = s => GlobalPrompt += s;
        int id = -1;
        foreach (string piece in pieces)
        {
            if (first)
            {
                first = false;
                addMore(piece);
                continue;
            }
            int end = piece.IndexOf('>', StringComparison.Ordinal);
            if (end == -1)
            {
                addMore($"<{piece}");
                continue;
            }
            string tag = piece[..end];
            int cidAt = tag.LastIndexOf("//cid=", StringComparison.Ordinal);
            if (cidAt >= 0 && int.TryParse(tag[(cidAt + 6)..], out int cid))
            {
                id = cid;
                tag = tag[..cidAt];
            }
            int colon = tag.IndexOf(':', StringComparison.Ordinal);
            string prefix = colon < 0 ? tag : tag[..colon];
            string regionData = colon < 0 ? "" : tag[(colon + 1)..];
            string content = piece[(end + 1)..];
            PartType type;
            if (prefix == "region")
            {
                type = PartType.Region;
                if (regionData == "end")
                {
                    GlobalPrompt += content;
                    addMore = s => GlobalPrompt += s;
                    continue;
                }
                if (regionData == "background")
                {
                    BackgroundPrompt += content;
                    addMore = s => BackgroundPrompt += s;
                    continue;
                }
            }
            else if (prefix == "base")
            {
                BasePrompt += content;
                addMore = s => BasePrompt += s;
                continue;
            }
            else if (prefix == "refiner")
            {
                RefinerPrompt += content;
                addMore = s => RefinerPrompt += s;
                continue;
            }
            else if (prefix == "pixeldecoder")
            {
                PixelDecoderPrompt += content;
                addMore = s => PixelDecoderPrompt += s;
                continue;
            }
            else if (prefix == "video")
            {
                VideoPrompt += content;
                addMore = s => VideoPrompt += s;
                continue;
            }
            else if (prefix == "videoswap")
            {
                VideoSwapPrompt += content;
                addMore = s => VideoSwapPrompt += s;
                continue;
            }
            else if (prefix == "object")
            {
                type = PartType.Object;
            }
            else if (prefix == "extend")
            {
                type = PartType.Extend;
            }
            else if (prefix == "segment")
            {
                type = PartType.Segment;
            }
            else if (prefix == "clear")
            {
                type = PartType.ClearSegment;
            }
            else
            {
                addMore($"<{piece}");
                continue;
            }
            Part p = new Part
            {
                Prompt = content,
                Type = type,
                Prefix = prefix,
                ContextID = id,
            };
            string[] coords = regionData.Split(',');
            if (type is PartType.Segment or PartType.ClearSegment)
            {
                ParseSegmentData(p, regionData, coords);
            }
            else if (type == PartType.Extend)
            {
                p.DataText = regionData;
            }
            else if (!TryParseRect(p, coords))
            {
                addMore($"<{piece}");
                continue;
            }
            Parts.Add(p);
            addMore = s => p.Prompt += s;
        }
        // An empty <segment> inherits the global prompt; an empty <extend> inherits the preceding part's prompt.
        string previous = GlobalPrompt;
        foreach (Part part in Parts)
        {
            if (part.Type == PartType.Segment && string.IsNullOrWhiteSpace(part.Prompt))
            {
                part.Prompt = GlobalPrompt;
            }
            if (part.Type == PartType.Extend && string.IsNullOrWhiteSpace(part.Prompt))
            {
                part.Prompt = previous;
            }
            previous = part.Prompt;
        }
    }

    private static void ParseSegmentData(Part p, string regionData, string[] coords)
    {
        p.DataText = regionData;
        if (coords.Length > 1 && float.TryParse(coords[^1], out float x))
        {
            p.Strength = Math.Clamp(x, -1, 1);
            p.DataText = string.Join(",", coords[..^1]);
        }
        else if (regionData.StartsWith("yolo-", StringComparison.Ordinal))
        {
            p.Strength = 0.25;
        }
        else
        {
            p.Strength = 0.5;
        }
        if (coords.Length > 2 && float.TryParse(coords[^2], out float y))
        {
            p.Strength2 = Math.Clamp(y, 0, 1);
            p.DataText = string.Join(",", coords[..^2]);
        }
        else
        {
            p.Strength2 = 0.6;
        }
    }

    private static bool TryParseRect(Part p, string[] coords)
    {
        if (coords.Length < 4 || coords.Length > 6
            || !float.TryParse(coords[0], out float x)
            || !float.TryParse(coords[1], out float y)
            || !float.TryParse(coords[2], out float width)
            || !float.TryParse(coords[3], out float height))
        {
            return false;
        }
        double strength = coords.Length > 4 && double.TryParse(coords[4], out double s) ? s : 1.0;
        double strength2 = coords.Length > 5 && double.TryParse(coords[5], out double s2) ? s2 : 1.0;
        x = Math.Clamp(x, 0, 1);
        y = Math.Clamp(y, 0, 1);
        p.Strength = Math.Clamp(strength, -1, 1);
        p.Strength2 = Math.Clamp(strength2, 0, 1);
        p.X = x;
        p.Y = y;
        p.Width = Math.Clamp(width, 0, 1 - x);
        p.Height = Math.Clamp(height, 0, 1 - y);
        return true;
    }
}
