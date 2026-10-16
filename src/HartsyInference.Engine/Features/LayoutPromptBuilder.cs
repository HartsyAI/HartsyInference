using System.Globalization;
using System.Text;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Engine.Features;

/// <summary>Turns SwarmUI <c>&lt;region:&gt;</c>/<c>&lt;object:&gt;</c> prompt parts into the native layout prompt of a
/// model trained on one, serialized exactly as the reference's Python <c>json.dumps(…, ensure_ascii=False)</c> writes it:
/// <list type="bullet">
/// <item>FLUX.3 (<see cref="Split"/>, ComfyUI <c>Flux3LayoutPrompt</c>): the untagged prompt, a space, then
/// <c>[{"id": "obj_N", "bbox": [ymin, xmin, ymax, xmax], "desc": "…"}]</c> on a 0–1000 grid.</item>
/// <item>MOSS-Image (<see cref="BuildSceneLayout"/>, OpenMOSS <c>compile_prompt.py</c>): one compact JSON object whose
/// <c>summary.caption</c> is the untagged prompt and whose <c>items</c> are the regions, boxed in pixels about the canvas
/// centre (<c>abs_centralized</c>, <c>yxyx</c>).</item>
/// </list></summary>
public static class LayoutPromptBuilder
{
    private const int Grid = 1000;

    /// <summary>Decimals a pixel edge is rounded to before it is floored or ceiled (see <see cref="CenteredPixelBox"/>).</summary>
    private const int EdgeDecimals = 3;

    /// <summary>Splits <paramref name="rawPrompt"/> into its untagged text and the layout table its region parts describe
    /// (null when it has none). Each region's emphasis grammar is dropped from its description, because the table is
    /// read as literal text. Cite box N in the prompt as <c>&lt;obj_N&gt;</c>.</summary>
    public static (string Prompt, string? Table) Split(string rawPrompt)
    {
        ArgumentNullException.ThrowIfNull(rawPrompt);
        if (!RegionalPromptResolver.HasRegionParts(rawPrompt))
            return (rawPrompt, null);
        PromptRegionParser parsed = new PromptRegionParser(rawPrompt);
        StringBuilder table = new StringBuilder("[");
        int index = 0;
        foreach (PromptRegionParser.Part part in parsed.Parts)
        {
            if (part.Type is not (PromptRegionParser.PartType.Region or PromptRegionParser.PartType.Object))
                continue;
            if (index > 0)
                table.Append(", ");
            index++;
            (int ymin, int xmin, int ymax, int xmax) = NormalizedBox(part.X, part.Y, part.Width, part.Height);
            table.Append("{\"id\": ");
            AppendJsonString(table, string.Create(CultureInfo.InvariantCulture, $"obj_{index}"));
            table.Append(CultureInfo.InvariantCulture, $", \"bbox\": [{ymin}, {xmin}, {ymax}, {xmax}], \"desc\": ");
            AppendJsonString(table, PromptWeighting.Join(PromptWeighting.Parse(part.Prompt)).Trim());
            table.Append('}');
        }
        if (index == 0)
            return (parsed.GlobalPrompt, null);
        table.Append(']');
        return (parsed.GlobalPrompt.TrimEnd(), table.ToString());
    }

    /// <summary>The ComfyUI <c>Flux3LayoutPrompt</c> string: the untagged prompt, a space, then the table; the prompt
    /// unchanged when it has no region parts.</summary>
    public static string Build(string rawPrompt)
    {
        (string prompt, string? table) = Split(rawPrompt);
        return table is null ? prompt : string.Concat(prompt, " ", table);
    }

    /// <summary>The MOSS-Image layout prompt for a <paramref name="width"/>×<paramref name="height"/> canvas, or
    /// <paramref name="rawPrompt"/> unchanged when it has no region parts. The untagged prompt becomes
    /// <c>summary.caption</c> as written (its emphasis grammar is read by the tokenizer like any prompt's), falling back to
    /// the region descriptions when it is blank, since the format always carries a caption; each region becomes an
    /// <c>object</c> item with its emphasis dropped, as in <see cref="Split"/>.</summary>
    public static string BuildSceneLayout(string rawPrompt, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(rawPrompt);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        if (!RegionalPromptResolver.HasRegionParts(rawPrompt))
            return rawPrompt;
        PromptRegionParser parsed = new PromptRegionParser(rawPrompt);
        List<string> captions = [];
        StringBuilder items = new StringBuilder();
        foreach (PromptRegionParser.Part part in parsed.Parts)
        {
            if (part.Type is not (PromptRegionParser.PartType.Region or PromptRegionParser.PartType.Object))
                continue;
            string caption = PromptWeighting.Join(PromptWeighting.Parse(part.Prompt)).Trim();
            (int ymin, int xmin, int ymax, int xmax) = CenteredPixelBox(part.X, part.Y, part.Width, part.Height, width, height);
            if (caption.Length == 0 || ymin >= ymax || xmin >= xmax)
            {
                Logs.Warning($"[Features][Layout] Dropping region at ({part.X}, {part.Y}, {part.Width}, {part.Height}): "
                    + (caption.Length == 0 ? "it has no description." : $"it is under a pixel on the {width}x{height} canvas."));
                continue;
            }
            captions.Add(caption);
            items.Append(items.Length == 0 ? "" : ",").Append("{\"label\":\"object\",\"caption\":");
            AppendJsonString(items, caption);
            items.Append(CultureInfo.InvariantCulture, $",\"bbox\":[{ymin},{xmin},{ymax},{xmax}]}}");
        }
        string summary = parsed.GlobalPrompt.Trim();
        if (captions.Count == 0)
            return summary;
        StringBuilder layout = new StringBuilder("{\"summary\":{\"caption\":");
        AppendJsonString(layout, summary.Length > 0 ? summary : string.Join("; ", captions));
        layout.Append("},\"items\":[").Append(items).Append("],\"bbox_coord_space\":\"abs_centralized\",\"bbox_format\":\"yxyx\"}");
        return layout.ToString();
    }

    /// <summary>A fractional <c>(x, y, w, h)</c> box as <c>[ymin, xmin, ymax, xmax]</c> in pixels about the centre of a
    /// <paramref name="width"/>×<paramref name="height"/> canvas, as OpenMOSS's <c>centered_box</c> exports it: the box
    /// clamped to the canvas, its top-left edges floored and bottom-right edges ceiled, then <c>height // 2</c> and
    /// <c>width // 2</c> subtracted. The reference rounds its pixel edges to 9 decimals first; these edges come from float
    /// fractions, whose last bit is worth ~1e-4 px on a 2K canvas, so they are rounded to a thousandth of a pixel instead,
    /// or an edge a region places exactly on pixel 800 could ceil to 801.</summary>
    public static (int Ymin, int Xmin, int Ymax, int Xmax) CenteredPixelBox(float x, float y, float boxWidth, float boxHeight,
        int width, int height)
    {
        double x2 = (double)x + boxWidth, y2 = (double)y + boxHeight;
        double left = Math.Clamp(Math.Min(x, x2) * width, 0, width);
        double right = Math.Clamp(Math.Max(x, x2) * width, 0, width);
        double top = Math.Clamp(Math.Min(y, y2) * height, 0, height);
        double bottom = Math.Clamp(Math.Max(y, y2) * height, 0, height);
        int halfW = width / 2, halfH = height / 2;
        return ((int)Math.Floor(Math.Round(top, EdgeDecimals)) - halfH, (int)Math.Floor(Math.Round(left, EdgeDecimals)) - halfW,
            (int)Math.Ceiling(Math.Round(bottom, EdgeDecimals)) - halfH, (int)Math.Ceiling(Math.Round(right, EdgeDecimals)) - halfW);
    }

    /// <summary>A fractional <c>(x, y, w, h)</c> box as <c>[ymin, xmin, ymax, xmax]</c> on the 0–1000 grid, rounded half
    /// to even like Python's <c>round</c>.</summary>
    public static (int Ymin, int Xmin, int Ymax, int Xmax) NormalizedBox(float x, float y, float width, float height)
    {
        int ymin = ToGrid(y), xmin = ToGrid(x), ymax = ToGrid(y + height), xmax = ToGrid(x + width);
        return (Math.Min(ymin, ymax), Math.Min(xmin, xmax), Math.Max(ymin, ymax), Math.Max(xmin, xmax));
    }

    private static int ToGrid(float value) => Math.Clamp((int)Math.Round((double)value * Grid, MidpointRounding.ToEven), 0, Grid);

    /// <summary>Python <c>json.dumps</c> string escaping with <c>ensure_ascii=False</c>.</summary>
    private static void AppendJsonString(StringBuilder sb, string value)
    {
        sb.Append('"');
        foreach (char c in value)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (c < ' ')
                        sb.Append(CultureInfo.InvariantCulture, $"\\u{(int)c:x4}");
                    else
                        sb.Append(c);
                    break;
            }
        }
        sb.Append('"');
    }
}
