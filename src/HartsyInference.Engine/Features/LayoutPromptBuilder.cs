using System.Globalization;
using System.Text;
using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Engine.Features;

/// <summary>Turns SwarmUI <c>&lt;region:&gt;</c>/<c>&lt;object:&gt;</c> prompt parts into a native layout table for models
/// trained on one (FLUX.3, ComfyUI <c>Flux3LayoutPrompt</c>): the untagged prompt, a space, then
/// <c>[{"id": "obj_N", "bbox": [ymin, xmin, ymax, xmax], "desc": "…"}]</c> with coordinates on a 0–1000 grid,
/// serialized exactly as Python's <c>json.dumps(…, ensure_ascii=False)</c> writes it.</summary>
public static class LayoutPromptBuilder
{
    private const int Grid = 1000;

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
