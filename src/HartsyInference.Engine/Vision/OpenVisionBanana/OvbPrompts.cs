using System.Globalization;
using System.Text;

namespace HartsyInference.Engine.Vision.OpenVisionBanana;

/// <summary>One class of a segmentation legend: the name the model reads and the colour it is asked to paint it.</summary>
public readonly record struct OvbLegendEntry(string Name, byte R, byte G, byte B);

/// <summary>How a semantic-segmentation prompt spells its colour mapping.</summary>
public enum OvbSemsegStyle
{
    /// <summary>The release README's prompt: <c>{"road": [8,4,249], ...}</c>, "leave everything else black".</summary>
    Quickstart,

    /// <summary>The benchmark prompt of the evaluation code (<c>make_selective_prompt</c>): <c>{ 'road': [8, 4, 249], ... }</c>,
    /// pixels outside the listed classes "must be black and must not be forced into a listed class".</summary>
    Selective,
}

/// <summary>The task prompts Open Vision Banana was trained on, verbatim from FunAILab/Open-Vision-Banana's README and the
/// OpenVisionBanana/OpenVisionBanana evaluation code @ d457d3f. The model reads them as typed: they go to the text encoder
/// without emphasis, tag or region parsing (see <see cref="Recipes.Image.Flux2EditRecipePipeline"/>).</summary>
public static class OvbPrompts
{
    /// <summary>Longest class or phrase accepted: the encoder reads 512 tokens in all, and a long name starves the legend.</summary>
    public const int MaxNameLength = 80;

    /// <summary>Relative inverse depth as neutral grayscale (closer is brighter), README prompt.</summary>
    public const string Depth =
        "Estimate a dense relative inverse-depth map for the input image. Produce exactly one output value "
        + "for every input pixel at the same pixel location, with closer points brighter and farther points "
        + "darker. Encode the result as neutral grayscale (R=G=B) using one consistent mapping over the "
        + "complete image. Preserve object silhouettes, thin structures, occlusion boundaries, smooth surfaces, "
        + "and small depth changes. Do not alter the scene or add texture, shading, color, text, labels, "
        + "legends, borders, or objects; return only the registered depth map.";

    /// <summary>Camera-space unit surface normals, <c>R=(1-x)/2, G=(1+y)/2, B=(1+z)/2</c>, README prompt.</summary>
    public const string Normals =
        "Create a dense camera-space unit surface-normal map registered to the input. Use +x image-right, "
        + "+y image-up, +z toward the camera and encode (x,y,z) as R=(1-x)/2, G=(1+y)/2, B=(1+z)/2. "
        + "Preserve planes, curved surfaces, thin structures, and sharp orientation boundaries; "
        + "return only the RGB normal map.";

    /// <summary>The colour the referring-segmentation prompt asks the matching region to be painted.</summary>
    public static readonly (byte R, byte G, byte B) ReferringColor = (220, 30, 30);

    /// <summary>The semantic-segmentation prompt for <paramref name="legend"/>, classes in the order given.</summary>
    public static string SemanticSegmentation(IReadOnlyList<OvbLegendEntry> legend, OvbSemsegStyle style = OvbSemsegStyle.Quickstart)
    {
        ArgumentNullException.ThrowIfNull(legend);
        if (legend.Count == 0)
            throw new ArgumentException("A segmentation prompt needs at least one class.", nameof(legend));
        StringBuilder map = new();
        for (int i = 0; i < legend.Count; i++)
        {
            OvbLegendEntry entry = legend[i];
            ValidateName(entry.Name, nameof(legend), allowApostrophe: style == OvbSemsegStyle.Quickstart);
            if (i > 0)
                map.Append(", ");
            map.Append(style == OvbSemsegStyle.Quickstart ? '"' : '\'').Append(entry.Name).Append(style == OvbSemsegStyle.Quickstart ? '"' : '\'').Append(": ");
            string sep = style == OvbSemsegStyle.Quickstart ? "," : ", ";
            map.Append('[').Append(entry.R.ToString(CultureInfo.InvariantCulture)).Append(sep)
                .Append(entry.G.ToString(CultureInfo.InvariantCulture)).Append(sep)
                .Append(entry.B.ToString(CultureInfo.InvariantCulture)).Append(']');
        }
        return style == OvbSemsegStyle.Quickstart
            ? "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, using this color mapping: {"
                + map + "} Color each pixel of a listed class with its color and leave everything else black. "
                + "Output only the segmentation visualization image."
            : "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, "
                + "using this color mapping for the listed classes: { " + map + " }. "
                + "Paint pixels belonging to a listed class with exactly its RGB color. "
                + "Pixels that do not belong to one of these listed classes must be black [0, 0, 0] "
                + "and must not be forced into a listed class. Output only the segmentation visualization image.";
    }

    /// <summary>The referring-segmentation prompt: the region matching <paramref name="description"/> is painted
    /// <see cref="ReferringColor"/>, everything else black.</summary>
    public static string ReferringSegmentation(string description)
    {
        ValidateName(description, nameof(description), allowApostrophe: true);
        (byte r, byte g, byte b) = ReferringColor;
        return "[TASK: REFERRING_SEGMENTATION] Segment the region that best matches this description: \"" + description
            + "\". Paint the matching region red [" + r + ", " + g + ", " + b + "] and leave everything else black [0, 0, 0]. "
            + "Output only the segmentation mask image.";
    }

    /// <summary>Refuses a name that would break the prompt's own structure: it sits inside quotes inside a braced mapping. An
    /// apostrophe only breaks the single-quoted spelling, so the double-quoted ones allow it.</summary>
    internal static void ValidateName(string? name, string parameter, bool allowApostrophe = false)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("A class name or description cannot be empty.", parameter);
        if (name.Length > MaxNameLength)
            throw new ArgumentException($"'{name[..20]}...' is longer than {MaxNameLength} characters.", parameter);
        foreach (char c in name)
        {
            if (c is '"' or '\\' or '{' or '}' || (c == '\'' && !allowApostrophe) || char.IsControl(c))
                throw new ArgumentException($"'{name}' contains a character the prompt format cannot carry ({(char.IsControl(c) ? "a control character" : c.ToString())}).", parameter);
        }
    }
}
