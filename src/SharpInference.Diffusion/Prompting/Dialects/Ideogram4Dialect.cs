using System.Buffers;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.RegularExpressions;
using SharpInference.Core.Exceptions;

namespace SharpInference.Diffusion.Prompting.Dialects;

/// <summary>Serializes a <see cref="StructuredPrompt"/> into the exact structured-JSON caption Ideogram 4 was trained on (upstream <c>docs/prompting.md</c>). The model is sensitive to these details, so the serializer is hand-written via <see cref="Utf8JsonWriter"/> to guarantee:
/// <list type="bullet">
///   <item>Fixed top-level key order: <c>high_level_description</c>, <c>style_description</c>, <c>compositional_deconstruction</c>.</item>
///   <item>Fixed element key order: object → <c>type, bbox?, desc, color_palette?</c>; text → <c>type, bbox?, text, desc, color_palette?</c>.</item>
///   <item>Compact separators (no spaces), literal Unicode (no <c>\uXXXX</c> escaping — needed for multilingual rendered text).</item>
///   <item>Uppercase <c>#RRGGBB</c> hex; <c>bbox = [y_min, x_min, y_max, x_max]</c> integers in 0–1000.</item>
/// </list>
/// Call <see cref="Validate"/> (or pass <c>validate: true</c> to <see cref="Serialize(StructuredPrompt, bool)"/>) to surface the common mistakes as actionable errors.</summary>
public sealed class Ideogram4Dialect : IPromptDialect
{
    private static readonly Regex _hexColor = new("^#[0-9A-Fa-f]{6}$", RegexOptions.Compiled);
    private static readonly JsonWriterOptions _writerOptions = new()
    {
        Indented = false,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <inheritdoc/>
    public string Name => "ideogram4";

    /// <inheritdoc/>
    public string Serialize(StructuredPrompt prompt) => Serialize(prompt, validate: true);

    /// <summary>Serializes the prompt; when <paramref name="validate"/> is true, runs <see cref="Validate"/> first.</summary>
    public string Serialize(StructuredPrompt prompt, bool validate)
    {
        if (validate) Validate(prompt);

        ArrayBufferWriter<byte> buffer = new();
        using (Utf8JsonWriter writer = new(buffer, _writerOptions))
        {
            writer.WriteStartObject();

            if (prompt.Summary is not null)
                writer.WriteString("high_level_description", prompt.Summary);

            if (prompt.Style is not null)
                WriteStyle(writer, prompt.Style);

            writer.WriteStartObject("compositional_deconstruction");
            writer.WriteString("background", prompt.Background ?? "");
            writer.WriteStartArray("elements");
            foreach (PromptElement element in prompt.Elements)
                WriteElement(writer, element);
            writer.WriteEndArray();
            writer.WriteEndObject();

            writer.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    private static void WriteStyle(Utf8JsonWriter writer, StyleBlock style)
    {
        writer.WriteStartObject("style_description");
        if (style.Aesthetics is not null) writer.WriteString("aesthetics", style.Aesthetics);
        if (style.Lighting is not null) writer.WriteString("lighting", style.Lighting);
        if (style.Medium is not null) writer.WriteString("medium", style.Medium);
        if (style.Photo is not null) writer.WriteString("photo", style.Photo);
        else if (style.ArtStyle is not null) writer.WriteString("art_style", style.ArtStyle);
        if (style.ColorPalette.Count > 0) WritePalette(writer, style.ColorPalette);
        writer.WriteEndObject();
    }

    private static void WriteElement(Utf8JsonWriter writer, PromptElement element)
    {
        writer.WriteStartObject();
        switch (element)
        {
            case TextElement text:
                writer.WriteString("type", "text");
                if (text.Bbox is BoundingBox tb) WriteBbox(writer, tb);
                writer.WriteString("text", text.Text);
                writer.WriteString("desc", text.Description);
                break;
            case ObjectElement obj:
                writer.WriteString("type", "obj");
                if (obj.Bbox is BoundingBox ob) WriteBbox(writer, ob);
                writer.WriteString("desc", obj.Description);
                break;
            default:
                throw new SharpInferenceException($"Unknown prompt element type: {element.GetType().Name}.");
        }
        if (element.ColorPalette.Count > 0) WritePalette(writer, element.ColorPalette);
        writer.WriteEndObject();
    }

    private static void WriteBbox(Utf8JsonWriter writer, BoundingBox b)
    {
        writer.WriteStartArray("bbox");
        writer.WriteNumberValue(b.YMin);
        writer.WriteNumberValue(b.XMin);
        writer.WriteNumberValue(b.YMax);
        writer.WriteNumberValue(b.XMax);
        writer.WriteEndArray();
    }

    private static void WritePalette(Utf8JsonWriter writer, IReadOnlyList<string> palette)
    {
        writer.WriteStartArray("color_palette");
        foreach (string c in palette)
            writer.WriteStringValue(c.ToUpperInvariant());
        writer.WriteEndArray();
    }

    /// <summary>Validates the prompt against Ideogram's schema constraints, throwing <see cref="SharpInferenceException"/> on the common mistakes (bad/short/lowercase hex, palette over limits, both photo+art_style set, missing background, malformed bbox).</summary>
    public static void Validate(StructuredPrompt prompt)
    {
        if (prompt.Background is null)
            throw new SharpInferenceException("Ideogram 4 requires a non-null background in compositional_deconstruction.");

        if (prompt.Style is { } style)
        {
            if (style.Photo is not null && style.ArtStyle is not null)
                throw new SharpInferenceException("style_description must set exactly one of photo / art_style, not both.");
            ValidatePalette(style.ColorPalette, 16, "style_description.color_palette");
        }
        ValidatePalette(prompt.ColorPalette, 16, "color_palette");

        foreach (PromptElement element in prompt.Elements)
        {
            element.Bbox?.Validate();
            ValidatePalette(element.ColorPalette, 5, "element.color_palette");
            if (element is TextElement t && string.IsNullOrEmpty(t.Text))
                throw new SharpInferenceException("A text element must have non-empty Text.");
        }
    }

    private static void ValidatePalette(IReadOnlyList<string> palette, int max, string where)
    {
        if (palette.Count > max)
            throw new SharpInferenceException($"{where} has {palette.Count} colors, max is {max}.");
        foreach (string c in palette)
            if (!_hexColor.IsMatch(c))
                throw new SharpInferenceException($"{where}: '{c}' is not a #RRGGBB hex color (shorthand and non-hex are rejected).");
    }
}
