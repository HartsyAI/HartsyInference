using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Fibo 2.0's edit prompt (ComfyUI <c>comfy/text_encoders/fibo2.py</c> <c>tokenize_with_weights</c> with images, and the
/// diffusers pipeline's <c>_get_edit_prompt</c>): one compact JSON object holding a Qwen3-VL vision marker per image
/// (<c>image</c> for one picture, <c>image_1</c>..<c>image_N</c> for several), the caption of the result when the prompt carries
/// one (<c>structured_caption</c>), then <c>edit_instruction</c>. The markers stand where the vision tower's tokens go, so the
/// instruction's <c>&lt;image_k&gt;</c> references keep their plain-text meaning and the images reach the model through the
/// markers, not through the words.</summary>
public static class Fibo2EditPrompt
{
    /// <summary>Where one image stands in the prompt before its vision tokens are expanded.</summary>
    public const string VisionMarker = "<|vision_start|><|image_pad|><|vision_end|>";

    /// <summary>Qwen3-VL's <c>&lt;|vision_start|&gt;</c> id.</summary>
    public const int VisionStartId = 151652;

    /// <summary>Qwen3-VL's <c>&lt;|vision_end|&gt;</c> id.</summary>
    public const int VisionEndId = 151653;

    /// <summary>Qwen3-VL's <c>&lt;|image_pad|&gt;</c> id: one per merged vision token.</summary>
    public const int ImagePadId = 151655;

    private static readonly string[] ScoreKeys = ["aesthetic_score", "preference_score"];

    private static readonly JsonSerializerOptions Compact = new()
    {
        WriteIndented = false,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary>The caption and edit instruction of a prompt (ComfyUI <c>parse_prompt</c>): a prompt that is not a JSON object comes
    /// back as itself with no instruction; an object gives its <c>edit_instruction</c>, and its caption (or
    /// <c>structured_caption</c>) with empty values and the score keys dropped, null when nothing is left.</summary>
    public static (JsonNode? Caption, JsonNode? Instruction) Parse(string prompt)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        JsonNode? parsed;
        try
        {
            parsed = JsonNode.Parse(prompt);
        }
        catch (JsonException)
        {
            return (JsonValue.Create(prompt), null);
        }
        if (parsed is not JsonObject root)
            return (JsonValue.Create(prompt), null);

        JsonNode? instruction = Take(root, "edit_instruction");
        if (root.ContainsKey("structured_caption"))
        {
            JsonNode? inner = Take(root, "structured_caption");
            if (inner is not JsonObject innerObject)
                return (Truthy(inner) ? inner : null, instruction);
            root = innerObject;
        }

        HashSet<string> keys = [.. root.Select(pair => pair.Key)];
        JsonObject stripped = new();
        foreach ((string key, JsonNode? value) in root.ToArray())
        {
            if (Array.IndexOf(ScoreKeys, key) >= 0)
                continue;
            if (value is JsonObject section)
            {
                JsonObject without = new();
                foreach ((string innerKey, JsonNode? innerValue) in section.ToArray())
                {
                    if (Array.IndexOf(ScoreKeys, innerKey) < 0)
                        without[innerKey] = Detach(innerValue);
                }
                stripped[key] = without;
            }
            else
            {
                stripped[key] = Detach(value);
            }
        }
        JsonObject caption = (JsonObject)Clean(stripped)!;
        if (keys.SetEquals(["short_description", "text_render"]))
        {
            // Exactly a short_description and a text_render list keeps both, the list even when empty: the image has no text.
            caption = new JsonObject
            {
                ["short_description"] = caption.TryGetPropertyValue("short_description", out JsonNode? description) && Truthy(description)
                    ? Detach(description) : JsonValue.Create(""),
                ["text_render"] = caption.TryGetPropertyValue("text_render", out JsonNode? render) && Truthy(render)
                    ? Detach(render) : new JsonArray(),
            };
        }
        return (caption.Count > 0 ? caption : null, instruction);
    }

    /// <summary>The edit prompt as compact JSON with <paramref name="imageCount"/> vision markers: the prompt is the instruction
    /// unless it is a JSON object carrying an <c>edit_instruction</c>.</summary>
    /// <exception cref="NotSupportedException">The caption asks for glyph text images (<c>text_render</c> entries with text).</exception>
    public static string Build(string prompt, int imageCount)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        if (imageCount < 1)
            throw new ArgumentOutOfRangeException(nameof(imageCount), imageCount, "An edit prompt needs at least one image.");
        (JsonNode? caption, JsonNode? instruction) = Parse(prompt);
        if (instruction is null)
        {
            // The prompt is the instruction alone.
            caption = null;
            instruction = JsonValue.Create(prompt);
        }
        if (HasGlyphText(caption))
            throw new NotSupportedException("glyph text images are not implemented yet for Fibo 2.0 (text_render entries)");

        JsonObject edit = new();
        if (imageCount == 1)
        {
            edit["image"] = VisionMarker;
        }
        else
        {
            for (int k = 1; k <= imageCount; k++)
                edit[$"image_{k}"] = VisionMarker;
        }
        if (Truthy(caption))
            edit["structured_caption"] = caption;
        edit["edit_instruction"] = instruction;
        return edit.ToJsonString(Compact);
    }

    /// <summary>Tokenizes <paramref name="editJson"/> (from <see cref="Build"/>) inside the Fibo 2.0 chat template, each vision
    /// marker expanded to <c>&lt;|vision_start|&gt;</c>, <paramref name="padCounts"/>[i] <c>&lt;|image_pad|&gt;</c> ids and
    /// <c>&lt;|vision_end|&gt;</c>. The marker splits the text as a special token does in a Hugging Face tokenizer: the text on
    /// either side is tokenized on its own.</summary>
    public static int[] Tokenize(Qwen3Tokenizer tokenizer, string editJson, IReadOnlyList<int> padCounts)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(editJson);
        ArgumentNullException.ThrowIfNull(padCounts);
        string[] parts = editJson.Split(VisionMarker);
        if (parts.Length - 1 != padCounts.Count)
            throw new ArgumentException($"The prompt holds {parts.Length - 1} vision markers for {padCounts.Count} images.", nameof(padCounts));

        List<int> ids = [Qwen3Tokenizer.ImStartId];
        // "user\n" and the text before the first marker are one tokenization, as in the chat template.
        ids.AddRange(tokenizer.EncodeRaw("user\n" + parts[0]));
        for (int i = 0; i < padCounts.Count; i++)
        {
            if (padCounts[i] < 1)
                throw new ArgumentOutOfRangeException(nameof(padCounts), $"Image {i + 1} needs at least one vision token, got {padCounts[i]}.");
            ids.Add(VisionStartId);
            for (int p = 0; p < padCounts[i]; p++)
                ids.Add(ImagePadId);
            ids.Add(VisionEndId);
            if (parts[i + 1].Length > 0)
                ids.AddRange(tokenizer.EncodeRaw(parts[i + 1]));
        }
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("assistant\n"));
        return [.. ids];
    }

    private static JsonNode? Take(JsonObject obj, string key)
    {
        if (!obj.TryGetPropertyValue(key, out JsonNode? value))
            return null;
        obj.Remove(key);
        return value;
    }

    private static JsonNode? Detach(JsonNode? node) => node?.DeepClone();

    private static bool Empty(JsonNode? node) => node switch
    {
        null => true,
        JsonObject o => o.Count == 0,
        JsonArray a => a.Count == 0,
        JsonValue v when v.TryGetValue(out string? s) => s.Length == 0,
        _ => false,
    };

    /// <summary>Python truthiness for the values a caption holds.</summary>
    private static bool Truthy(JsonNode? node)
    {
        if (Empty(node))
            return false;
        if (node is JsonValue value)
        {
            if (value.TryGetValue(out bool b))
                return b;
            if (value.TryGetValue(out double d))
                return d != 0;
        }
        return true;
    }

    /// <summary>Training captions hold no empty values (null, "", {}, []), at any depth.</summary>
    private static JsonNode? Clean(JsonNode? node)
    {
        switch (node)
        {
            case JsonObject obj:
                JsonObject cleanedObject = new();
                foreach ((string key, JsonNode? value) in obj.ToArray())
                {
                    JsonNode? cleaned = Clean(value);
                    if (!Empty(cleaned))
                        cleanedObject[key] = cleaned;
                }
                return cleanedObject;
            case JsonArray array:
                JsonArray cleanedArray = new();
                foreach (JsonNode? item in array.ToArray())
                {
                    JsonNode? cleaned = Clean(item);
                    if (!Empty(cleaned))
                        cleanedArray.Add(cleaned);
                }
                return cleanedArray;
            default:
                return node?.DeepClone();
        }
    }

    private static bool HasGlyphText(JsonNode? caption)
    {
        if (caption is not JsonObject obj || obj["text_render"] is not JsonArray entries)
            return false;
        foreach (JsonNode? entry in entries)
        {
            if (entry is JsonObject e && e["text"] is JsonValue text && text.TryGetValue(out string? s) && s.Length > 0)
                return true;
        }
        return false;
    }
}
