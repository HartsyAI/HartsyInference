namespace HartsyInference.Diffusion.Prompting;

/// <summary>ComfyUI's <c>parse_prompt</c> for Fibo 2.0 (<c>comfy/text_encoders/fibo2.py</c>, commit 87ed5a066d9c): the caption as
/// training saw it. Score keys and empty values are dropped; a prompt that is not a JSON object comes back as it is.</summary>
internal static class Fibo2CaptionCleaner
{
    private static readonly string[] ScoreKeys = ["aesthetic_score", "preference_score"];

    /// <summary>Returns the cleaned caption: a <see cref="PyDict"/>, a string (the prompt as given), another JSON value from a
    /// non-object <c>structured_caption</c>, or null when nothing is left.</summary>
    public static object? Parse(string prompt)
    {
        if (!PythonJson.TryParse(prompt, out object? parsed) || parsed is not PyDict caption)
            return prompt;

        caption.Pop("edit_instruction");
        if (caption.ContainsKey("structured_caption"))
        {
            object? inner = caption.Pop("structured_caption");
            if (inner is not PyDict innerDict)
                return Truthy(inner) ? inner : null;
            caption = innerDict;
        }

        HashSet<string> keys = new(caption.Keys, StringComparer.Ordinal);
        PyDict filtered = new();
        foreach (string key in caption.Keys)
        {
            if (Array.IndexOf(ScoreKeys, key) >= 0)
                continue;
            object? value = caption.Get(key);
            if (value is PyDict nested)
            {
                PyDict withoutScores = new();
                foreach (string nestedKey in nested.Keys)
                {
                    if (Array.IndexOf(ScoreKeys, nestedKey) < 0)
                        withoutScores.Set(nestedKey, nested.Get(nestedKey));
                }
                value = withoutScores;
            }
            filtered.Set(key, value);
        }

        PyDict cleaned = (PyDict)Clean(filtered)!;
        if (keys.Count == 2 && keys.Contains("short_description") && keys.Contains("text_render"))
        {
            // exactly these two keys keep both, the list even when empty: the image has no text
            PyDict both = new();
            both.Set("short_description", Truthy(cleaned.Get("short_description")) ? cleaned.Get("short_description") : "");
            both.Set("text_render", Truthy(cleaned.Get("text_render")) ? cleaned.Get("text_render") : new List<object?>());
            cleaned = both;
        }
        return cleaned.Count == 0 ? null : cleaned;
    }

    private static object? Clean(object? value)
    {
        if (value is PyDict dict)
        {
            PyDict kept = new();
            foreach (string key in dict.Keys)
            {
                object? child = Clean(dict.Get(key));
                if (!IsEmpty(child))
                    kept.Set(key, child);
            }
            return kept;
        }
        if (value is List<object?> list)
        {
            List<object?> kept = [];
            foreach (object? item in list)
            {
                object? child = Clean(item);
                if (!IsEmpty(child))
                    kept.Add(child);
            }
            return kept;
        }
        return value;
    }

    private static bool IsEmpty(object? value) => value switch
    {
        null => true,
        double number => double.IsNaN(number),
        string text => text.Length == 0,
        PyDict dict => dict.Count == 0,
        List<object?> list => list.Count == 0,
        _ => false,
    };

    /// <summary>Python truthiness for the JSON value types.</summary>
    private static bool Truthy(object? value) => value switch
    {
        null => false,
        bool flag => flag,
        double number => number != 0,
        PyInt integer => integer.Digits != "0",
        string text => text.Length > 0,
        PyDict dict => dict.Count > 0,
        List<object?> list => list.Count > 0,
        _ => true,
    };
}
