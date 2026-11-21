namespace HartsyInference.ModelAssets.Metadata;

/// <summary>The closed vocabulary of <see cref="PromptProfile"/>, spelled exactly as the SwarmUI-LLMAssistant extension
/// spells it so a consumer reads the engine's values without translating them. Plain strings, not enums: the API's
/// string-enum converter would write <c>TextRendering</c> where the wire value is <c>text-rendering</c>.</summary>
public static class PromptVocabulary
{
    /// <summary>The domains a recipe serves.</summary>
    public static readonly IReadOnlyList<string> Kinds = ["image", "video"];

    /// <summary>The kinds of text encoder: a CLIP tower, a T5-family encoder, a text LLM, a vision-language model.</summary>
    public static readonly IReadOnlyList<string> Encoders = ["clip", "t5", "llm", "vlm"];

    /// <summary>What the finished prompt looks like.</summary>
    public static readonly IReadOnlyList<string> Formats = ["text", "tags", "json"];

    /// <summary>How the model takes a negative prompt: not at all, as a separate field, or written into the prompt.</summary>
    public static readonly IReadOnlyList<string> Negatives = ["none", "separate", "inline"];

    /// <summary>What the model honours in a prompt.</summary>
    public static readonly IReadOnlyList<string> Supports = ["weights", "break", "regions", "boxes", "text-rendering"];

    /// <summary>The <c>supports</c> value that makes a box form meaningful.</summary>
    public const string Boxes = "boxes";

    /// <summary>Box axis orders: <c>yxyx</c> is <c>[y_min, x_min, y_max, x_max]</c>, <c>xyxy</c> is <c>[x_min, y_min, x_max, y_max]</c>.</summary>
    public static readonly IReadOnlyList<string> BoxAxes = ["yxyx", "xyxy"];
}
