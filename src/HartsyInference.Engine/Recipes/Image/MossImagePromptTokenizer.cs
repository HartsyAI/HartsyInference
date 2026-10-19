using System.Text;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Renders a MOSS-Image prompt the way its pipeline does: Qwen3.5's chat template around a single user turn with
/// thinking enabled, <c>&lt;|im_start|&gt;user\n{prompt}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n&lt;think&gt;\n</c>
/// (transformers' <c>apply_chat_template(enable_thinking=True, add_generation_prompt=True)</c>, ComfyUI's
/// <c>MOSSImageTokenizer</c>), tokenized with special tokens matched in the text as the HF tokenizer does, NFC-normalized
/// like its <c>tokenizer.json</c> normalizer, and truncated to the 4096 tokens the model was trained on. No padding: the
/// model reads every prompt at its own length.</summary>
/// <remarks>Emphasis weights (<c>(word:1.2)</c>) never reach the encoder: ComfyUI's tokenizer disables them, so SwarmUI
/// encodes at weight 1 and scales each token's conditioning row afterwards (<see cref="PromptWeightingMode.CondScale"/>).</remarks>
public sealed class MossImagePromptTokenizer(ILlmTokenizer tokenizer, int maxTokens = MossImageConfig.MaxTextTokens)
{
    /// <summary>The template text before the prompt.</summary>
    public const string TemplatePrefix = "<|im_start|>user\n";

    /// <summary>The template text after the prompt: the end of the user turn and an open thinking block.</summary>
    public const string TemplateSuffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n";

    /// <summary>The special tokens the template spells out; a tokenizer missing any of them is not Qwen3.5's.</summary>
    public static IReadOnlyList<string> TemplateSpecialTokens { get; } = ["<|im_start|>", "<|im_end|>", "<think>"];

    /// <summary>The Qwen3.5 tokenizer.</summary>
    public ILlmTokenizer Tokenizer { get; } = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

    /// <summary>The token budget; longer prompts are truncated, as the reference truncates at its maximum sequence length.</summary>
    public int MaxTokens { get; } = maxTokens > 0 ? maxTokens : throw new ArgumentOutOfRangeException(nameof(maxTokens));

    /// <summary>Tokenizes <paramref name="prompt"/> (emphasis grammar included) into ids and per-token weights.</summary>
    public WeightedTokenSequence Tokenize(string prompt) => Tokenize(prompt, null, 0, -1);

    /// <summary>Tokenizes an edit prompt: <paramref name="placeholderPrefix"/> (the reference images' <c>Picture N:</c>
    /// blocks, <see cref="VisionPlaceholderPrompt.Render"/>) sits at the start of the user turn, before the prompt, as the
    /// Ostris edit node writes it. The run of <paramref name="imagePadId"/> must come out exactly
    /// <paramref name="expectedPlaceholders"/> long, and a prompt that would need truncating is refused instead, since a
    /// cut would leave the vision tokens without their placeholders.</summary>
    public WeightedTokenSequence Tokenize(string prompt, string? placeholderPrefix, int expectedPlaceholders, int imagePadId)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        string normalized = prompt.Normalize(NormalizationForm.FormC);
        string prefix = string.Concat(TemplatePrefix, placeholderPrefix);
        List<WeightedSpan> spans = [.. PromptWeighting.Parse(normalized)];
        WeightedTokenSequence sequence;
        if (!PromptWeighting.HasWeights(spans))
        {
            // One encode over the whole rendered template: the pre-tokenizer sees "user\n" and the prompt as one
            // segment, exactly as the HF tokenizer does, so a prompt opening with whitespace merges the same way.
            int[] ids = Encode(string.Concat(prefix, PromptWeighting.Join(spans), TemplateSuffix));
            float[] ones = new float[ids.Length];
            Array.Fill(ones, 1f);
            sequence = new WeightedTokenSequence(ids, ones) { UniformWeight = 1f };
        }
        else
        {
            sequence = WeightedTokenBuilder.Build(spans, Encode, Encode(prefix), Encode(TemplateSuffix));
        }
        if (string.IsNullOrEmpty(placeholderPrefix))
            return sequence.Tokens.Length > MaxTokens ? sequence.Truncate(MaxTokens) : sequence;
        if (sequence.Tokens.Length > MaxTokens)
        {
            throw new ArgumentException($"The edit prompt is {sequence.Tokens.Length} tokens with its reference images; MOSS-Image reads "
                + $"at most {MaxTokens}. Shorten the prompt or use fewer or smaller references.");
        }
        int pads = 0;
        foreach (int id in sequence.Tokens)
        {
            if (id == imagePadId) pads++;
        }
        if (pads != expectedPlaceholders)
            throw new InvalidDataException($"The tokenizer produced {pads} image placeholders for {expectedPlaceholders} vision tokens.");
        return sequence;
    }

    private int[] Encode(string text) => Tokenizer.Encode(text, addSpecial: true);
}
