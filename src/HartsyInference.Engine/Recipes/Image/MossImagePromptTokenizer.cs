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

    private readonly ILlmTokenizer _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
    private readonly int _maxTokens = maxTokens > 0 ? maxTokens : throw new ArgumentOutOfRangeException(nameof(maxTokens));

    /// <summary>Tokenizes <paramref name="prompt"/> (emphasis grammar included) into ids and per-token weights.</summary>
    public WeightedTokenSequence Tokenize(string prompt)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        string normalized = prompt.Normalize(NormalizationForm.FormC);
        List<WeightedSpan> spans = [.. PromptWeighting.Parse(normalized)];
        WeightedTokenSequence sequence;
        if (!PromptWeighting.HasWeights(spans))
        {
            // One encode over the whole rendered template: the pre-tokenizer sees "user\n" and the prompt as one
            // segment, exactly as the HF tokenizer does, so a prompt opening with whitespace merges the same way.
            int[] ids = Encode(string.Concat(TemplatePrefix, PromptWeighting.Join(spans), TemplateSuffix));
            float[] ones = new float[ids.Length];
            Array.Fill(ones, 1f);
            sequence = new WeightedTokenSequence(ids, ones) { UniformWeight = 1f };
        }
        else
        {
            sequence = WeightedTokenBuilder.Build(spans, Encode, Encode(TemplatePrefix), Encode(TemplateSuffix));
        }
        return sequence.Tokens.Length > _maxTokens ? sequence.Truncate(_maxTokens) : sequence;
    }

    private int[] Encode(string text) => _tokenizer.Encode(text, addSpecial: true);
}
