using System.Runtime.InteropServices;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Tokenizes a prompt through whichever text stack a FLUX.2-family checkpoint carries, with the per-token
/// weights its emphasis grammar asks for: the Qwen3 chat template for the Klein builds, or the spliced Mistral tekken
/// conditioning ids for Dev. Exactly one of <paramref name="qwenTokenizer"/> / <paramref name="mistralTokenizer"/> is
/// non-null; both are borrowed.</summary>
public sealed class Flux2PromptTokenizer(Flux2Config config, Qwen3Tokenizer? qwenTokenizer, ErnieTokenizer? mistralTokenizer,
    string mistralSystemPrompt)
{
    private readonly Flux2Config _config = config;
    private readonly Qwen3Tokenizer? _qwenTokenizer = qwenTokenizer;
    private readonly ErnieTokenizer? _mistralTokenizer = mistralTokenizer;
    private readonly string _mistralSystemPrompt = mistralSystemPrompt;

    /// <summary>The prompt's token ids and weights. Klein's <see cref="Qwen3Tokenizer.EncodeChat"/> merges
    /// <c>user\n</c> with the prompt in ONE BPE call, which is load-bearing for a prompt that begins with whitespace —
    /// so an unweighted prompt keeps that call verbatim and only a weighted one splits per span.</summary>
    /// <remarks>A WEIGHTED prompt therefore tokenizes per leaf, and a leaf that begins with whitespace will not merge
    /// with the <c>user\n</c> before it: <c>"\n(cat:1.5)"</c> emits two newline tokens where <c>"\ncat"</c> emits
    /// one. That is SwarmUI's own behaviour — <c>calc_leaf</c> tokenizes each leaf alone — so it is parity, not drift,
    /// and <c>ChatTemplateWeightingTests</c> pins it so nobody later "fixes" it into a boundary shift.</remarks>
    public WeightedTokenSequence Tokenize(string prompt) => TokenizeWithLength(prompt).Tokens;

    /// <summary><see cref="Tokenize"/> plus the count of real tokens before any right padding: the Qwen3 stack pads to
    /// its maximum length, the Mistral stack does not pad.</summary>
    public (WeightedTokenSequence Tokens, int ValidLength) TokenizeWithLength(string prompt)
    {
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse(prompt);
        if (_config.TextEncoderType == Flux2TextEncoderType.Mistral)
        {
            WeightedTokenSequence mistral = BuildMistralDevTokens(_mistralTokenizer!, spans);
            return (mistral, mistral.Tokens.Length);
        }
        if (!PromptWeighting.HasWeights(spans))
        {
            (int[] plain, int realLength) = _qwenTokenizer!.EncodeChatWithLength(PromptWeighting.Join(spans));
            return (new WeightedTokenSequence(plain, Ones(plain.Length)) { UniformWeight = 1f }, realLength);
        }
        (int[] prefix, int[] suffix) = _qwenTokenizer!.ChatTemplateIds();
        WeightedTokenSequence built = WeightedTokenBuilder.Build(spans, _qwenTokenizer.EncodeRaw, prefix, suffix)
            .Truncate(_qwenTokenizer.MaxLength);
        return (built.PadTo(_qwenTokenizer.MaxLength, _qwenTokenizer.PadTokenId), built.Tokens.Length);
    }

    /// <summary>Builds Flux.2 Dev conditioning ids: <c>&lt;s&gt;[SYSTEM_PROMPT]sys[/SYSTEM_PROMPT][INST]prompt[/INST]</c>. Special markers are spliced as raw ids (BOS=1, [SYSTEM_PROMPT]=17, [/SYSTEM_PROMPT]=18, [INST]=3, [/INST]=4) around byte-level BPE segments — special strings are pre-token boundaries in the HF reference, so segment-wise encoding is id-exact. No EOS (ComfyUI <c>has_end_token=False</c>). The byte-level map is per-byte, so encoding each emphasis span separately concatenates to the same string the whole-prompt call would have produced.</summary>
    private WeightedTokenSequence BuildMistralDevTokens(ErnieTokenizer tokenizer, IReadOnlyList<WeightedSpan> spans)
    {
        List<int> prefix = new List<int>(256) { 1, 17 };
        prefix.AddRange(tokenizer.EncodeRaw(ByteLevelCodec.Encode(_mistralSystemPrompt)));
        prefix.Add(18);
        prefix.Add(3);
        return WeightedTokenBuilder.Build(spans, text => tokenizer.EncodeRaw(ByteLevelCodec.Encode(text)),
            CollectionsMarshal.AsSpan(prefix), [4]);
    }

    private static float[] Ones(int length)
    {
        float[] weights = new float[length];
        Array.Fill(weights, 1f);
        return weights;
    }
}
