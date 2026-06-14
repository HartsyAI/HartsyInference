using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>HunyuanImage 2.1's primary text encoder: the Qwen2.5-VL-7B-Instruct text stack (<see cref="LlamaStyleEncoderConfig.Qwen2_5_VL_7B"/>) run as a feature extractor. Reproduces diffusers' <c>_get_qwen_prompt_embeds</c> (<c>pipeline_hunyuanimage.py</c>): the prompt is wrapped in the fixed encode template (see <see cref="SystemPrompt"/>; tokenize via <c>Qwen2Tokenizer.EncodeChat(prompt, systemPrompt: SystemPrompt, addGenerationPrompt: false)</c> + <c>PadToLength(PaddedLength)</c>), the model is tapped at <c>hidden_states[-3]</c> (HF index N−2 — skip the last two blocks and the final norm), and the first <see cref="TemplatePrefixTokens"/> rows (the template prefix) are dropped, leaving <c>[1, ≤1000, 3584]</c> conditioning.
///
/// <para>Instead of forwarding the full 1034-token padded window with an attention mask, the input is trimmed to its real (mask=1) length before the forward pass: with causal attention the kept rows are bit-identical to the masked padded run, and the DiT consumes only mask=1 tokens downstream — so trimming is exactly equivalent and skips the pad-token compute.</para></summary>
public sealed class HunyuanImageQwenTextEncoder : IDisposable
{
    /// <summary>System message of HunyuanImage 2.1's encode template (verbatim from <c>pipeline_hunyuanimage.py:prompt_template_encode</c>, including the trailing colon).</summary>
    public const string SystemPrompt =
        "Describe the image by detailing the color, shape, size, texture, quantity, text, " +
        "spatial relationships of the objects and background:";

    /// <summary>Token count of the template prefix (<c>prompt_template_encode_start_idx</c>) — dropped from the hidden states before the DiT.</summary>
    public const int TemplatePrefixTokens = 34;

    /// <summary>Padded tokenizer window: <c>tokenizer_max_length</c> 1000 + the 34-token template prefix.</summary>
    public const int PaddedLength = 1034;

    private readonly LlamaStyleEncoder _encoder;
    private readonly int _hiddenStateLayer;

    /// <summary>Constructs the wrapper. The encoder's lifetime is owned by the caller — disposing this wrapper does NOT dispose it.</summary>
    /// <param name="encoder">Pre-loaded Qwen2.5-VL-7B text stack (<see cref="LlamaStyleEncoderConfig.Qwen2_5_VL_7B"/>).</param>
    /// <param name="hiddenStateLayer">HuggingFace-indexed hidden state to tap. Default <c>NumLayers - 2</c> = diffusers' <c>hidden_states[-3]</c> (<c>hidden_state_skip_layer=2</c>).</param>
    public HunyuanImageQwenTextEncoder(LlamaStyleEncoder encoder, int? hiddenStateLayer = null)
    {
        _encoder = encoder;
        _hiddenStateLayer = hiddenStateLayer ?? (encoder.NumLayers - 2);
    }

    /// <summary>Encodes one template-wrapped, padded prompt and returns <c>[1, realLen − 34, hidden]</c> F32 conditioning with the template prefix dropped.</summary>
    /// <param name="paddedTokenIds">Token ids padded to <see cref="PaddedLength"/> (pad id 151643).</param>
    /// <param name="attentionMask">1 for real tokens, 0 for pad — same length as <paramref name="paddedTokenIds"/>.</param>
    public Tensor Encode(IBackend backend, int[] paddedTokenIds, int[] attentionMask)
    {
        if (paddedTokenIds is null || paddedTokenIds.Length == 0)
            throw new ArgumentException("Token ids must be non-empty.", nameof(paddedTokenIds));
        if (attentionMask is null || attentionMask.Length != paddedTokenIds.Length)
            throw new ArgumentException(
                $"Attention mask length {attentionMask?.Length ?? 0} != token count {paddedTokenIds.Length}.",
                nameof(attentionMask));

        int realLen = 0;
        for (int i = 0; i < attentionMask.Length; i++)
            if (attentionMask[i] != 0) realLen++;
        if (realLen <= TemplatePrefixTokens)
            throw new ArgumentException(
                $"Real token count {realLen} must exceed the {TemplatePrefixTokens}-token template prefix — " +
                "was the prompt encoded with the HunyuanImage template?", nameof(attentionMask));

        // Causal attention: hidden states of the first realLen tokens don't depend on trailing pad,
        // so encoding the trimmed sequence matches the masked padded forward exactly.
        int[] trimmed = paddedTokenIds[..realLen];
        Tensor full = _encoder.EncodeMultiLayer(backend, [trimmed], [_hiddenStateLayer]);

        int hidden = (int)full.Shape[2];
        int keep = realLen - TemplatePrefixTokens;
        Tensor result = new Tensor(new TensorShape(1, keep, hidden), DType.F32);
        ReadOnlySpan<float> src = full.AsReadOnlySpan<float>();
        src.Slice(TemplatePrefixTokens * hidden, keep * hidden).CopyTo(result.AsSpan<float>());
        full.Dispose();
        return result;
    }

    /// <summary>Forwards to the underlying encoder (for GPU preload paths).</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _encoder.EnumerateWeights();

    /// <summary>No-op — the wrapped encoder is owned by the caller.</summary>
    public void Dispose() { }
}
