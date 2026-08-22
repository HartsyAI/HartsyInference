using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Llama-shaped <see cref="IErnieTextEncoder"/> for ERNIE-Image: Baidu's <c>text_encoder/config.json</c> describes a Mistral3 ("ministral3") decoder — see <see cref="LlamaStyleEncoderConfig.Ministral3B"/> — so this wraps a <see cref="LlamaStyleEncoder"/> and exposes diffusers' <c>output.hidden_states[-2]</c> tap: with N layers, HF <c>hidden_states</c> has N+1 entries, so <c>[-2]</c> is entry N−1 = the output of block N−2 with the final block and final RMSNorm both skipped, which <see cref="LlamaStyleEncoder.EncodeMultiLayer"/> reproduces by taking HF layer index N−1 directly. **Caveat:** matching diffusers' <c>encode_prompt</c> (<c>pipeline_ernie_image.py:124-163</c>), which tokenizes one prompt at a time, every call to <see cref="Encode"/> must receive a <c>tokenIds</c> batch where every row is pre-padded to the same length, with per-row real lengths passed via <c>realLens</c>.</summary>
public sealed class ErnieImageLlamaTextEncoder : IErnieTextEncoder
{
    private readonly LlamaStyleEncoder _encoder;
    private readonly int _hiddenStateLayer;
    private int _capturedHiddenSize;

    /// <summary>Constructs the wrapper.</summary>
    /// <param name="encoder">Pre-loaded LlamaStyleEncoder. Lifetime is owned by the caller — disposing this wrapper does NOT dispose the encoder (matches dotLLM's "shared resources" pattern).</param>
    /// <param name="hiddenStateLayer">Which HuggingFace-indexed hidden state to emit. Default <c>NumLayers - 1</c> = diffusers' <c>hidden_states[-2]</c> (second-to-last entry, before the final block and final norm).</param>
    public ErnieImageLlamaTextEncoder(LlamaStyleEncoder encoder, int? hiddenStateLayer = null)
    {
        _encoder = encoder;
        _hiddenStateLayer = hiddenStateLayer ?? (encoder.NumLayers - 1);
    }

    /// <summary>Hidden dim of the wrapped encoder. <see cref="LlamaStyleEncoder"/> doesn't expose its config, so this is 0 until either <see cref="WithHiddenSize"/> is called at wiring time or the first <see cref="Encode"/> captures it from the output shape.</summary>
    public int OutputDim => _capturedHiddenSize;

    /// <summary>Sets the reported hidden size up front (call once at construction time when wiring up, using the value from your <see cref="LlamaStyleEncoderConfig"/>).</summary>
    public ErnieImageLlamaTextEncoder WithHiddenSize(int hiddenSize)
    {
        _capturedHiddenSize = hiddenSize;
        return this;
    }

    /// <summary>Encodes a batch of pre-tokenized prompts (all rows same length) and returns <c>[B, Tmax, hidden]</c> hidden states from the configured layer slice.</summary>
    public (Tensor HiddenStates, int[] TextLens) Encode(IBackend backend, int[][] tokenIds, int[] realLens)
    {
        if (tokenIds is null || tokenIds.Length == 0)
            throw new ArgumentException("tokenIds must contain at least one prompt.", nameof(tokenIds));
        if (realLens.Length != tokenIds.Length)
            throw new ArgumentException($"realLens length {realLens.Length} != batch {tokenIds.Length}.", nameof(realLens));

        int seqLen = tokenIds[0].Length;
        for (int i = 1; i < tokenIds.Length; i++)
        {
            if (tokenIds[i].Length != seqLen)
                throw new ArgumentException("All prompts must be the same padded length; pad shorter prompts upstream.", nameof(tokenIds));
        }

        Tensor multi = _encoder.EncodeMultiLayer(backend, tokenIds, [_hiddenStateLayer]);
        // EncodeMultiLayer returns [B, S, K*hidden]; with K=1 the third dim IS the hidden size.
        if (_capturedHiddenSize == 0) _capturedHiddenSize = (int)multi.Shape[2];
        return (multi, (int[])realLens.Clone());
    }

    /// <summary>Forwards to the underlying encoder.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _encoder.EnumerateWeights();

    /// <summary>No-op — the wrapped encoder is owned by the caller.</summary>
    public void Dispose() { }
}
