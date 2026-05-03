using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.TextEncoders;

/// <summary>Llama-shaped <see cref="IErnieTextEncoder"/> fallback. Wraps an existing <see cref="LlamaStyleEncoder"/> and exposes the second-to-last hidden state — matching the diffusers reference's <c>output.hidden_states[-2]</c> convention. Use this if (and only if) Baidu's <c>text_encoder/config.json</c> turns out to describe a Llama-class architecture (Qwen3-shaped or Mistral-shaped); otherwise build a dedicated encoder that implements <see cref="IErnieTextEncoder"/>.
///
/// **Caveat:** the diffusers <c>encode_prompt</c> implementation (<c>pipeline_ernie_image.py:124-163</c>) tokenizes one prompt at a time and squeezes out the batch dimension. We follow the same pattern: each call to <see cref="Encode"/> must receive a <c>tokenIds</c> batch where every row has the same length — pre-pad shorter prompts to the longest in the batch and provide the per-row real lengths via <c>realLens</c>. The encoder runs once over the padded batch and emits <c>[B, Tmax, hidden]</c>.</summary>
public sealed class ErnieImageLlamaTextEncoder : IErnieTextEncoder
{
    private readonly LlamaStyleEncoder _encoder;
    private readonly int _hiddenStateLayer;

    /// <summary>Constructs the wrapper.</summary>
    /// <param name="encoder">Pre-loaded LlamaStyleEncoder. Lifetime is owned by the caller — disposing this wrapper does NOT dispose the encoder (matches dotLLM's "shared resources" pattern).</param>
    /// <param name="hiddenStateLayer">Which HuggingFace-indexed hidden state to emit. Default <c>NumLayers - 1</c> = second-to-last (matches diffusers' <c>hidden_states[-2]</c>).</param>
    public ErnieImageLlamaTextEncoder(LlamaStyleEncoder encoder, int? hiddenStateLayer = null)
    {
        _encoder = encoder;
        _hiddenStateLayer = hiddenStateLayer ?? (encoder.NumLayers - 1);
    }

    /// <summary>Hidden dim of the wrapped encoder.</summary>
    public int OutputDim
    {
        get
        {
            // We ask for one layer slice; its dim is the encoder's hidden size.
            // LlamaStyleEncoder doesn't expose the hidden size directly, but EncodeMultiLayer
            // returns [B, S, K*hidden] for K=1, so the third dim of a single-token shape probe
            // would tell us — at construction we don't have a backend, so we surface it via the
            // encoder.LastConfigured (not present in LlamaStyleEncoder API). Use a reflection-free
            // alternative: the caller knows the dim from LlamaStyleEncoderConfig.
            // We delegate to the dump-friendly public NumLayers; for OutputDim we expect callers
            // to also know the hidden size from their LlamaStyleEncoderConfig.
            // To make this self-contained, we capture the hidden size via the first Encode call.
            return _capturedHiddenSize;
        }
    }

    private int _capturedHiddenSize;

    /// <summary>Sets the captured hidden size manually (call once at construction time when wiring up).</summary>
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
