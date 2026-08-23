using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Adapter for ERNIE-Image's text encoder — currently served by <see cref="ErnieImageLlamaTextEncoder"/> over <see cref="LlamaStyleEncoderConfig.Ministral3B"/> — kept as an interface so an alternative encoder can be swapped in without touching the pipeline; implementations must reproduce diffusers' <c>output.hidden_states[-2][0]</c> tap as the conditioning vector.</summary>
public interface IErnieTextEncoder : IDisposable
{
    /// <summary>Encodes a single prompt's already-tokenized input into the conditioning sequence the transformer's <c>text_proj</c> expects.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="tokenIds">Pre-tokenized prompt token ids (one row per prompt). All rows must be the same length — the caller is responsible for pre-padding if batching multiple prompts of different real lengths.</param>
    /// <param name="realLens">Per-prompt real (non-padded) token counts. Length must equal <c>tokenIds.Length</c>. Used to build <c>text_lens</c> for the transformer's RoPE position-id offsets and the padding-aware attention mask.</param>
    /// <returns>Tensor of shape <c>[B, Tmax, text_in_dim]</c> as F32 (or whatever the transformer's <c>text_proj</c> expects), and the per-batch real lengths echoed back unchanged.</returns>
    (Tensor HiddenStates, int[] TextLens) Encode(IBackend backend, int[][] tokenIds, int[] realLens);

    /// <summary>The output feature dimension (third axis of the returned tensor).</summary>
    int OutputDim { get; }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    IEnumerable<Tensor> EnumerateWeights();
}
