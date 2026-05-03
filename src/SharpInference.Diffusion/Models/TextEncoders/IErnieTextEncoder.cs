using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.TextEncoders;

/// <summary>Adapter for ERNIE-Image's text encoder. Baidu's published <c>baidu/ERNIE-Image</c> uses an in-house ERNIE-class encoder loaded via HuggingFace <c>AutoModel</c>; the exact architecture is determined by <c>text_encoder/config.json</c> on the HF repo. This interface lets the pipeline ignore those details and swap in either:
/// <list type="bullet">
///   <item>An <see cref="ErnieImageLlamaTextEncoder"/> fallback when the encoder turns out to be Llama-shaped (Qwen3 / Mistral-style).</item>
///   <item>A custom Baidu encoder once the architecture is confirmed.</item>
///   <item>The <see cref="ErnieImagePlaceholderTextEncoder"/> stub for compile-time wiring without a real encoder.</item>
/// </list>
///
/// Implementations are responsible for whatever pre/post-processing the ERNIE pipeline expects (e.g., diffusers takes <c>output.hidden_states[-2][0]</c> — second-to-last hidden state — as the conditioning vector).</summary>
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
