using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Video.Tokenizers;

/// <summary>A discrete video tokenizer: maps RGB video ↔ a sequence of integer codebook tokens. The reusable
/// substrate for autoregressive video models (Cosmos-Predict1 V2W, and the Phase-10 action-conditioned world
/// models that share the DV8x16x16 token space). Encode is the conditioning path (tokenize the input frames);
/// decode renders sampled tokens back to pixels (the DV-decoder-only "fast" path, no diffusion decoder).</summary>
public interface IDiscreteVideoTokenizer
{
    /// <summary>Number of distinct token ids (= product of the FSQ levels; 64,000 for DV8x16x16).</summary>
    int VocabSize { get; }

    /// <summary>Latent grid <c>(T, H, W)</c> for a pixel clip of <paramref name="frames"/>×<paramref name="height"/>
    /// ×<paramref name="width"/> given this tokenizer's compression ratio. Token count = <c>T·H·W</c>.</summary>
    (int T, int H, int W) LatentGrid(int frames, int height, int width);

    /// <summary>Tokenizes RGB video <paramref name="video"/> <c>[B, 3, T, H, W]</c> (normalized to <c>[-1, 1]</c>)
    /// into flattened integer codes <c>[B, T_lat·H_lat·W_lat]</c> Int32, row-major in <c>(t, h, w)</c> order.</summary>
    Tensor Encode(IBackend backend, Tensor video);

    /// <summary>Renders flattened integer codes <paramref name="codes"/> <c>[B, T_lat·H_lat·W_lat]</c> back to RGB
    /// video <c>[B, 3, T, H, W]</c> in <c>[-1, 1]</c>, given the latent grid dimensions.</summary>
    Tensor Decode(IBackend backend, Tensor codes, int latentT, int latentH, int latentW);
}
