using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Reads the Fibo 2.0 Qwen3-VL text encoder's structure from its checkpoint shapes. The RoPE and norm constants come from
/// <see cref="LlamaStyleEncoderConfig.Qwen3_VL_4B"/>, so a smaller build with the same key layout still resolves.</summary>
public static class Fibo2TextEncoderConfig
{
    /// <summary>Highest HF hidden-state index the Fibo bundles read; the encoder needs at least this many blocks.</summary>
    public static int RequiredLayers => Fibo2TextBundles.TapLayers[^1];

    /// <summary>Builds the encoder configuration from <paramref name="weights"/>, accepting the optional <c>model.</c> wrapper that
    /// <see cref="LlamaStyleEncoder.LoadWeights"/> also accepts.</summary>
    public static LlamaStyleEncoderConfig FromWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        string prefix = weights.ContainsKey("model.embed_tokens.weight") ? "model." : string.Empty;
        Tensor embed = Require(weights, prefix + "embed_tokens.weight");
        Tensor qProj = Require(weights, prefix + "layers.0.self_attn.q_proj.weight");
        Tensor kProj = Require(weights, prefix + "layers.0.self_attn.k_proj.weight");
        Tensor qNorm = Require(weights, prefix + "layers.0.self_attn.q_norm.weight");
        Tensor gate = Require(weights, prefix + "layers.0.mlp.gate_proj.weight");

        int headDim = (int)qNorm.Shape[0];
        if (headDim <= 0 || qProj.Shape[0] % headDim != 0 || kProj.Shape[0] % headDim != 0)
            throw new InvalidDataException($"Fibo 2.0 text encoder head dim {headDim} does not divide its query/key projections.");

        return LlamaStyleEncoderConfig.Qwen3_VL_4B with
        {
            VocabSize = (int)embed.Shape[0],
            HiddenSize = (int)embed.Shape[1],
            NumLayers = CountLayers(weights, prefix),
            NumQueryHeads = (int)(qProj.Shape[0] / headDim),
            NumKvHeads = (int)(kProj.Shape[0] / headDim),
            HeadDim = headDim,
            IntermediateSize = (int)gate.Shape[0],
        };
    }

    /// <summary>Checks the text encoder against the denoiser: the caption bundles are three taps wide, so the cap feature width must be
    /// three times the encoder's hidden size, and the encoder must reach the highest tap.</summary>
    /// <exception cref="InvalidDataException">The encoder and denoiser disagree.</exception>
    public static void ValidateFor(LlamaStyleEncoderConfig textEncoder, Fibo2Config denoiser)
    {
        ArgumentNullException.ThrowIfNull(textEncoder);
        ArgumentNullException.ThrowIfNull(denoiser);
        if (denoiser.CapFeatDim != 3 * textEncoder.HiddenSize)
        {
            throw new InvalidDataException(
                $"Fibo 2.0 cap_feat_dim is {denoiser.CapFeatDim}, but three Qwen3-VL taps of hidden {textEncoder.HiddenSize} make "
                + $"{3 * textEncoder.HiddenSize}. The text encoder does not match this denoiser.");
        }
        if (textEncoder.NumLayers < RequiredLayers)
        {
            throw new InvalidDataException(
                $"Fibo 2.0 reads hidden state {RequiredLayers}, but the text encoder has {textEncoder.NumLayers} layers.");
        }
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key) =>
        weights.TryGetValue(key, out Tensor? tensor)
            ? tensor
            : throw new KeyNotFoundException($"Fibo 2.0 text encoder is missing '{key}'.");

    private static int CountLayers(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        int count = 0;
        while (weights.ContainsKey($"{prefix}layers.{count}.self_attn.q_proj.weight"))
            count++;
        if (count == 0)
            throw new KeyNotFoundException($"Fibo 2.0 text encoder has no '{prefix}layers.0' blocks.");
        return count;
    }
}
