using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Zeta-Chroma (<c>lodestones/Zeta-Chroma</c> pixel-proto checkpoints). Despite the
/// name, this is NOT a Chroma MMDiT — it is the Z-Image S3-DiT (NextDiT, 30 single-stream layers, hidden 3840,
/// Qwen3-4B captions) retrained for <b>x0 prediction directly in pixel space</b>: <c>x_embedder</c> consumes raw
/// 32×32 RGB patches (in-dim 3072 = 3·32²) and a DeCo-style <c>dec_net</c> decoder head replaces
/// <c>final_layer</c>. The model is mid-pretraining — every value here is validation-gated (no checkpoint diffed
/// locally yet). See docs/Research/CHROMA_RADIANCE_ARCHITECTURE.md §Zeta-Chroma.</summary>
public sealed record ZetaChromaConfig
{
    /// <summary>Z-Image backbone configuration with pixel-space IO: <c>InChannels = 3</c> and
    /// <c>PatchSize</c> = the pixel patch (32 reported).</summary>
    public required ZImageConfig Backbone { get; init; }

    /// <summary>Pixel patch size, inferred from the <c>x_embedder.weight</c> in-dim: <c>round(sqrt(in/3))</c>
    /// (reported 3072 → 32).</summary>
    public int PatchSize { get; init; } = 32;

    /// <summary>Decoder head hidden width (<c>dec_net.cond_embed.weight</c> out-dim; default 3840).</summary>
    public int DecoderHidden { get; init; } = 3840;

    /// <summary>Number of AdaLN residual blocks in the decoder head (4 reported).</summary>
    public int DecoderResBlocks { get; init; } = 4;

    /// <summary>DCT frequency count per axis in the decoder's NerfEmbedder (8 → 64 constant features, since the
    /// decoder treats the whole flattened patch as ONE token: patch_size 1 → all cosines collapse to 1/(1+u·v)).</summary>
    public int DecoderMaxFreqs { get; init; } = 8;

    /// <summary>Default CFG scale (5.0 — validation-gated, mid-pretraining recommendation).</summary>
    public float DefaultCfgScale { get; init; } = 5.0f;

    /// <summary>Default inference steps (50).</summary>
    public int DefaultSteps { get; init; } = 50;

    /// <summary>Flow-match Euler static shift. ComfyUI's ZImagePixelSpace inherits Z-Image's shift 3.0
    /// (validation-gated, research doc item 9).</summary>
    public float SchedulerShift { get; init; } = 3.0f;

    /// <summary>True when a Z-Image-family weight dict is a Zeta-Chroma pixel checkpoint (decoder head present).</summary>
    public static bool IsZetaChroma(IReadOnlyDictionary<string, Tensor> weights)
    {
        foreach (string key in weights.Keys)
        {
            if (key.StartsWith("dec_net.", StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    /// <summary>Builds a config from converted weights, inferring the pixel patch size and the decoder head dims
    /// from tensor shapes. Backbone depth/FFN come from <see cref="ZImageConfig.FromWeights"/>.</summary>
    public static ZetaChromaConfig FromWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        int patchSize = 32;
        int hidden = 3840;
        if (weights.TryGetValue("x_embedder.weight", out Tensor? xEmbed) && xEmbed.Shape.Rank == 2)
        {
            hidden = (int)xEmbed.Shape[0];
            int inDim = (int)xEmbed.Shape[1];
            int p = (int)Math.Round(Math.Sqrt(inDim / 3.0));
            if (p > 0 && p * p * 3 == inDim)
                patchSize = p;
        }

        int decoderHidden = 3840;
        if (weights.TryGetValue("dec_net.cond_embed.weight", out Tensor? condEmbed))
            decoderHidden = (int)condEmbed.Shape[0];

        int resBlocks = 0;
        while (weights.ContainsKey($"dec_net.res_blocks.{resBlocks}.mlp.0.weight"))
            resBlocks++;
        if (resBlocks == 0) resBlocks = 4;

        int maxFreqs = 8;
        if (weights.TryGetValue("dec_net.input_embedder.embedder.0.weight", out Tensor? embedW))
        {
            // in_dim = patch²·3 (whole flattened patch as one token) + maxFreqs².
            int posFeatures = (int)embedW.Shape[1] - patchSize * patchSize * 3;
            int freqs = (int)Math.Round(Math.Sqrt(posFeatures));
            if (freqs > 0 && freqs * freqs == posFeatures)
                maxFreqs = freqs;
        }

        ZImageConfig backbone = ZImageConfig.FromWeights(weights) with
        {
            HiddenSize = hidden,
            NumHeads = hidden / 128,
            InChannels = 3,
            PatchSize = patchSize,
        };

        return new ZetaChromaConfig
        {
            Backbone = backbone,
            PatchSize = patchSize,
            DecoderHidden = decoderHidden,
            DecoderResBlocks = resBlocks,
            DecoderMaxFreqs = maxFreqs,
        };
    }
}
