using HartsyInference.Core.Exceptions;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>Configuration for a Qwen-Image DiT ControlNet (diffusers <c>QwenImageControlNetModel</c>, the
/// InstantX architecture), auto-derived from the checkpoint header. The block stack reuses the base
/// transformer's <c>QwenImageBlock</c> geometry, so most fields mirror <c>QwenImageConfig</c>.</summary>
public sealed record QwenImageControlNetConfig
{
    /// <summary>Transformer hidden size (3072 for the released checkpoints).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention head count.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Number of ControlNet transformer blocks (5 for ControlNet-Union) — also the residual count.</summary>
    public required int Depth { get; init; }

    /// <summary>FFN inner dimension.</summary>
    public required int MlpDim { get; init; }

    /// <summary>Packed input width of <c>img_in</c>/<c>controlnet_x_embedder</c> (patch² · latent channels = 64).</summary>
    public required int InDim { get; init; }

    /// <summary>Text-encoder hidden size consumed by <c>txt_in</c> (3584 for Qwen2.5-VL).</summary>
    public required int ContextDim { get; init; }

    /// <summary>RoPE theta — matches the base transformer.</summary>
    public int RopeTheta { get; init; } = 10000;

    /// <summary>Whether text positions get RoPE (base-transformer parity).</summary>
    public bool RopeText { get; init; } = true;

    /// <summary>QK-norm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Derives the config from a checkpoint's tensor descriptors. Throws when the header does not
    /// look like a Qwen-Image ControlNet.</summary>
    public static QwenImageControlNetConfig FromDescriptors(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        if (!descriptors.TryGetValue("img_in.weight", out SafeTensorDescriptor? imgIn))
            throw new HartsyInferenceException("Qwen-Image ControlNet checkpoint is missing 'img_in.weight'.");
        if (!descriptors.TryGetValue("controlnet_x_embedder.weight", out SafeTensorDescriptor? cnxEmbed))
            throw new HartsyInferenceException("Qwen-Image ControlNet checkpoint is missing 'controlnet_x_embedder.weight'.");
        int hidden = (int)imgIn.Shape[0];
        int inDim = (int)imgIn.Shape[1];
        if (cnxEmbed.Shape[0] != hidden)
        {
            throw new HartsyInferenceException(
                $"Qwen-Image ControlNet controlnet_x_embedder rows ({cnxEmbed.Shape[0]}) != img_in rows ({hidden}).");
        }

        int depth = 0;
        foreach (string key in descriptors.Keys)
        {
            if (!key.StartsWith("transformer_blocks.", StringComparison.Ordinal))
                continue;
            int end = key.IndexOf('.', "transformer_blocks.".Length);
            if (end > 0 && int.TryParse(key.AsSpan("transformer_blocks.".Length, end - "transformer_blocks.".Length), out int idx)
                && idx + 1 > depth)
            {
                depth = idx + 1;
            }
        }
        if (depth == 0)
            throw new HartsyInferenceException("Qwen-Image ControlNet checkpoint has no 'transformer_blocks.*' keys.");

        int headDim = descriptors.TryGetValue("transformer_blocks.0.attn.norm_q.weight", out SafeTensorDescriptor? normQ)
            ? (int)normQ.Shape[0] : 128;
        int contextDim = descriptors.TryGetValue("txt_in.weight", out SafeTensorDescriptor? txtIn)
            ? (int)txtIn.Shape[1] : 3584;
        int mlpDim = descriptors.TryGetValue("transformer_blocks.0.img_mlp.net.0.proj.weight", out SafeTensorDescriptor? ff)
            ? (int)ff.Shape[0] : hidden * 4;

        return new QwenImageControlNetConfig
        {
            HiddenSize = hidden,
            NumHeads = hidden / headDim,
            HeadDim = headDim,
            Depth = depth,
            MlpDim = mlpDim,
            InDim = inDim,
            ContextDim = contextDim,
        };
    }
}
