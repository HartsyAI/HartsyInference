using HartsyInference.Core.Exceptions;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>Configuration for Flux DiT ControlNets (diffusers <c>FluxControlNetModel</c>). The adapter is a
/// reduced copy of the base FluxTransformer — its own <c>x_embedder</c> / <c>context_embedder</c> /
/// <c>time_text_embed</c> plus <see cref="Depth"/> double-stream and <see cref="DepthSingleBlocks"/>
/// single-stream blocks — with a zero-init <c>controlnet_x_embedder</c> that injects the packed VAE-encoded
/// control image, and one zero-init output Linear per block producing the residuals the base transformer adds
/// onto its image stream. Union checkpoints (InstantX Union / Shakker Union-Pro) carry an extra
/// <c>controlnet_mode_embedder</c> whose <see cref="NumMode"/> rows select the conditioning type per request;
/// Union-Pro-2.0 removed it (<see cref="NumMode"/> = null) and infers the mode from the control image alone.</summary>
public sealed record FluxControlNetConfig
{
    /// <summary>Transformer hidden dimension (3072 for Flux.1).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads (24 for Flux.1; head dim = HiddenSize / NumHeads).</summary>
    public required int NumHeads { get; init; }

    /// <summary>Number of double-stream blocks (6 for Union-Pro-2.0, 5 for Union/Union-Pro).</summary>
    public required int Depth { get; init; }

    /// <summary>Number of single-stream blocks (0 for Union-Pro-2.0, 10 for Union/Union-Pro).</summary>
    public required int DepthSingleBlocks { get; init; }

    /// <summary>Input channels per packed patch (64 = 16 latent channels × 2×2 packing).</summary>
    public int InChannels { get; init; } = 64;

    /// <summary>T5 text embedding dimension consumed by <c>context_embedder</c> (4096 for T5-XXL).</summary>
    public int ContextInDim { get; init; } = 4096;

    /// <summary>CLIP pooled vector dimension consumed by <c>time_text_embed.text_embedder</c> (768 for CLIP-L).</summary>
    public int PooledProjectionDim { get; init; } = 768;

    /// <summary>Whether the checkpoint carries a guidance embedding MLP (true for FLUX.1-dev ControlNets).</summary>
    public bool GuidanceEmbed { get; init; } = true;

    /// <summary>Union-mode embedding count (<c>controlnet_mode_embedder</c> rows), or null for single-mode /
    /// mode-free checkpoints. InstantX Union uses 7 (canny/tile/depth/blur/pose/gray/lq).</summary>
    public int? NumMode { get; init; }

    /// <summary>RoPE axis dimensions; must sum to the head dim. Default [16, 56, 56] (Flux.1).</summary>
    public int[] AxesDim { get; init; } = [16, 56, 56];

    /// <summary>RoPE base frequency.</summary>
    public int Theta { get; init; } = 10000;

    /// <summary>MLP ratio (4.0 for Flux.1: mlp_dim = hidden × 4).</summary>
    public float MlpRatio { get; init; } = 4.0f;

    /// <summary>Whether Q/K/V projections have bias (true for Flux.1).</summary>
    public bool QkvBias { get; init; } = true;

    /// <summary>QK-norm RMSNorm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>True when this is a union checkpoint requiring a per-request control mode id.</summary>
    public bool IsUnion => NumMode is not null;

    /// <summary>Derives the config from a diffusers-layout Flux ControlNet checkpoint header: block depths from
    /// the <c>transformer_blocks.*</c> / <c>single_transformer_blocks.*</c> key ranges, union mode count from the
    /// <c>controlnet_mode_embedder.weight</c> rows (absent on Union-Pro-2.0), guidance from
    /// <c>time_text_embed.guidance_embedder.*</c> presence, and dims from the embedder weight shapes.</summary>
    public static FluxControlNetConfig FromDescriptors(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        foreach (string key in descriptors.Keys)
        {
            if (key.StartsWith("input_hint_block.", StringComparison.Ordinal))
            {
                throw new UnsupportedModelException(
                    "Flux ControlNet checkpoints with an input_hint_block (pixel-space hint encoder) are not supported; " +
                    "only latent-conditioned checkpoints (controlnet_x_embedder over the packed VAE latent) are.");
            }
        }

        if (!descriptors.TryGetValue("x_embedder.weight", out SafeTensorDescriptor? xEmbed))
            throw new HartsyInferenceException("Flux ControlNet checkpoint is missing 'x_embedder.weight'.");
        if (!descriptors.TryGetValue("controlnet_x_embedder.weight", out SafeTensorDescriptor? cnxEmbed))
            throw new HartsyInferenceException("Flux ControlNet checkpoint is missing 'controlnet_x_embedder.weight'.");

        int hidden = (int)xEmbed.Shape[0];
        int inChannels = (int)xEmbed.Shape[1];
        if (cnxEmbed.Shape[0] != hidden)
        {
            throw new HartsyInferenceException(
                $"Flux ControlNet controlnet_x_embedder rows ({cnxEmbed.Shape[0]}) != x_embedder rows ({hidden}).");
        }

        int depth = 0;
        int depthSingle = 0;
        foreach (string key in descriptors.Keys)
        {
            if (TryParseBlockIndex(key, "transformer_blocks.", out int idx) && idx + 1 > depth)
                depth = idx + 1;
            else if (TryParseBlockIndex(key, "single_transformer_blocks.", out int sIdx) && sIdx + 1 > depthSingle)
                depthSingle = sIdx + 1;
        }
        if (depth == 0)
            throw new HartsyInferenceException("Flux ControlNet checkpoint has no 'transformer_blocks.*' keys.");

        int contextInDim = descriptors.TryGetValue("context_embedder.weight", out SafeTensorDescriptor? ctx)
            ? (int)ctx.Shape[1] : 4096;
        int pooledDim = descriptors.TryGetValue("time_text_embed.text_embedder.linear_1.weight", out SafeTensorDescriptor? pooled)
            ? (int)pooled.Shape[1] : 768;
        bool guidance = descriptors.ContainsKey("time_text_embed.guidance_embedder.linear_1.weight");
        int? numMode = descriptors.TryGetValue("controlnet_mode_embedder.weight", out SafeTensorDescriptor? mode)
            ? (int)mode.Shape[0] : null;
        float mlpRatio = descriptors.TryGetValue("transformer_blocks.0.ff.net.0.proj.weight", out SafeTensorDescriptor? ff)
            ? (float)ff.Shape[0] / hidden : 4.0f;

        // Head dim from the per-head QK-norm weight length (128 for every Flux.1 DiT ControlNet).
        int headDim = descriptors.TryGetValue("transformer_blocks.0.attn.norm_q.weight", out SafeTensorDescriptor? normQ)
            ? (int)normQ.Shape[0] : 128;

        return new FluxControlNetConfig
        {
            HiddenSize = hidden,
            NumHeads = hidden / headDim,
            Depth = depth,
            DepthSingleBlocks = depthSingle,
            InChannels = inChannels,
            ContextInDim = contextInDim,
            PooledProjectionDim = pooledDim,
            GuidanceEmbed = guidance,
            NumMode = numMode,
            MlpRatio = mlpRatio,
        };
    }

    private static bool TryParseBlockIndex(string key, string prefix, out int index)
    {
        index = -1;
        if (!key.StartsWith(prefix, StringComparison.Ordinal))
            return false;
        int dot = key.IndexOf('.', prefix.Length);
        return dot > prefix.Length && int.TryParse(key.AsSpan(prefix.Length, dot - prefix.Length), out index);
    }
}
