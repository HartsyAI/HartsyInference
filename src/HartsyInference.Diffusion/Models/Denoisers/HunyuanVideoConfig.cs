namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Config for the HunyuanVideo MM-DiT (the base offline-video transformer; also the backbone the
/// Hunyuan-GameCraft world model finetunes). Dual-stream + single-stream blocks with 3-axis (T,H,W) RoPE,
/// reusing <see cref="DiTBlocks.HunyuanImageBlock"/> / <see cref="DiTBlocks.HunyuanImageSingleBlock"/>.
/// <para>The GameCraft presets carry the action-conditioned input width (33 channels = noisy 16 + history 16 +
/// mask 1). All dims are <b>validation-gated</b> against the real checkpoint.</para></summary>
public sealed record HunyuanVideoConfig
{
    /// <summary>Transformer hidden width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads (head_dim = HiddenSize / NumHeads).</summary>
    public required int NumHeads { get; init; }

    /// <summary>Dual-stream block count.</summary>
    public required int NumDoubleBlocks { get; init; }

    /// <summary>Single-stream block count.</summary>
    public required int NumSingleBlocks { get; init; }

    /// <summary>FFN intermediate size.</summary>
    public required int MlpDim { get; init; }

    /// <summary>Patchify patch size (temporal, height, width).</summary>
    public (int T, int H, int W) PatchSize { get; init; } = (1, 2, 2);

    /// <summary>Patchify input channels (16 for plain HunyuanVideo; 33 for GameCraft's noisy+history+mask).</summary>
    public required int InChannels { get; init; }

    /// <summary>VAE latent channels predicted by the final layer (the velocity is over the noisy 16-ch latent).</summary>
    public int OutChannels { get; init; } = 16;

    /// <summary>3-axis RoPE dim split over (T, H, W); sums to head_dim.</summary>
    public int[] RopeAxesDim { get; init; } = [16, 56, 56];

    /// <summary>RoPE base frequency.</summary>
    public float RopeTheta { get; init; } = 256.0f;

    /// <summary>Primary text-encoder hidden (Llava-Llama-3-8B = 4096) projected to <see cref="HiddenSize"/>.</summary>
    public int TextEmbedDim { get; init; } = 4096;

    /// <summary>Pooled CLIP-L dim (768) feeding the global modulation vector.</summary>
    public int PooledEmbedDim { get; init; } = 768;

    /// <summary>Head dim (derived).</summary>
    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>Hunyuan-GameCraft base (50-step, CFG 2.0).</summary>
    public static HunyuanVideoConfig GameCraftBase => new()
    {
        HiddenSize = 3072, NumHeads = 24, NumDoubleBlocks = 19, NumSingleBlocks = 38,
        MlpDim = 12288, InChannels = 33, OutChannels = 16, RopeAxesDim = [16, 56, 56],
    };

    /// <summary>Hunyuan-GameCraft distilled (PCM, 8-step, CFG 1.0) — same architecture, different weights.</summary>
    public static HunyuanVideoConfig GameCraftDistilled => GameCraftBase;
}
