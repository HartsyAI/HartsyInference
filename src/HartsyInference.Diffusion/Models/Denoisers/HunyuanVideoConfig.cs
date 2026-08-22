namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Config for the HunyuanVideo MM-DiT (the base offline-video transformer; also the backbone the Hunyuan-GameCraft world model finetunes). Dual-stream + single-stream blocks with 3-axis (T,H,W) RoPE, reusing <see cref="DiTBlocks.HunyuanImageBlock"/> / <see cref="DiTBlocks.HunyuanImageSingleBlock"/>. <para>The GameCraft presets carry the action-conditioned input width (33 channels = noisy 16 + history 16 + mask 1). All dims are <b>validation-gated</b> against the real checkpoint.</para></summary>
public sealed record HunyuanVideoConfig
{
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

    /// <summary>Whether the model consumes a separate guidance-scale embedding (HunyuanVideo distill-guidance path). Plain HunyuanVideo sets <c>guidance_embeds=True</c> and feeds <see cref="EmbeddedGuidanceScale"/> through a <c>guidance_in</c> MLP into the modulation vector. GameCraft sets <c>guidance_embed=False</c> and relies on classifier-free guidance instead.</summary>
    public bool GuidanceEmbed { get; init; } = false;

    /// <summary>Embedded guidance scalar fed through the <c>guidance_in</c> MLP when <see cref="GuidanceEmbed"/> is true (plain HunyuanVideo). Diffusers default is <b>6.0</b>. Ignored when guidance is not embedded.</summary>
    public float EmbeddedGuidanceScale { get; init; } = 6.0f;

    /// <summary>Flow-match sampling shift (<c>--flow-shift-eval-video</c>). 5.0 for both GameCraft variants.</summary>
    public float FlowShift { get; init; } = 5.0f;

    /// <summary>Default sampler steps (<c>--infer-steps</c>): 50 base, 8 distilled.</summary>
    public int InferSteps { get; init; } = 50;

    /// <summary>Default classifier-free guidance scale (<c>--cfg-scale</c>): 2.0 base, 1.0 distilled.</summary>
    public float CfgScale { get; init; } = 2.0f;

    /// <summary>Head dim (derived).</summary>
    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>Plain HunyuanVideo T2V preset (HYVideo-T/2: 20 double + 40 single blocks, 13B). 16-channel input (no action conditioning), embedded guidance (<c>guidance_embeds=True</c>, scale 6.0), flow shift 7.0, 50 steps. Matches <c>hunyuanvideo-community/HunyuanVideo</c>.</summary>
    public static HunyuanVideoConfig T2V => new()
    {
        HiddenSize = 3072, NumHeads = 24, NumDoubleBlocks = 20, NumSingleBlocks = 40,
        MlpDim = 12288, InChannels = 16, OutChannels = 16, RopeAxesDim = [16, 56, 56],
        GuidanceEmbed = true, EmbeddedGuidanceScale = 6.0f, FlowShift = 7.0f, InferSteps = 50, CfgScale = 1.0f,
    };

    /// <summary>Hunyuan-GameCraft base (HYVideo-T/2: 20 double + 40 single blocks; 50-step, CFG 2.0).</summary>
    public static HunyuanVideoConfig GameCraftBase => new()
    {
        HiddenSize = 3072, NumHeads = 24, NumDoubleBlocks = 20, NumSingleBlocks = 40,
        MlpDim = 12288, InChannels = 33, OutChannels = 16, RopeAxesDim = [16, 56, 56],
        GuidanceEmbed = false, FlowShift = 5.0f, InferSteps = 50, CfgScale = 2.0f,
    };

    /// <summary>Hunyuan-GameCraft distilled (PCM): identical architecture to <see cref="GameCraftBase"/>, different weights and sampler settings (8-step, CFG 1.0).</summary>
    public static HunyuanVideoConfig GameCraftDistilled => GameCraftBase with { InferSteps = 8, CfgScale = 1.0f };
}
