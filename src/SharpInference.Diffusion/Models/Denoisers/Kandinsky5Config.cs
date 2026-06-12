namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Kandinsky 5.0 text-to-image transformer (<c>Kandinsky5Transformer3DModel</c>,
/// kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers, ~6B params).
///
/// The "3D" naming comes from the joint video+image variant — for the image variant duration is
/// always 1 and <c>patch_size[0] == 1</c> collapses the temporal axis. The architecture has:
/// <list type="bullet">
/// <item>Two encoder blocks for the text stream (<c>num_text_blocks=2</c>): self-attn + FFN with 6
/// modulation params (shift/scale/gate × 2).</item>
/// <item>Fifty decoder blocks for the visual stream (<c>num_visual_blocks=50</c>): self-attn +
/// cross-attn-to-text + FFN with 9 modulation params (shift/scale/gate × 3).</item>
/// <item>All sub-block <c>LayerNorm</c>s are non-affine (<c>elementwise_affine=False</c>).</item>
/// <item>QK norm is <c>RMSNorm</c> with learnable scale per <c>head_dim</c>.</item>
/// <item>Attention QKV/output linears all have biases.</item>
/// <item>FFN is <c>Linear → GELU → Linear</c> with no biases — NOT SwiGLU.</item>
/// <item>3D RoPE for visual tokens with axes <c>[32, 48, 48]</c> (sums to <c>head_dim=128</c>).</item>
/// <item>1D RoPE for text tokens (text_rope_pos is a packed cumulative-sequence-lengths tensor, see
/// <see cref="Kandinsky5Pipeline"/> for the encoding scheme).</item>
/// <item>Dual text encoders: Qwen2.5-VL (3584-dim sequence embeddings) plus CLIP-L (768-dim pooled).
/// The pooled CLIP embedding is projected to <c>time_dim</c> and added to the timestep embedding.</item>
/// <item>Flux VAE (16-channel latent, 8× downsample).</item>
/// <item>Flow-matching with <c>FlowMatchEulerDiscreteScheduler(shift=5.0)</c>.</item>
/// </list>
/// </summary>
public sealed record Kandinsky5Config
{
    /// <summary>Visual latent channels (16 for Flux VAE).</summary>
    public required int InVisualDim { get; init; }

    /// <summary>Output visual channels (typically equal to <see cref="InVisualDim"/>).</summary>
    public required int OutVisualDim { get; init; }

    /// <summary>Timestep embedding hidden dim. The pooled CLIP embedding is also projected to this dim and added.</summary>
    public required int TimeDim { get; init; }

    /// <summary>Patch size as <c>(temporal, height, width)</c>. For image t2i this is <c>(1, 2, 2)</c> so duration=1 collapses.</summary>
    public required (int T, int H, int W) PatchSize { get; init; }

    /// <summary>Transformer hidden size (= num_heads × head_dim).</summary>
    public required int ModelDim { get; init; }

    /// <summary>FFN inner dimension. <c>Linear(model_dim → ff_dim, bias=False) → GELU → Linear(ff_dim → model_dim, bias=False)</c>.</summary>
    public required int FfDim { get; init; }

    /// <summary>Number of encoder blocks for the text stream.</summary>
    public required int NumTextBlocks { get; init; }

    /// <summary>Number of decoder blocks for the visual stream (each runs self-attn + cross-attn-to-text + FFN).</summary>
    public required int NumVisualBlocks { get; init; }

    /// <summary>Per-axis RoPE dims for the visual stream. Must sum to <see cref="HeadDim"/>. Default <c>[32, 48, 48]</c>.</summary>
    public required int[] AxesDims { get; init; }

    /// <summary>Qwen2.5-VL sequence embedding dim (3584 for Qwen2.5-VL-7B-Instruct).</summary>
    public required int InTextDim { get; init; }

    /// <summary>CLIP pooled embedding dim (768 for CLIP-L).</summary>
    public required int InTextDim2 { get; init; }

    /// <summary>Whether the model accepts a visual conditioning input. <c>false</c> for the pure t2i variant;
    /// <c>true</c> for the T2V/I2V video checkpoints, where the model input is
    /// <c>concat([noisy(16), cond(16), mask(1)]) = 33</c> channels even in pure T2V mode (cond+mask zeros).</summary>
    public bool VisualCond { get; init; } = false;

    /// <summary>Total model-input channels: <c>2·InVisualDim + 1</c> with <see cref="VisualCond"/>, else <see cref="InVisualDim"/>.</summary>
    public int VisualEmbedDim => VisualCond ? 2 * InVisualDim + 1 : InVisualDim;

    /// <summary>Per-head attention dim — derived as <c>sum(AxesDims)</c>. For Lite this is 128.</summary>
    public int HeadDim
    {
        get
        {
            int sum = 0;
            for (int i = 0; i < AxesDims.Length; i++) sum += AxesDims[i];
            return sum;
        }
    }

    /// <summary>Number of attention heads — derived as <c>ModelDim / HeadDim</c>. For Lite this is 2560 / 128 = 20.</summary>
    public int NumHeads => ModelDim / HeadDim;

    /// <summary>1D RoPE max position used for text tokens. Matches diffusers <c>Kandinsky5RoPE1D(max_pos=1024)</c>.</summary>
    public int TextRopeMaxPos { get; init; } = 1024;

    /// <summary>3D RoPE max positions per axis. Matches diffusers <c>Kandinsky5RoPE3D(max_pos=(128,128,128))</c>.</summary>
    public (int T, int H, int W) VisualRopeMaxPos { get; init; } = (128, 128, 128);

    /// <summary>RoPE base period. Matches diffusers default 10000.</summary>
    public float RopeMaxPeriod { get; init; } = 10000.0f;

    /// <summary>QK-norm epsilon (RMSNorm default 1e-5).</summary>
    public float QkNormEps { get; init; } = 1e-5f;

    /// <summary>Lite preset (kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers; ~6B params).
    /// Source: <c>transformer/config.json</c> — model_dim=2560, ff_dim=10240, 2 text blocks + 50 visual blocks,
    /// axes_dims=[32,48,48] (head_dim=128, num_heads=20), in_text_dim=3584 (Qwen2.5-VL), in_text_dim2=768 (CLIP-L),
    /// time_dim=512, patch_size=(1,2,2), 16-channel Flux VAE.</summary>
    public static Kandinsky5Config Lite => new()
    {
        InVisualDim = 16,
        OutVisualDim = 16,
        TimeDim = 512,
        PatchSize = (1, 2, 2),
        ModelDim = 2560,
        FfDim = 10240,
        NumTextBlocks = 2,
        NumVisualBlocks = 50,
        AxesDims = [32, 48, 48],
        InTextDim = 3584,
        InTextDim2 = 768,
        VisualCond = false,
    };

    /// <summary>T2V Lite 2B preset (kandinskylab/Kandinsky-5.0-T2V-Lite-*-Diffusers).
    /// Source: <c>transformer/config.json</c> — model_dim=1792, ff_dim=7168, 2 text + 32 visual blocks,
    /// axes_dims=[16,24,24] (head_dim=64, 28 heads), time_dim=512, visual_cond=true (33-channel input),
    /// HunyuanVideo VAE (16-ch latent, 8× spatial / 4× temporal). RoPE scale factors are resolution-dependent
    /// and supplied per-call by the pipeline (see <c>Kandinsky5VideoPipeline.GetRopeScaleFactor</c>).</summary>
    public static Kandinsky5Config VideoLite2B => new()
    {
        InVisualDim = 16,
        OutVisualDim = 16,
        TimeDim = 512,
        PatchSize = (1, 2, 2),
        ModelDim = 1792,
        FfDim = 7168,
        NumTextBlocks = 2,
        NumVisualBlocks = 32,
        AxesDims = [16, 24, 24],
        InTextDim = 3584,
        InTextDim2 = 768,
        VisualCond = true,
    };

    /// <summary>T2V Pro 19B preset — config-only support (weights untested in this engine):
    /// model_dim=4096, ff_dim=16384, 4 text + 60 visual blocks, axes_dims=[32,48,48], time_dim=1024.</summary>
    public static Kandinsky5Config VideoPro19B => new()
    {
        InVisualDim = 16,
        OutVisualDim = 16,
        TimeDim = 1024,
        PatchSize = (1, 2, 2),
        ModelDim = 4096,
        FfDim = 16384,
        NumTextBlocks = 4,
        NumVisualBlocks = 60,
        AxesDims = [32, 48, 48],
        InTextDim = 3584,
        InTextDim2 = 768,
        VisualCond = true,
    };
}
