namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Qwen-Image 2.1 (Alibaba, ~7B single-stream DiT). Despite the version number this shares
/// no block structure with <see cref="QwenImageConfig"/>: 2.1 is a <b>single-stream</b> transformer over the
/// concatenated <c>[text, image]</c> sequence (no <c>add_q_proj</c>/<c>txt_mlp</c> dual stream), every block reads
/// <b>one shared modulation</b> rather than its own, the MLP is a fused-<c>gate_up</c> SwiGLU instead of GELU, and the
/// latent is 64-channel at patch 1 (the VAE does all 16× of the spatial reduction). Mirrors ComfyUI's
/// <c>comfy/ldm/qwen_image21/model.py</c> (<c>QwenImage21Transformer2DModel</c>).</summary>
public sealed record QwenImage21Config
{
    /// <summary>Model dimension; equals <see cref="NumHeads"/> × <see cref="HeadDim"/>.</summary>
    public required int HiddenSize { get; init; }

    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension. Must equal the sum of <see cref="AxesDim"/> — RoPE covers the whole head.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Number of single-stream transformer blocks.</summary>
    public required int Depth { get; init; }

    /// <summary>Latent channels in and out. The VAE is 16×-spatial / 64-channel and the DiT does not patchify, so one
    /// latent cell is exactly one token.</summary>
    public int InChannels { get; init; } = 64;

    public int OutChannels { get; init; } = 64;

    /// <summary>Text-encoder hidden size feeding <c>txt_in</c> — Qwen3-VL-8B's 4096.</summary>
    public int ContextDim { get; init; } = 4096;

    /// <summary>SwiGLU inner width as a multiple of <see cref="HiddenSize"/>. The checkpoint stores gate and up fused
    /// as one <c>[2·ratio·hidden, hidden]</c> matrix, so the ratio is read back as <c>gate_up.Shape[0] / 2 / hidden</c>.</summary>
    public int MlpRatio { get; init; } = 3;

    /// <summary>Per-axis RoPE split <c>(sequence, height, width)</c>; sums to <see cref="HeadDim"/>.</summary>
    public int[] AxesDim { get; init; } = [16, 56, 56];

    /// <summary>RoPE base frequency.</summary>
    public int RopeTheta { get; init; } = 10000;

    /// <summary>Epsilon shared by the QK RMSNorms, the zero-centered text norm and the modulated LayerNorms.</summary>
    public float Eps { get; init; } = 1e-6f;

    /// <summary>The released <c>Comfy-Org/Qwen-Image-2.1</c> checkpoint: 32 blocks, hidden 4096, 32 heads of 128,
    /// SwiGLU inner 12288, 64-channel latent. 265 tensors, all bias-free.</summary>
    public static QwenImage21Config V21 => new()
    {
        HiddenSize = 4096,
        NumHeads = 32,
        Depth = 32,
    };
}
