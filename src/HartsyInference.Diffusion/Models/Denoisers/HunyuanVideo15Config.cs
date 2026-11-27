namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Shape of the HunyuanVideo-1.5 DiT (<c>HunyuanVideo_1_5_DiffusionTransformer</c>): double-stream blocks only, 1x1x1 patches over a
/// 32-channel 16x-spatial 4x-temporal latent, Qwen2.5-VL text (refined) plus a ByT5 glyph stream. MovieRoll's streaming DiT has the same
/// shape (<c>movieroll/models/transformers/model_configs.py</c>), so one config serves both.
/// Sources: <c>tencent/HunyuanVideo-1.5</c> <c>transformer/480p_t2v/config.json</c> (rev 9b49404b) and MovieRoll @0b5f73e.</summary>
public sealed record HunyuanVideo15Config
{
    /// <summary>Double-stream blocks (54; there are no single-stream blocks).</summary>
    public int NumBlocks { get; init; } = 54;

    /// <summary>Hidden width (2048).</summary>
    public int HiddenSize { get; init; } = 2048;

    /// <summary>Attention heads (16, head dim 128).</summary>
    public int NumHeads { get; init; } = 16;

    /// <summary>MLP width ratio (4, GELU tanh).</summary>
    public float MlpRatio { get; init; } = 4f;

    /// <summary>Latent channels in and out (32).</summary>
    public int LatentChannels { get; init; } = 32;

    /// <summary><c>img_in</c> input channels: latent, conditioning copy and a mask, <c>2 * LatentChannels + 1</c> (65). The conditioning
    /// block and the mask are zeros for text-to-video; MovieRoll feeds zeros for image-to-video as well.</summary>
    public int ImgInChannels => 2 * LatentChannels + 1;

    /// <summary>Qwen2.5-VL hidden size feeding the token refiner (3584).</summary>
    public int TextDim { get; init; } = 3584;

    /// <summary>ByT5-small hidden size feeding <c>byt5_in</c> (1472).</summary>
    public int ByT5Dim { get; init; } = 1472;

    /// <summary>Per-axis RoPE split (frame, height, width); sums to the head dim (16 + 56 + 56 = 128).</summary>
    public int[] RopeAxesDim { get; init; } = [16, 56, 56];

    /// <summary>RoPE base frequency (256).</summary>
    public float RopeTheta { get; init; } = 256f;

    /// <summary>Head dimension.</summary>
    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>Feed-forward inner width.</summary>
    public int MlpDim => (int)(HiddenSize * MlpRatio);

    /// <summary>The published 8.3B shape (HunyuanVideo-1.5 480p/720p and MovieRoll).</summary>
    public static HunyuanVideo15Config Full => new();
}
