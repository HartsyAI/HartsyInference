namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Configuration for a Wav2Vec2 audio feature extractor (the frozen encoder Wan2.2-S2V runs over the speech
/// waveform). A 7-layer strided Conv1d feature encoder → feature projection → grouped positional conv → an N-layer
/// transformer; all intermediate hidden states are harvested (S2V stacks them, like LTX-2's Gemma layer harvest).</summary>
public sealed record Wav2Vec2EncoderConfig
{
    /// <summary>Output channels of each feature-encoder Conv1d.</summary>
    public required int[] ConvDims { get; init; }

    /// <summary>Stride of each feature-encoder Conv1d.</summary>
    public required int[] ConvStrides { get; init; }

    /// <summary>Kernel size of each feature-encoder Conv1d.</summary>
    public required int[] ConvKernels { get; init; }

    /// <summary>Whether the feature-encoder convs carry a bias.</summary>
    public bool ConvBias { get; init; }

    /// <summary>True for <c>feat_extract_norm="group"</c> (GroupNorm on conv layer 0 only); false would be per-layer LayerNorm.</summary>
    public bool GroupNormFirstConvOnly { get; init; } = true;

    /// <summary>Transformer hidden size.</summary>
    public required int Hidden { get; init; }

    /// <summary>Transformer layers.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>FFN intermediate size.</summary>
    public required int Intermediate { get; init; }

    /// <summary>HF <c>do_stable_layer_norm</c> (true for wav2vec2-large): pre-norm transformer layers, the encoder
    /// LayerNorm runs AFTER the layers, and the harvested hidden states are each layer's INPUT plus the final normed
    /// output (post-norm collects the normed input plus each layer's output).</summary>
    public bool StableLayerNorm { get; init; }

    /// <summary>HF <c>do_normalize</c>: z-normalize the waveform (zero mean, unit unbiased variance) before the convs.</summary>
    public bool NormalizeInput { get; init; }

    /// <summary>Positional conv embedding kernel size.</summary>
    public int PosConvKernel { get; init; } = 128;

    /// <summary>Positional conv embedding groups.</summary>
    public int PosConvGroups { get; init; } = 16;

    /// <summary>LayerNorm / GroupNorm epsilon.</summary>
    public float Eps { get; init; } = 1e-5f;

    /// <summary>Total samples consumed per feature frame (product of conv strides — ≈320 for the standard config).</summary>
    public int HopSamples
    {
        get { int h = 1; foreach (int s in ConvStrides) h *= s; return h; }
    }

    /// <summary>Number of harvested hidden states (embedding output + each layer).</summary>
    public int NumHiddenStates => NumLayers + 1;

    /// <summary>chinese-wav2vec2-base / wav2vec2-base preset: hidden 768, 12 layers, 12 heads, intermediate 3072,
    /// conv dims 512×7, strides (5,2,2,2,2,2,2), kernels (10,3,3,3,3,2,2), group-norm first conv, no conv bias.</summary>
    public static Wav2Vec2EncoderConfig Base => new()
    {
        ConvDims = [512, 512, 512, 512, 512, 512, 512],
        ConvStrides = [5, 2, 2, 2, 2, 2, 2],
        ConvKernels = [10, 3, 3, 3, 3, 2, 2],
        ConvBias = false,
        GroupNormFirstConvOnly = true,
        Hidden = 768, NumLayers = 12, NumHeads = 12, Intermediate = 3072,
        PosConvKernel = 128, PosConvGroups = 16,
    };

    /// <summary>wav2vec2-large preset (the S2V front-end): hidden 1024, 24 layers, 16 heads, intermediate 4096,
    /// per-conv LayerNorm feature encoder with biases, stable (pre-norm) layers, normalized input.</summary>
    public static Wav2Vec2EncoderConfig Large => Base with
    {
        Hidden = 1024, NumLayers = 24, NumHeads = 16, Intermediate = 4096,
        ConvBias = true, GroupNormFirstConvOnly = false, StableLayerNorm = true, NormalizeInput = true,
    };
}
