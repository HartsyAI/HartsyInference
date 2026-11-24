using System.Text.Json;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Geometry of the LongCat-Video DiT (<c>LongCatVideoTransformer3DModel</c>, meituan-longcat/LongCat-Video). The audio-conditioned
/// avatar models extend this with audio fields in the phase that adds the audio cross-attention.</summary>
public sealed record LongCatConfig
{
    /// <summary>Epsilon of every LayerNorm and of the per-head QK RMSNorm.</summary>
    public const float NormEps = 1e-6f;

    /// <summary>RoPE frequency base.</summary>
    public const double RopeTheta = 10000.0;

    /// <summary>Latent channels in.</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Latent channels out.</summary>
    public int OutChannels { get; init; } = 16;

    /// <summary>Token width.</summary>
    public int HiddenSize { get; init; } = 4096;

    /// <summary>Transformer blocks.</summary>
    public int Depth { get; init; } = 48;

    /// <summary>Attention heads.</summary>
    public int NumHeads { get; init; } = 32;

    /// <summary>Width of the text encoder features.</summary>
    public int CaptionChannels { get; init; } = 4096;

    /// <summary>FFN width factor before the SwiGLU 2/3 rule.</summary>
    public int MlpRatio { get; init; } = 4;

    /// <summary>Width of the timestep embedding that feeds every block's adaLN projection.</summary>
    public int AdalnTembedDim { get; init; } = 512;

    /// <summary>Width of the sinusoidal timestep features.</summary>
    public int FrequencyEmbeddingSize { get; init; } = 256;

    /// <summary>Patch size (T, H, W); the temporal patch is 1.</summary>
    public (int T, int H, int W) PatchSize { get; init; } = (1, 2, 2);

    /// <summary>Text rows past the valid length are zeroed after the caption projection and still attended (both shipped configs
    /// set this); off drops them.</summary>
    public bool TextTokensZeroPad { get; init; } = true;

    /// <summary>Width of one attention head.</summary>
    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>SwiGLU hidden width: <c>int(2·hidden·ratio / 3)</c> rounded up to a multiple of 256.</summary>
    public int FfnHiddenDim => (int)((2L * HiddenSize * MlpRatio / 3 + 255) / 256 * 256);

    /// <summary>Elements of one patch vector: channels × patch volume.</summary>
    public int PatchVector => InChannels * PatchSize.T * PatchSize.H * PatchSize.W;

    /// <summary>Throws unless the geometry can run: heads divide the width, the RoPE split needs a head width that is a multiple of 8.</summary>
    public void Validate()
    {
        if (HiddenSize <= 0 || Depth <= 0 || NumHeads <= 0 || HiddenSize % NumHeads != 0)
            throw new ArgumentException($"LongCat needs a positive hidden size divisible by the head count; got {HiddenSize} / {NumHeads}.");
        if (HeadDim % 8 != 0)
            throw new ArgumentException($"The 3D RoPE needs a head width that is a multiple of 8; got {HeadDim}.");
        if (PatchSize.T != 1)
            throw new ArgumentException("The temporal patch size must be 1: the per-frame modulation maps one latent frame to one timestep.");
        if (AdalnTembedDim <= 0 || FrequencyEmbeddingSize <= 0 || FrequencyEmbeddingSize % 2 != 0)
            throw new ArgumentException("The timestep embedding widths must be positive and the sinusoidal width even.");
    }

    /// <summary>Reads a <c>config.json</c> of the DiT folder; absent fields keep the shipped defaults.</summary>
    public static LongCatConfig FromJson(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        LongCatConfig defaults = new();
        (int T, int H, int W) patch = defaults.PatchSize;
        if (root.TryGetProperty("patch_size", out JsonElement patchElement) && patchElement.GetArrayLength() == 3)
            patch = (patchElement[0].GetInt32(), patchElement[1].GetInt32(), patchElement[2].GetInt32());
        bool zeroPad = defaults.TextTokensZeroPad;
        if (root.TryGetProperty("text_tokens_zero_pad", out JsonElement zero) && zero.ValueKind is JsonValueKind.True or JsonValueKind.False)
            zeroPad = zero.GetBoolean();
        LongCatConfig config = new()
        {
            InChannels = Int(root, "in_channels", defaults.InChannels),
            OutChannels = Int(root, "out_channels", defaults.OutChannels),
            HiddenSize = Int(root, "hidden_size", defaults.HiddenSize),
            Depth = Int(root, "depth", defaults.Depth),
            NumHeads = Int(root, "num_heads", defaults.NumHeads),
            CaptionChannels = Int(root, "caption_channels", defaults.CaptionChannels),
            MlpRatio = Int(root, "mlp_ratio", defaults.MlpRatio),
            AdalnTembedDim = Int(root, "adaln_tembed_dim", defaults.AdalnTembedDim),
            FrequencyEmbeddingSize = Int(root, "frequency_embedding_size", defaults.FrequencyEmbeddingSize),
            PatchSize = patch,
            TextTokensZeroPad = zeroPad,
        };
        config.Validate();
        return config;
    }

    /// <summary>Derives the geometry from tensor shapes. The head count follows from the per-head QK norm width; the patch size and the zero-pad
    /// flag are not in the weights and are passed in.</summary>
    public static LongCatConfig FromShapes(Func<string, TensorShape?> shapeOf, int depth, (int T, int H, int W) patch, bool textTokensZeroPad)
    {
        TensorShape patchShape = shapeOf("x_embedder.proj.weight") ?? throw new ArgumentException("The checkpoint has no x_embedder.proj.weight.");
        TensorShape qNorm = Required(shapeOf, "blocks.0.attn.q_norm.weight");
        TensorShape caption = Required(shapeOf, "y_embedder.y_proj.0.weight");
        TensorShape adaln = Required(shapeOf, "blocks.0.adaLN_modulation.1.weight");
        TensorShape timeIn = shapeOf("t_embedder.mlp.0.weight") ?? throw new ArgumentException("The checkpoint has no t_embedder.mlp.0.weight.");
        TensorShape finalLinear = Required(shapeOf, "final_layer.linear.weight");
        int hidden = (int)patchShape[0], headDim = (int)qNorm[0];
        int patchVolume = patch.T * patch.H * patch.W;
        LongCatConfig config = new()
        {
            HiddenSize = hidden,
            Depth = depth,
            NumHeads = hidden / headDim,
            InChannels = (int)patchShape[1],
            OutChannels = (int)finalLinear[0] / patchVolume,
            CaptionChannels = (int)caption[1],
            AdalnTembedDim = (int)adaln[1],
            FrequencyEmbeddingSize = (int)timeIn[1],
            PatchSize = patch,
            TextTokensZeroPad = textTokensZeroPad,
        };
        config.Validate();
        return config;
    }

    private static TensorShape Required(Func<string, TensorShape?> shapeOf, string key) =>
        shapeOf(key) ?? throw new ArgumentException($"The checkpoint has no {key}.");

    private static int Int(JsonElement root, string name, int fallback) =>
        root.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.Number ? value.GetInt32() : fallback;
}
