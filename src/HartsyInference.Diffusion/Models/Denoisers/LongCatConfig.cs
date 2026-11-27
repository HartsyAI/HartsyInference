using System.Text.Json;
using HartsyInference.Core.Backends.Attention;
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

    /// <summary>The block-sparse self-attention of the 720p refinement (<c>bsa_params</c>). It runs only on a forward that asks
    /// for it.</summary>
    public LongCatSparseAttention SparseAttention { get; init; } = LongCatSparseAttention.Released;

    /// <summary>Width of the audio tokens the avatar's audio cross-attention reads (the reference's <c>output_dim</c>). Used only when a block
    /// is given audio weights.</summary>
    public int AudioOutputDim { get; init; } = 768;

    /// <summary>Audio window of the avatar projection (the reference's <c>audio_window</c>).</summary>
    public int AudioWindow { get; init; } = 5;

    /// <summary>Audio blocks per window (the reference's <c>audio_block</c>; 5 for Avatar 1.6).</summary>
    public int AudioBlock { get; init; } = 5;

    /// <summary>Channels per audio block (the reference's <c>audio_channel</c>; 1280 for Avatar 1.6, the Whisper hidden size).</summary>
    public int AudioChannel { get; init; } = 1280;

    /// <summary>Hidden width of the avatar audio projection (the reference's <c>intermediate_dim</c>).</summary>
    public int AudioIntermediateDim { get; init; } = 512;

    /// <summary>Audio tokens per latent frame (the reference's <c>context_tokens</c>).</summary>
    public int AudioContextTokens { get; init; } = 32;

    /// <summary>Video frames per latent frame (the reference's <c>vae_scale</c>); the latter audio windows are grouped by it.</summary>
    public int AudioVaeScale { get; init; } = 4;

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
            patch = (Whole(patchElement[0], "patch_size"), Whole(patchElement[1], "patch_size"), Whole(patchElement[2], "patch_size"));
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
            SparseAttention = SparseAttentionOf(root, defaults.SparseAttention),
            AudioOutputDim = Int(root, "output_dim", defaults.AudioOutputDim),
            AudioWindow = Int(root, "audio_window", defaults.AudioWindow),
            AudioBlock = Int(root, "audio_block", defaults.AudioBlock),
            AudioChannel = Int(root, "audio_channel", defaults.AudioChannel),
            AudioIntermediateDim = Int(root, "intermediate_dim", defaults.AudioIntermediateDim),
            AudioContextTokens = Int(root, "context_tokens", defaults.AudioContextTokens),
            AudioVaeScale = Int(root, "vae_scale", defaults.AudioVaeScale),
        };
        // The avatar configs ship audio_prenorm false; the pre-audio norm is not implemented, so true is refused rather than ignored.
        if (root.TryGetProperty("audio_prenorm", out JsonElement prenorm) && prenorm.ValueKind == JsonValueKind.True)
            throw new InvalidDataException("audio_prenorm true is not supported: the audio pre-norm is not implemented.");
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

    /// <summary>The <c>bsa_params</c> of a config, or <paramref name="fallback"/> when it has none. Only the top-k form is
    /// implemented, so any other parameter is refused rather than ignored.</summary>
    private static LongCatSparseAttention SparseAttentionOf(JsonElement root, LongCatSparseAttention fallback)
    {
        if (!root.TryGetProperty("bsa_params", out JsonElement parameters) || parameters.ValueKind != JsonValueKind.Object)
            return fallback;
        LongCatSparseAttention sparse = fallback;
        foreach (JsonProperty parameter in parameters.EnumerateObject())
        {
            JsonElement value = parameter.Value;
            sparse = parameter.Name switch
            {
                "sparsity" when value.ValueKind == JsonValueKind.Number && value.GetDouble() is >= 0.0 and <= 1.0 =>
                    sparse with { Sparsity = value.GetDouble() },
                "chunk_3d_shape_q" => sparse with { QueryBlock = Block(value, parameter.Name) },
                "chunk_3d_shape_k" => sparse with { KeyBlock = Block(value, parameter.Name) },
                _ => throw new InvalidDataException($"The LongCat config's bsa_params.{parameter.Name} is not supported: "
                    + "sparsity in [0, 1], chunk_3d_shape_q and chunk_3d_shape_k are."),
            };
        }
        return sparse;
    }

    private static TokenGrid3D Block(JsonElement value, string name)
    {
        if (value.ValueKind == JsonValueKind.Array && value.GetArrayLength() == 3)
        {
            (int frames, int height, int width) = (Whole(value[0], name), Whole(value[1], name), Whole(value[2], name));
            if (frames > 0 && height > 0 && width > 0)
                return new TokenGrid3D(frames, height, width);
        }
        throw new InvalidDataException($"The LongCat config's bsa_params.{name} must be three positive whole numbers.");
    }

    private static int Int(JsonElement root, string name, int fallback) =>
        root.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.Number ? Whole(value, name) : fallback;

    /// <summary>A JSON number that is a whole value, written as an integer or as an integral float (<c>4.0</c>, which diffusers configs use).</summary>
    private static int Whole(JsonElement value, string name)
    {
        if (value.TryGetInt32(out int whole))
            return whole;
        double real = value.GetDouble();
        if (real == Math.Floor(real) && real is >= int.MinValue and <= int.MaxValue)
            return (int)real;
        throw new InvalidDataException($"The LongCat config field '{name}' must be a whole number; got {value.GetRawText()}.");
    }
}
