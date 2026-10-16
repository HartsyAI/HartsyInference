using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for MOSS-Image transformers (OpenMOSS, a single-stream DiT over <c>[image ; text]</c> with one
/// adaLN modulation shared by every block). Every dimension that varies between builds is read from the checkpoint
/// by <see cref="FromWeights"/>; the rest is fixed by the architecture (OpenMOSS's ComfyUI module
/// <c>comfyui/moss-image/model.py</c>, the same math as SGLang's <c>runtime/models/dits/moss_image.py</c>).</summary>
public sealed record MossImageConfig
{
    /// <summary>The reference sampling shift (<c>flow_shift</c>).</summary>
    public const float ReferenceShift = 5f;

    /// <summary>The reference step count.</summary>
    public const int ReferenceSteps = 50;

    /// <summary>The reference CFG scale.</summary>
    public const float ReferenceCfg = 3f;

    /// <summary>The joint block the reference skips on the unconditional branch (0-indexed; 24 and 26 also help).</summary>
    public const int ReferenceUncondSkipLayer = 22;

    /// <summary>Longest formatted prompt, chat template included, the model was trained on.</summary>
    public const int MaxTextTokens = 4096;

    /// <summary>Hidden dimension (= <see cref="NumHeads"/> × <see cref="HeadDim"/>).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head width; a multiple of 8, since RoPE gives each of its four axes a quarter of the head.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Joint blocks over <c>[image ; text]</c>.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Text-refiner blocks between the text projection and the joint stack.</summary>
    public required int NumTextBlocks { get; init; }

    /// <summary>SiTU-GLU inner width (half the <c>ffn_up_gate</c> rows).</summary>
    public required int FfnHidden { get; init; }

    /// <summary>Channels per packed latent token: the 32-channel FLUX.2 VAE latent × a 2×2 patch.</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Text feature width: Qwen3.5-9B's final hidden state.</summary>
    public int TextDim { get; init; } = 4096;

    /// <summary>Sinusoidal timestep embedding width.</summary>
    public int FreqDim { get; init; } = 256;

    /// <summary>RoPE base frequency.</summary>
    public float RopeBase { get; init; } = 10000f;

    /// <summary>Epsilon of every RMSNorm in the model.</summary>
    public float Eps { get; init; } = 1e-6f;

    /// <summary>FLUX.2 VAE latent channels before the 2×2 patchify.</summary>
    public int VaeLatentChannels { get; init; } = 32;

    /// <summary>Latent patch size.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Image pixels per transformer token along each axis (VAE 8× × 2×2 patch).</summary>
    public int VaeDownscaleFactor { get; init; } = 16;

    /// <summary>Derives the configuration from converted transformer weights. Shapes are logical element counts for every
    /// storage format (a packed GGUF or FP4 weight reports its unpacked <c>[out, in]</c>), so columns can be read too.</summary>
    public static MossImageConfig FromWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        int hidden = (int)Require(weights, "video_embedder.weight").Shape[0];
        int headDim = (int)Require(weights, "layers.0.q_norm.weight").ElementCount;
        if (hidden <= 0 || headDim <= 0 || headDim % 8 != 0 || hidden % headDim != 0)
            throw new InvalidDataException($"MOSS-Image hidden size {hidden} and head width {headDim} are inconsistent; the head must be a multiple of 8 dividing the hidden size.");
        int ffnRows = (int)Require(weights, "layers.0.proj.ffn_up_gate.weight").Shape[0];
        if (ffnRows % 2 != 0)
            throw new InvalidDataException($"MOSS-Image ffn_up_gate has {ffnRows} rows; a gated FFN input must be even.");
        int layers = CountBlocks(weights, "layers.", ".proj.o.weight");
        int textBlocks = CountBlocks(weights, "text_embedder.blocks.", ".qkv.weight");
        if (layers == 0 || textBlocks == 0)
            throw new InvalidDataException($"MOSS-Image checkpoint has {layers} joint and {textBlocks} text blocks; both must be non-zero.");
        return new MossImageConfig
        {
            HiddenSize = hidden,
            NumHeads = hidden / headDim,
            HeadDim = headDim,
            NumLayers = layers,
            NumTextBlocks = textBlocks,
            FfnHidden = ffnRows / 2,
            LatentChannels = (int)Require(weights, "final_linear_video.weight").Shape[0],
            TextDim = (int)Require(weights, "text_embedder.proj.weight").Shape[1],
            FreqDim = (int)Require(weights, "time_embedding.0.weight").Shape[1],
        };
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key) =>
        weights.TryGetValue(key, out Tensor? tensor)
            ? tensor : throw new InvalidDataException($"MOSS-Image checkpoint is missing '{key}'.");

    /// <summary>Counts contiguous <c>{prefix}{i}{suffix}</c> blocks starting at 0.</summary>
    private static int CountBlocks(IReadOnlyDictionary<string, Tensor> weights, string prefix, string suffix)
    {
        int count = 0;
        while (weights.ContainsKey($"{prefix}{count}{suffix}")) count++;
        return count;
    }
}
