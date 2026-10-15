using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for FLUX.3 Klein transformers. Every dimension that varies between builds is read from the
/// checkpoint by <see cref="FromWeights"/>; the remaining fields are fixed by the architecture
/// (ComfyUI <c>comfy/ldm/flux3/model.py</c>).</summary>
public sealed record Flux3Config
{
    /// <summary>The static flow-match time shift <c>mu = ln(6.93)</c> the reference samples with.</summary>
    public const float ReferenceTimeShiftMu = 1.93586f;

    /// <summary>Hidden dimension (= <see cref="NumHeads"/> × <see cref="HeadDim"/>).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads; the head width is fixed at 128.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Blocks per conditioning mode (image, image_cond, text) before the joint stack.</summary>
    public required int Depth { get; init; }

    /// <summary>Joint blocks over the concatenated <c>[image, image_cond, text]</c> sequence.</summary>
    public required int DepthSingleBlocks { get; init; }

    /// <summary>SwiGLU inner width (half the <c>mlp_in</c> rows).</summary>
    public required int MlpHiddenSize { get; init; }

    /// <summary>True for the step-distilled flow-map build, which also embeds the time each step jumps to.</summary>
    public required bool FlowMap { get; init; }

    /// <summary>Per-head dimension; equals the sum of <see cref="AxesDim"/>.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Channels per packed latent token: 32-channel Flux.2 VAE latent × 2×2 patch.</summary>
    public int InChannels { get; init; } = 128;

    /// <summary>Text feature width into <c>txt_in</c>: eight Qwen3-VL-4B taps × 2560.</summary>
    public int ContextInDim { get; init; } = 20480;

    /// <summary>RoPE per-axis widths (t, row, col, l).</summary>
    public int[] AxesDim { get; init; } = [32, 32, 32, 32];

    /// <summary>RoPE base frequency.</summary>
    public int Theta { get; init; } = 10000;

    /// <summary>Scale applied to image row/column positions.</summary>
    public float ImageRopeScale { get; init; } = 0.5f;

    /// <summary>Sinusoidal timestep embedding width.</summary>
    public int TimestepChannels { get; init; } = 256;

    /// <summary>LayerNorm epsilon of the modulated pre-norm and the final layer.</summary>
    public float LayerNormEps { get; init; } = 1e-6f;

    /// <summary>Per-head Q/K RMSNorm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Flux.2 VAE latent channels before patchify.</summary>
    public int VaeLatentChannels { get; init; } = 32;

    /// <summary>Latent patch size.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Image pixels per transformer token along each axis (VAE 8× × 2×2 patch).</summary>
    public int VaeDownscaleFactor { get; init; } = 16;

    /// <summary>Hidden-state taps of the Qwen3-VL-4B text encoder, in HuggingFace <c>hidden_states</c> indexing.</summary>
    public int[] TextEncoderLayers { get; init; } = [4, 8, 12, 16, 20, 24, 28, 32];

    /// <summary>Prompt tokens are right-padded to a multiple of this.</summary>
    public int TextPadMultiple { get; init; } = 80;

    /// <summary>Most reference images one generation conditions on.</summary>
    public int MaxReferenceImages { get; init; } = 10;

    /// <summary>Derives the configuration from converted transformer weights (see <c>Flux3CheckpointConverter</c>).
    /// Only row counts are read, so packed-quant column layouts cannot skew the result.</summary>
    public static Flux3Config FromWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        int hidden = (int)Require(weights, "content_mode_blocks.image.0.attn_out.weight").Shape[0];
        const int headDim = 128;
        if (hidden <= 0 || hidden % headDim != 0)
            throw new InvalidDataException($"FLUX.3 hidden size {hidden} is not a positive multiple of {headDim}.");
        int mlpRows = (int)Require(weights, "content_mode_blocks.image.0.mlp_in.weight").Shape[0];
        if (mlpRows % 2 != 0)
            throw new InvalidDataException($"FLUX.3 mlp_in has {mlpRows} rows; a SwiGLU input must be even.");
        int depth = CountBlocks(weights, "txt_mode_blocks.");
        int singles = CountBlocks(weights, "single_blocks.");
        if (depth == 0 || singles == 0)
            throw new InvalidDataException($"FLUX.3 checkpoint has {depth} per-mode and {singles} joint blocks; both must be non-zero.");
        return new Flux3Config
        {
            HiddenSize = hidden,
            NumHeads = hidden / headDim,
            HeadDim = headDim,
            Depth = depth,
            DepthSingleBlocks = singles,
            MlpHiddenSize = mlpRows / 2,
            FlowMap = weights.ContainsKey("ref_time_in.in_layer.weight"),
        };
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key) =>
        weights.TryGetValue(key, out Tensor? tensor)
            ? tensor : throw new InvalidDataException($"FLUX.3 checkpoint is missing '{key}'.");

    /// <summary>Counts contiguous <c>{prefix}{i}.attn_out.weight</c> blocks starting at 0.</summary>
    private static int CountBlocks(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        int count = 0;
        while (weights.ContainsKey($"{prefix}{count}.attn_out.weight")) count++;
        return count;
    }
}
