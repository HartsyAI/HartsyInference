using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Fibo 2.0 denoiser configuration. <see cref="Checkpoint"/> holds ComfyUI's defaults;
/// <see cref="FromWeights"/> replaces the structural fields the checkpoint itself defines.</summary>
public sealed record Fibo2Config
{
    /// <summary>Sinusoidal timestep width and adaLN input width.</summary>
    public const int TimestepEmbeddingDim = 256;

    /// <summary>Epsilon of the final LayerNorm (no affine), which differs from the block norms' <see cref="NormEps"/>.</summary>
    public const float FinalNormEps = 1e-6f;

    /// <summary>Latent channels before patchify.</summary>
    public required int InChannels { get; init; }

    /// <summary>Spatial patch size; tokens carry <c>InChannels · PatchSize²</c> features.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Model width.</summary>
    public required int Dim { get; init; }

    /// <summary>Attention heads; <c>NumHeads · HeadDim == Dim</c>.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head width; the RoPE axes sum to it.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Main (noisy and gist) block count.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Global-modulation blocks run on the image tokens before the Perceiver.</summary>
    public required int NumRefinerLayers { get; init; }

    /// <summary>Main-block indices that carry a tanh-gated cross-attention to a text bundle, in bundle order.</summary>
    public required int[] InjectionLayers { get; init; }

    /// <summary>Width of one text bundle (three Qwen3-VL taps concatenated).</summary>
    public required int CapFeatDim { get; init; }

    /// <summary>Learned Perceiver queries; equal to <see cref="MaxGist"/>.</summary>
    public required int PerceiverQueries { get; init; }

    /// <summary>Perceiver blocks (cross-attention, self-attention, FFN).</summary>
    public required int PerceiverLayers { get; init; }

    /// <summary>SwiGLU hidden width (the padded <c>H</c> of <c>feed_forward.w13</c>); read from the checkpoint.</summary>
    public required int FfnDim { get; init; }

    /// <summary>Block norm epsilon, applied to the RMSNorms and the Q/K norms.</summary>
    public required float NormEps { get; init; }

    /// <summary>RoPE base frequency.</summary>
    public required float RopeTheta { get; init; }

    /// <summary>RoPE pair counts per axis (frame, row, col); they sum to <see cref="HeadDim"/>.</summary>
    public required int[] AxesDims { get; init; }

    /// <summary>Multiplier applied to the sigma before the noisy timestep embedding.</summary>
    public required float TimestepScale { get; init; }

    /// <summary>Largest gist count; also the frame plane of the image tokens is <c>MaxGist + 1</c>.</summary>
    public required int MaxGist { get; init; }

    /// <summary>Smallest gist count.</summary>
    public required int MinGist { get; init; }

    /// <summary>Gist counts are multiples of this step.</summary>
    public required int GistStep { get; init; }

    /// <summary>Text length at or below which the gist count is <see cref="MinGist"/>.</summary>
    public required int GistMinTextLen { get; init; }

    /// <summary>Text length at or above which the gist count is <see cref="MaxGist"/>.</summary>
    public required int GistMaxTextLen { get; init; }

    /// <summary>ComfyUI's Fibo 2.0 defaults.</summary>
    public static Fibo2Config Checkpoint { get; } = new()
    {
        InChannels = 32,
        PatchSize = 2,
        Dim = 3584,
        NumHeads = 28,
        HeadDim = 128,
        NumLayers = 44,
        NumRefinerLayers = 2,
        InjectionLayers = [6, 14, 22, 30, 38],
        CapFeatDim = 7680,
        PerceiverQueries = 512,
        PerceiverLayers = 2,
        FfnDim = 9728,
        NormEps = 1e-5f,
        RopeTheta = 256f,
        AxesDims = [32, 48, 48],
        TimestepScale = 1000f,
        MaxGist = 512,
        MinGist = 256,
        GistStep = 64,
        GistMinTextLen = 256,
        GistMaxTextLen = 2048,
    };

    /// <summary>Returns <paramref name="template"/> with every structural field the checkpoint defines replaced by its
    /// tensor shape: width, head dim, depth, FFN width (<c>w2</c> is <c>[dim, H]</c>; the upstream 8/3 formula gives a
    /// different value for this checkpoint), Perceiver size, text bundle width and the injection layers.</summary>
    public static Fibo2Config FromWeights(IReadOnlyDictionary<string, Tensor> weights, Fibo2Config template)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(template);

        int headDim = (int)Require(weights, "layers.0.attention.q_norm.weight").Shape[0];
        int dim = (int)Require(weights, "layers.0.attention.qkv.weight").Shape[1];
        int ffnDim = (int)Require(weights, "layers.0.feed_forward.w2.weight").Shape[1];
        Tensor w13 = Require(weights, "layers.0.feed_forward.w13.weight");
        if (w13.Shape[0] != 2L * ffnDim || w13.Shape[1] != dim)
            throw new InvalidDataException($"Fibo2 feed_forward.w13 is [{w13.Shape[0]}, {w13.Shape[1]}], expected [{2 * ffnDim}, {dim}].");
        if (dim % headDim != 0)
            throw new InvalidDataException($"Fibo2 width {dim} is not a multiple of head dim {headDim}.");

        int numLayers = CountBlocks(weights, "layers.", "attention.qkv.weight");
        int numRefiners = CountBlocks(weights, "noise_refiner.", "attention.qkv.weight");
        int perceiverLayers = CountBlocks(weights, "perceiver.layers.", "self_attention.to_kv.weight");
        int capFeatDim = (int)Require(weights, "perceiver.layers.0.cross_attention.to_kv.weight").Shape[1];
        int queries = (int)Require(weights, "perceiver.latent_queries").Shape[0];

        List<int> injection = [];
        for (int i = 0; i < numLayers; i++)
        {
            if (weights.ContainsKey($"layers.{i}.cross_attention.to_q.weight"))
                injection.Add(i);
        }

        Fibo2Config config = template with
        {
            Dim = dim,
            NumHeads = dim / headDim,
            HeadDim = headDim,
            NumLayers = numLayers,
            NumRefinerLayers = numRefiners,
            PerceiverLayers = perceiverLayers,
            PerceiverQueries = queries,
            FfnDim = ffnDim,
            CapFeatDim = capFeatDim,
            InjectionLayers = injection.ToArray(),
        };
        config.Validate();
        return config;
    }

    /// <summary>Gist token count for a text of <paramref name="textLength"/> tokens: <see cref="MinGist"/> up to
    /// <see cref="GistMinTextLen"/>, rising linearly to <see cref="MaxGist"/> at <see cref="GistMaxTextLen"/>, rounded up
    /// to a multiple of <see cref="GistStep"/>.</summary>
    public int GistCount(int textLength)
    {
        if (textLength < 0)
            throw new ArgumentOutOfRangeException(nameof(textLength), "Text length cannot be negative.");
        double ratio = Math.Clamp((textLength - GistMinTextLen) / (double)(GistMaxTextLen - GistMinTextLen), 0.0, 1.0);
        double raw = MinGist + ratio * (MaxGist - MinGist);
        int rounded = (int)Math.Ceiling(raw / GistStep) * GistStep;
        return Math.Clamp(rounded, MinGist, MaxGist);
    }

    /// <summary>Checks the invariants the blocks and the RoPE tables rely on.</summary>
    public void Validate()
    {
        if (InChannels <= 0 || PatchSize <= 0 || Dim <= 0 || NumHeads <= 0 || HeadDim <= 0 || FfnDim <= 0 || CapFeatDim <= 0)
            throw new InvalidOperationException("Fibo2 dimensions must be positive.");
        if (NumHeads * HeadDim != Dim)
            throw new InvalidOperationException($"Fibo2: {NumHeads} heads x head dim {HeadDim} != width {Dim}.");
        int axesTotal = 0;
        foreach (int axis in AxesDims)
            axesTotal += axis;
        if (AxesDims.Length == 0 || axesTotal != HeadDim)
            throw new InvalidOperationException($"Fibo2 RoPE axes must sum to the head dim {HeadDim}.");
        if (MaxGist != PerceiverQueries)
            throw new InvalidOperationException($"Fibo2 Perceiver has {PerceiverQueries} queries but MaxGist is {MaxGist}.");
        if (MinGist <= 0 || MinGist > MaxGist || GistStep <= 0 || GistMinTextLen < 0 || GistMaxTextLen <= GistMinTextLen)
            throw new InvalidOperationException("Fibo2 gist schedule is inconsistent.");
        foreach (int layer in InjectionLayers)
        {
            if (layer < 0 || layer >= NumLayers)
                throw new InvalidOperationException($"Fibo2 injection layer {layer} is outside 0..{NumLayers - 1}.");
        }
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key) =>
        weights.TryGetValue(key, out Tensor? tensor) ? tensor : throw new KeyNotFoundException($"Fibo2 checkpoint is missing '{key}'.");

    private static int CountBlocks(IReadOnlyDictionary<string, Tensor> weights, string prefix, string suffix)
    {
        int count = 0;
        foreach (string key in weights.Keys)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal) && key.EndsWith(suffix, StringComparison.Ordinal))
                count++;
        }
        for (int i = 0; i < count; i++)
        {
            if (!weights.ContainsKey($"{prefix}{i}.{suffix}"))
                throw new InvalidDataException($"Fibo2 block indices under '{prefix}' are not contiguous (missing {i}).");
        }
        return count;
    }
}
