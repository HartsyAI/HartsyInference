using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>MiniMax-H3 ("Hailuo 03") DiT config — a single-stream packed-token transformer that denoises video and
/// stereo audio latents jointly. Every field is derived from checkpoint shapes by <see cref="Detect"/>; the property
/// defaults are only the reference's nominal values.</summary>
public sealed record MiniMaxH3Config
{
    public int HiddenSize { get; init; } = 5376;
    public int NumLayers { get; init; } = 50;
    public int TokenRefinerNumLayers { get; init; } = 2;
    public int NumAttentionHeads { get; init; } = 56;
    public int AttentionHeadDim { get; init; } = 128;
    public int FfnHiddenSize { get; init; } = 14336;

    /// <summary>Video latent channels (24); the patch is 1x2x2 so the head emits 4x this.</summary>
    public int LatentsDim { get; init; } = 24;

    /// <summary>Audio latent channels (32), stereo at 40 Hz.</summary>
    public int AudioLatentsDim { get; init; } = 32;

    public int PatchT { get; init; } = 1;
    public int PatchH { get; init; } = 2;
    public int PatchW { get; init; } = 2;

    /// <summary>Qwen3-VL hidden width feeding <c>condition_proj</c> (5120 for the 32B).</summary>
    public int TextDim { get; init; } = 5120;

    public int TimestepInputDim { get; init; } = 256;
    public int TimeEmbedHiddenSize { get; init; } = 5376;
    public int TimeEmbedDim { get; init; } = 2688;
    public int RopeInvFreqLen { get; init; } = 16;

    public float NormEps { get; init; } = 1e-5f;
    public float QkNormEps { get; init; } = 1e-5f;
    public float FinalNormEps { get; init; } = 1e-5f;

    /// <summary>Flow-shift for the video stream; the sampler runs on this schedule.</summary>
    public float SigmaShiftVideo { get; init; } = 12.0f;

    /// <summary>Flow-shift for the audio stream, derived from the video sigma in closed form.</summary>
    public float SigmaShiftAudio { get; init; } = 3.0f;

    /// <summary>Row count of <c>adaln_t_table</c> when the checkpoint ships the pruned curve-basis form; null means the
    /// full variant with a <c>time_embedder</c>. The curve form drops the SiLU before the adaln linears.</summary>
    public int? AdalnCurveGrid { get; init; }

    /// <summary>True when adaln reads the precomputed time-embedding curve instead of a time embedder.</summary>
    public bool UseAdalnCurves => AdalnCurveGrid is not null;

    /// <summary>Video rows carry <c>LatentsDim * patch volume</c> features.</summary>
    public int VideoPatchDim => LatentsDim * PatchT * PatchH * PatchW;

    /// <summary>True when <paramref name="weights"/> carries the H3 signature (both patch projections).</summary>
    public static bool Matches(IReadOnlyDictionary<string, Tensor> weights) =>
        weights.ContainsKey("video_patch_proj.weight") && weights.ContainsKey("audio_patch_proj.weight");

    /// <summary>Derives the config from checkpoint shapes, mirroring ComfyUI's <c>model_detection</c> H3 branch —
    /// nothing is assumed from the nominal defaults.</summary>
    public static MiniMaxH3Config Detect(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (!Matches(weights))
        {
            throw new ArgumentException(
                "Not a MiniMax-H3 checkpoint: both 'video_patch_proj.weight' and 'audio_patch_proj.weight' are required.");
        }

        int hidden = (int)weights["video_patch_proj.weight"].Shape[0];
        int headDim = (int)Require(weights, "blocks.0.attn.q_norm.weight").Shape[0];
        long qkvRows = Require(weights, "blocks.0.attn.qkv_proj.weight").Shape[0];
        int heads = (int)(qkvRows / (3L * headDim));
        // fc1 is the gated SwiGLU pair, so the true ffn width is half its rows.
        int ffn = (int)(Require(weights, "blocks.0.mlp.fc1.weight").Shape[0] / 2);
        int textDim = (int)Require(weights, "condition_proj.weight").Shape[1];
        // video_out emits the patch volume (1x2x2 = 4) times the latent channels.
        int latentsDim = (int)(Require(weights, "final_layer.video_out.weight").Shape[0] / 4);
        int audioLatentsDim = (int)Require(weights, "final_layer.audio_out.weight").Shape[0];
        int ropeLen = (int)Require(weights, "rope.inv_freq").Shape[0];

        MiniMaxH3Config config = new MiniMaxH3Config
        {
            HiddenSize = hidden,
            NumLayers = CountBlocks(weights, "blocks."),
            TokenRefinerNumLayers = CountBlocks(weights, "token_refiner.blocks."),
            NumAttentionHeads = heads,
            AttentionHeadDim = headDim,
            FfnHiddenSize = ffn,
            LatentsDim = latentsDim,
            AudioLatentsDim = audioLatentsDim,
            TextDim = textDim,
            RopeInvFreqLen = ropeLen,
        };

        if (weights.TryGetValue("adaln_t_table", out Tensor? table))
        {
            return config with { AdalnCurveGrid = (int)table.Shape[0], TimeEmbedDim = (int)table.Shape[1] };
        }
        Tensor projIn = Require(weights, "time_embedder.proj_in.weight");
        return config with
        {
            TimestepInputDim = (int)projIn.Shape[1],
            TimeEmbedHiddenSize = (int)projIn.Shape[0],
            TimeEmbedDim = (int)Require(weights, "time_embedder.proj_out.weight").Shape[0],
        };
    }

    /// <summary>Highest <c>{prefix}{i}.</c> index present, plus one.</summary>
    private static int CountBlocks(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        int count = 0;
        while (weights.Keys.Any(k => k.StartsWith($"{prefix}{count}.", StringComparison.Ordinal)))
        {
            count++;
        }
        return count;
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key) =>
        weights.TryGetValue(key, out Tensor? t) ? t
            : throw new ArgumentException($"MiniMax-H3 checkpoint is missing '{key}'.");
}
