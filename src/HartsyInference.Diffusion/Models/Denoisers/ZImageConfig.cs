using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Z-Image transformers (Tongyi Lab, Apache 2.0). Lumina2/NextDiT architecture: 30 main DiT layers + 2 noise refiner blocks + 2 context refiner blocks. RMSNorm everywhere, multi-axis RoPE [32,48,48] θ=256, AdaLN with 4 outputs, SwiGLU FFN. See docs/Research/Z_IMAGE_ARCHITECTURE.md.</summary>
public sealed record ZImageConfig
{
    /// <summary>Whether these are the undistilled Base weights. Base uses the official shift=6 schedule and keeps
    /// its transformer token stream in F32: its wider activation range is not covered by Turbo's F16 sandwich
    /// damping audit.</summary>
    public bool IsBase { get; init; }

    /// <summary>Hidden dimension (3840 for Z-Image-Turbo and Base).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads (30 for Z-Image; full MHA, no GQA in DiT).</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension (128 for Z-Image, = HiddenSize / NumHeads).</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Number of main transformer layers (30 for Z-Image).</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of refiner layers per refiner stack (2 for noise_refiner, 2 for context_refiner).</summary>
    public int NumRefinerLayers { get; init; } = 2;

    /// <summary>Latent in-channels (16, matches Flux VAE).</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Spatial patch size (2 → 2×2 patchify on the H/W axes).</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Frame patch size (1 for image-only Z-Image).</summary>
    public int FramePatchSize { get; init; } = 1;

    /// <summary>Caption feature dim into the transformer (2560, matches Qwen3-4B hidden_size). cap_embedder projects this to HiddenSize.</summary>
    public int CapFeatDim { get; init; } = 2560;

    /// <summary>AdaLN embedding dim (256). Linear projects min(HiddenSize, 256) → 4*HiddenSize per modulation site.</summary>
    public int AdaLNEmbedDim { get; init; } = 256;

    /// <summary>RMSNorm epsilon throughout.</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>RoPE base frequency (256 for Z-Image, vs Flux's 10000). Much higher base period → smaller frequencies.</summary>
    public float RopeTheta { get; init; } = 256.0f;

    /// <summary>Per-axis RoPE dimensions [frame, height, width]. Must sum to HeadDim. Default [32,48,48] for Z-Image.</summary>
    public int[] AxesDims { get; init; } = [32, 48, 48];

    /// <summary>Maximum precomputable axis lengths [frame, height, width]. Used to cap freq tables.</summary>
    public int[] AxesLens { get; init; } = [1536, 512, 512];

    /// <summary>Sequence-padding multiple (32). Caption and image token sequences are each padded to a multiple of this with learned pad tokens.</summary>
    public int SeqMultiOf { get; init; } = 32;

    /// <summary>Sinusoidal timestep scale (1000 for Z-Image's t_embedder).</summary>
    public float TimestepScale { get; init; } = 1000.0f;

    /// <summary>FFN inner dimension. For Z-Image-Turbo this is 10240 (≈ 8/3 × 3840 rounded to multiple of 256). Verify by inspecting layers.0.feed_forward.w1.weight shape on load.</summary>
    public required int FfnDim { get; init; }

    /// <summary>VAE scale factor (Flux VAE: 0.3611). Latents are scaled by (latent - shift) * scale before transformer input.</summary>
    public float VaeScaleFactor { get; init; } = 0.3611f;

    /// <summary>VAE shift factor (Flux VAE: 0.1159).</summary>
    public float VaeShiftFactor { get; init; } = 0.1159f;

    /// <summary>VAE spatial downscale factor (8 for Flux VAE).</summary>
    public int VaeDownscaleFactor { get; init; } = 8;

    /// <summary>Static shift for FlowMatchEulerDiscreteScheduler (3.0 for Turbo, 6.0 for Base; no dynamic shifting).</summary>
    public float SchedulerShift { get; init; } = 3.0f;

    /// <summary>Z-Image-Turbo preset (6B, 8-NFE distilled, no CFG). Recommended sampling: 8 steps at <c>cfgScale=1.0</c>. FfnDim verified from Tongyi-MAI/Z-Image-Turbo transformer/config.json (= 10240). SchedulerShift=3.0.</summary>
    public static ZImageConfig Turbo => new()
    {
        HiddenSize = 3840,
        NumHeads = 30,
        HeadDim = 128,
        NumLayers = 30,
        NumRefinerLayers = 2,
        FfnDim = 10240,
        IsBase = false,
        SchedulerShift = 3.0f,
    };

    /// <summary>Z-Image-Base preset (6B, un-distilled foundation model, full CFG). Architecturally identical to Turbo per Tongyi-MAI/Z-Image transformer/config.json (released 2026-01-28); only difference is the un-distilled weights and the sampling regime — recommended <c>cfgScale=3.0..5.0</c> with a negative prompt and 28–50 steps. SchedulerShift=6.0 (Turbo uses 3.0).</summary>
    public static ZImageConfig Base => new()
    {
        HiddenSize = 3840,
        NumHeads = 30,
        HeadDim = 128,
        NumLayers = 30,
        NumRefinerLayers = 2,
        FfnDim = 10240,
        IsBase = true,
        SchedulerShift = 6.0f,
    };

    /// <summary>Auto-detects depth and FFN dim from a transformer weight dict using Z-Image's diffusers/single-file naming.</summary>
    public static ZImageConfig FromWeights(IReadOnlyDictionary<string, Tensor> weights, bool isBase = false)
    {
        int maxLayer = -1;
        foreach (string key in weights.Keys)
        {
            if (key.StartsWith("layers.", StringComparison.Ordinal))
            {
                int dot = key.IndexOf('.', 7);
                if (dot > 0 && int.TryParse(key.AsSpan(7, dot - 7), out int idx) && idx > maxLayer)
                    maxLayer = idx;
            }
        }
        int numLayers = maxLayer + 1;

        int ffnDim = 10240;
        if (weights.TryGetValue("layers.0.feed_forward.w1.weight", out Tensor? w1))
        {
            ffnDim = (int)w1.Shape[0];
        }

        int hidden = 3840;
        if (weights.TryGetValue("layers.0.attention.to_q.weight", out Tensor? toQ))
        {
            hidden = (int)toQ.Shape[0];
        }

        return new ZImageConfig
        {
            HiddenSize = hidden,
            NumHeads = hidden / 128,
            HeadDim = 128,
            NumLayers = numLayers > 0 ? numLayers : 30,
            FfnDim = ffnDim,
            IsBase = isBase,
            SchedulerShift = isBase ? 6.0f : 3.0f,
        };
    }
}
