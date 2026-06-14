using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Lumina-Image-2.0 transformers (Alpha-VLLM, Apache 2.0). Sibling NextDiT family member of Z-Image — same overall block structure (RMSNorm + AdaLN modulation + GQA attention + GeGLU/SwiGLU FFN + multi-axis RoPE) but architecturally divergent in: separate (not fused) Q/K/V projections, GQA (24 Q heads / 8 KV heads), no learned cap_pad/x_pad tokens, no fixed seq-multiple-of padding, RoPE θ=10000 (vs Z-Image's 256), axes [32,32,32] (vs [32,48,48]), and Gemma 2 (not Qwen3) text encoder. See diffusers transformer_lumina2.py and Alpha-VLLM/Lumina-Image-2.0 model card.</summary>
public sealed record Lumina2Config
{
    /// <summary>Hidden dimension (2304 for Lumina-Image-2.0).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of query attention heads (24 for Lumina-Image-2.0).</summary>
    public required int NumHeads { get; init; }

    /// <summary>Number of key/value heads. Less than NumHeads for grouped-query attention. Lumina 2.0 uses 8 (3:1 GQA on 24 Q heads).</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Per-head dimension. Default 96 = 2304/24 for Lumina-Image-2.0. Verify by inspecting attn.to_q.weight first dim.</summary>
    public int HeadDim { get; init; } = 96;

    /// <summary>Number of main transformer layers (26 for Lumina-Image-2.0).</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of refiner layers per refiner stack (2 for noise_refiner, 2 for context_refiner — same as Z-Image).</summary>
    public int NumRefinerLayers { get; init; } = 2;

    /// <summary>Latent in-channels (16, matches the Flux VAE that Lumina-Image-2.0 reuses).</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Spatial patch size (2 → 2x2 patchify on the H/W axes).</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Frame patch size (1 — image-only).</summary>
    public int FramePatchSize { get; init; } = 1;

    /// <summary>Caption feature dim into the transformer (2304, matches Gemma 2 2B hidden_size; cap_embedder projects this to HiddenSize).</summary>
    public int CapFeatDim { get; init; } = 2304;

    /// <summary>Modulation embedding dim. Lumina 2.0 sets this to <c>min(hidden_size, 1024) = 1024</c> for hidden=2304 — the inner dim used in time/caption embedders and AdaLN. Z-Image uses a constant 256; Lumina 2.0's larger 1024 is the main scale-up.</summary>
    public int AdaLNEmbedDim { get; init; } = 1024;

    /// <summary>RMSNorm epsilon throughout.</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>RoPE base frequency (10000 for Lumina 2.0; Z-Image uses 256).</summary>
    public float RopeTheta { get; init; } = 10_000.0f;

    /// <summary>Per-axis RoPE dimensions [frame, height, width]. Must sum to HeadDim. Default [32,32,32] for Lumina 2.0 (sums to 96).</summary>
    public int[] AxesDims { get; init; } = [32, 32, 32];

    /// <summary>Maximum precomputable axis lengths [frame, height, width]. Used to cap freq tables.</summary>
    public int[] AxesLens { get; init; } = [300, 512, 512];

    /// <summary>Sinusoidal timestep scale. Lumina 2.0 feeds the inverted timestep <c>(1 - sigma)</c> directly with no extra multiplier — the sinusoidal embedder uses its standard internal frequency table.</summary>
    public float TimestepScale { get; init; } = 1.0f;

    /// <summary>FFN inner dimension. For Lumina-Image-2.0 the diffusers config uses <c>multiple_of=256</c> with no explicit ffn_dim_multiplier — the actual size is determined by the Lumina 2.0 reference: int(8/3 * hidden) rounded up to multiple of 256. For hidden=2304 this is 6144. Verify on load by inspecting <c>layers.0.feed_forward.linear_1.weight</c> shape.</summary>
    public required int FfnDim { get; init; }

    /// <summary>VAE scale factor (Flux VAE: 0.3611). Latents are scaled by (latent - shift) * scale before transformer input.</summary>
    public float VaeScaleFactor { get; init; } = 0.3611f;

    /// <summary>VAE shift factor (Flux VAE: 0.1159).</summary>
    public float VaeShiftFactor { get; init; } = 0.1159f;

    /// <summary>VAE spatial downscale factor (8 for Flux VAE).</summary>
    public int VaeDownscaleFactor { get; init; } = 8;

    /// <summary>Static shift for FlowMatchEulerDiscreteScheduler. Lumina 2.0's diffusers pipeline computes a dynamic shift based on image sequence length (base=0.5, max=1.15); for fixed-resolution generation we use a static default of 6.0 which matches the Z-Image-Base setting and is a reasonable midpoint for 1024×1024.</summary>
    public float SchedulerShift { get; init; } = 6.0f;

    /// <summary>Lumina-Image-2.0 (2B) preset — 26 layers, hidden=2304, GQA 24:8, axes [32,32,32], θ=10000. FfnDim=6144 (8/3 * 2304 rounded to multiple of 256). Verified against <c>Alpha-VLLM/Lumina-Image-2.0/transformer/config.json</c>.</summary>
    public static Lumina2Config V2_0 => new()
    {
        HiddenSize = 2304,
        NumHeads = 24,
        NumKvHeads = 8,
        HeadDim = 96,
        NumLayers = 26,
        NumRefinerLayers = 2,
        FfnDim = 6144,
        SchedulerShift = 6.0f,
    };

    /// <summary>Auto-detects depth, hidden, and FFN dim from a Lumina 2.0 transformer weight dict. Uses the diffusers naming convention (separate <c>attn.to_q/to_k/to_v</c>, <c>feed_forward.linear_1..3</c>).</summary>
    public static Lumina2Config FromWeights(IReadOnlyDictionary<string, Tensor> weights)
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

        int hidden = 2304;
        int qOutDim = 2304;
        if (weights.TryGetValue("layers.0.attn.to_q.weight", out Tensor? toQ))
        {
            hidden = (int)toQ.Shape[1];
            qOutDim = (int)toQ.Shape[0];
        }

        int kvOutDim = qOutDim;
        if (weights.TryGetValue("layers.0.attn.to_k.weight", out Tensor? toK))
            kvOutDim = (int)toK.Shape[0];

        int ffnDim = 6144;
        if (weights.TryGetValue("layers.0.feed_forward.linear_1.weight", out Tensor? linear1))
            ffnDim = (int)linear1.Shape[0];

        // qOutDim = numHeads * headDim, kvOutDim = numKvHeads * headDim. We don't know numHeads vs headDim
        // independently from the weights alone — assume the canonical Lumina 2.0 split (head_dim=96 for the 2B).
        int headDim = 96;
        int numHeads = qOutDim / headDim;
        int numKvHeads = kvOutDim / headDim;

        return new Lumina2Config
        {
            HiddenSize = hidden,
            NumHeads = numHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            NumLayers = numLayers > 0 ? numLayers : 26,
            FfnDim = ffnDim,
        };
    }
}
