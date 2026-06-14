using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Chroma Radiance (<c>lodestones/Chroma1-Radiance</c>) — the Chroma backbone operating
/// directly in pixel space. The transformer body is byte-identical to classic Chroma (19 double + 38 single blocks,
/// hidden 3072, distilled-guidance approximator); only the IO ends differ: a <c>Conv2d(3→3072, k16, s16)</c> input
/// patchifier replaces <c>img_in</c>/VAE, and a per-patch hypernetwork "NeRF head" replaces <c>final_layer</c>.
/// The model predicts <b>x0</b> (the clean image), not velocity. See docs/Research/CHROMA_RADIANCE_ARCHITECTURE.md.</summary>
public sealed record ChromaRadianceConfig
{
    /// <summary>Chroma backbone configuration (unchanged from classic Chroma v1).</summary>
    public required ChromaConfig Backbone { get; init; }

    /// <summary>Pixel patch size of the conv patchifier (16 → one transformer token per 16×16 RGB patch).</summary>
    public int PatchSize { get; init; } = 16;

    /// <summary>Per-pixel embedding width inside the NeRF head (64).</summary>
    public int NerfHidden { get; init; } = 64;

    /// <summary>Number of hypernetwork GLU blocks in the NeRF head (4).</summary>
    public int NerfDepth { get; init; } = 4;

    /// <summary>Cosine positional basis frequency count per axis (8 → 8² = 64 features per pixel).</summary>
    public int MaxFreqs { get; init; } = 8;

    /// <summary>GLU expansion ratio inside each NeRF block (4 → inner dim 256 = NerfHidden × 4).</summary>
    public int NerfMlpRatio { get; init; } = 4;

    /// <summary>Default CFG scale (3.5 per the reference release).</summary>
    public float DefaultCfgScale { get; init; } = 3.5f;

    /// <summary>Default inference steps (50).</summary>
    public int DefaultSteps { get; init; } = 50;

    /// <summary>Flow-match Euler static shift (mu = 1.0 — validation-gated, see research doc item 5).</summary>
    public float SchedulerShift { get; init; } = 1.0f;

    /// <summary>Chroma1-Radiance preset.</summary>
    public static ChromaRadianceConfig V1 => new() { Backbone = ChromaConfig.V1 };

    /// <summary>True when a converted Chroma-family weight dict is a Radiance checkpoint (NeRF head present).</summary>
    public static bool IsRadiance(IReadOnlyDictionary<string, Tensor> weights)
        => weights.ContainsKey("nerf_blocks.0.norm.scale");

    /// <summary>Builds a config from converted weights, inferring NeRF head dims from tensor shapes. Backbone dims
    /// stay at the V1 preset (the only published Radiance release).</summary>
    public static ChromaRadianceConfig FromWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        int patchSize = 16;
        if (weights.TryGetValue("img_in_patch.weight", out Tensor? patchW) && patchW.Shape.Rank == 4)
            patchSize = (int)patchW.Shape[2];

        int nerfHidden = 64;
        int maxFreqs = 8;
        if (weights.TryGetValue("nerf_image_embedder.embedder.0.weight", out Tensor? embedW))
        {
            nerfHidden = (int)embedW.Shape[0];
            // in_dim = 3 (rgb) + maxFreqs² positional features.
            int posFeatures = (int)embedW.Shape[1] - 3;
            int freqs = (int)Math.Round(Math.Sqrt(posFeatures));
            if (freqs * freqs == posFeatures && freqs > 0)
                maxFreqs = freqs;
        }

        int nerfDepth = 0;
        while (weights.ContainsKey($"nerf_blocks.{nerfDepth}.norm.scale"))
            nerfDepth++;
        if (nerfDepth == 0) nerfDepth = 4;

        return new ChromaRadianceConfig
        {
            Backbone = ChromaConfig.V1,
            PatchSize = patchSize,
            NerfHidden = nerfHidden,
            MaxFreqs = maxFreqs,
            NerfDepth = nerfDepth,
        };
    }
}
