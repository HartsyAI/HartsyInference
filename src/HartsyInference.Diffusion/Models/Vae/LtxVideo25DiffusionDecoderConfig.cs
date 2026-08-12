namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Geometry of the LTX-2.5 <c>NADiffusionDecoder</c>. Defaults are the shipped 2.5 video VAE, read off the
/// real checkpoint's metadata; <see cref="LtxVideo25DiffusionDecoder.LoadWeights"/> re-checks every value against the
/// weight shapes, so a future variant fails loudly instead of decoding garbage.</summary>
public sealed record LtxVideo25DiffusionDecoderConfig
{
    /// <summary>Latent channels entering <c>conv_in</c>.</summary>
    public int InChannels { get; init; } = 128;

    /// <summary>RGB channels leaving <c>conv_out</c> after unpatchify.</summary>
    public int OutChannels { get; init; } = 3;

    /// <summary>Spatial patch folded into the stage-5 channel axis; also the last spatial upscale factor.</summary>
    public int PatchSize { get; init; } = 4;

    public int HeadDim { get; init; } = 64;

    /// <summary>Channels of the four deterministic stages plus the stage-5 diffusion trunk.</summary>
    public int[] StageChannels { get; init; } = [2048, 1024, 512, 512, 256];

    /// <summary>Block counts per stage; the last entry is the number of <c>diff_blocks</c>.</summary>
    public int[] StageDepths { get; init; } = [4, 6, 4, 2, 8];

    /// <summary>Neighborhood-attention window per deterministic stage (the fifth entry is unused — stage 5 uses
    /// <see cref="Stage5Kernel"/>) — kept at index 4 so the array lines up with the reference config.</summary>
    public (int T, int H, int W)[] StageKernels { get; init; } =
        [(3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5), (11, 11, 11)];

    /// <summary>Per-upsample pixel-shuffle stride and the channel-reduction divisor applied on top of it.</summary>
    public ((int T, int H, int W) Stride, int Reduction)[] Upsamples { get; init; } =
        [((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2)];

    public (int T, int H, int W) Stage5Kernel { get; init; } = (11, 11, 11);

    public int TimestepEmbedDim { get; init; } = 384;

    /// <summary>Width of the sinusoidal embedding feeding <c>t_embedder.mlp.0</c>.</summary>
    public int TimestepFreqDim { get; init; } = 256;

    /// <summary>The scheduler timestep is multiplied by this before the sinusoidal embedding.</summary>
    public float TimestepScaleMultiplier { get; init; } = 1000f;

    public float TimestepMaxPeriod { get; init; } = 10000f;

    /// <summary>Denoising steps. The shipped checkpoint is single-step <c>x0</c>, so one forward pass is the result.</summary>
    public int NumInferenceSteps { get; init; } = 1;

    public float RopeBase { get; init; } = 10000f;

    public float NormEps { get; init; } = 1e-6f;

    /// <summary>SwiGLU hidden width, rounded up to a multiple of 16 exactly as the reference does.</summary>
    public static int SwiGluHidden(int dim, float mlpRatio = 4f) => ((int)(dim * mlpRatio) + 15) / 16 * 16;

    /// <summary>Product of the temporal upsample strides (8 for the shipped config).</summary>
    public int TemporalUpscale
    {
        get
        {
            int product = 1;
            foreach (((int T, int H, int W) stride, int _) in Upsamples) product *= stride.T;
            return product;
        }
    }

    /// <summary>Product of the spatial upsample strides times <see cref="PatchSize"/> (32 for the shipped config).</summary>
    public int SpatialUpscale
    {
        get
        {
            int product = PatchSize;
            foreach (((int T, int H, int W) stride, int _) in Upsamples) product *= stride.H;
            return product;
        }
    }

    /// <summary>Latent frames replicated past the end before stages 1-4, then cropped off the context. NATTEN slides
    /// its window inward at the border, so the true last frames would otherwise be attended asymmetrically.</summary>
    public int TrailingPadLatentFrames => StageKernels[0].T / 2 * 2;

    /// <summary>Output frames for a latent frame count: every temporally-strided upsample drops its duplicated
    /// leading frame, so <c>t → 2t−1</c> per such stage (8t−7 for the shipped config's three).</summary>
    public int OutputFrames(int latentFrames)
    {
        int frames = latentFrames;
        foreach (((int T, int H, int W) stride, int _) in Upsamples)
        {
            frames *= stride.T;
            if (stride.T == 2) frames -= 1;
        }
        return frames;
    }

    /// <summary>Per-axis RoPE dimension split <c>(t, h, w)</c> of one head, matching the reference's
    /// <c>default_rope_dim_split</c>. All three chunks must be positive and even.</summary>
    public static (int T, int H, int W) RopeDimSplit(int headDim)
    {
        if (headDim % 8 != 0)
            throw new ArgumentException($"head_dim {headDim} must be a multiple of 8 for the default RoPE split.", nameof(headDim));
        int dimT = headDim / 4 / 2 * 2;
        int dimHw = (headDim - dimT) / 2;
        if (dimHw % 2 != 0)
        {
            dimT -= 2;
            dimHw = (headDim - dimT) / 2;
        }
        if (dimT <= 0 || dimHw <= 0)
            throw new ArgumentException($"head_dim {headDim} yields a degenerate RoPE split ({dimT}, {dimHw}, {dimHw}).", nameof(headDim));
        return (dimT, dimHw, dimHw);
    }

    /// <summary>Inverse RoPE frequencies <c>1/base^(i/dim)</c> for even <c>i</c>, computed in double like the
    /// reference's float64 path before the f32 cast.</summary>
    public static float[] RopeInverseFrequencies(int dim, float baseValue)
    {
        float[] inverse = new float[dim / 2];
        for (int i = 0; i < inverse.Length; i++)
            inverse[i] = (float)(1.0 / Math.Pow(baseValue, 2.0 * i / dim));
        return inverse;
    }
}
