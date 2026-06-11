namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for ACE-Step v1.5 turbo (2B) — the Qwen3-block flow-matching music DiT over Oobleck-VAE
/// latents (25 Hz, 64-ch, 48 kHz stereo). Defaults verbatim from <c>configuration_acestep_v15.py</c>; the fixed
/// 8-step turbo timestep tables come from the shipped <c>SHIFT_TIMESTEPS</c>. See
/// <c>docs/Research/ACE_STEP_15_INFERENCE.md</c>.</summary>
public sealed record AceStep15Config
{
    /// <summary>Model width (= heads · head dim).</summary>
    public int HiddenSize { get; init; } = 2048;

    /// <summary>DiT layers.</summary>
    public int NumLayers { get; init; } = 24;

    /// <summary>Query heads.</summary>
    public int NumHeads { get; init; } = 16;

    /// <summary>Key/value heads (GQA 16:8).</summary>
    public int NumKvHeads { get; init; } = 8;

    /// <summary>Per-head dim.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>SwiGLU MLP inner dim.</summary>
    public int IntermediateSize { get; init; } = 6144;

    /// <summary>RMSNorm epsilon.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>RoPE base.</summary>
    public double RopeTheta { get; init; } = 1_000_000.0;

    /// <summary>Bidirectional sliding-attention window (token distance) on alternating layers.</summary>
    public int SlidingWindow { get; init; } = 128;

    /// <summary>DiT input channels: latent · 3 (src ‖ chunk mask ‖ noisy latent — reference concat order).</summary>
    public int InChannels { get; init; } = 192;

    /// <summary>Conv1d patchify kernel/stride (25 Hz latents → 12.5 Hz tokens).</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Oobleck latent channels (DiT output channels).</summary>
    public int LatentChannels { get; init; } = 64;

    /// <summary>Qwen3-Embedding-0.6B hidden width (text AND lyric features).</summary>
    public int TextHiddenDim { get; init; } = 1024;

    /// <summary>Reference-audio acoustic latent width (Oobleck latents).</summary>
    public int TimbreHiddenDim { get; init; } = 64;

    /// <summary>Lyric encoder depth.</summary>
    public int LyricEncoderLayers { get; init; } = 8;

    /// <summary>Timbre encoder depth.</summary>
    public int TimbreEncoderLayers { get; init; } = 4;

    /// <summary>Timestep sinusoid width (cos-first, t scaled ×1000).</summary>
    public int FreqDim { get; init; } = 256;

    /// <summary>Latent frames per second.</summary>
    public int LatentRate { get; init; } = 25;

    /// <summary>Output sample rate.</summary>
    public int SampleRate { get; init; } = 48_000;

    /// <summary>PCM samples per latent frame (Oobleck 2·4·4·6·10).</summary>
    public int SamplesPerLatent { get; init; } = 1920;

    /// <summary>Turbo step count (fixed table, no CFG).</summary>
    public int NumInferenceSteps { get; init; } = 8;

    /// <summary>Default timestep-table shift.</summary>
    public float FlowShift { get; init; } = 3.0f;

    /// <summary>True for layers using the 128-token sliding window: even indices (the reference computes
    /// <c>"sliding_attention" if (i + 1) % 2 else "full_attention"</c>, so layer 0 slides, layer 1 is full, …).
    /// Shared by the DiT and both condition encoders.</summary>
    public bool IsSlidingLayer(int layerIndex) => (layerIndex + 1) % 2 == 1;

    /// <summary>Latent frames for a duration at 25 Hz.</summary>
    public int LatentFrames(double durationSeconds) => (int)Math.Round(durationSeconds * LatentRate);

    /// <summary>The fixed turbo timestep table (8 descending values; the final step integrates to 0). The shift is
    /// snapped to the nearest of the reference's <c>VALID_SHIFTS</c> [1, 2, 3].</summary>
    public static float[] GetTimesteps(float shift)
    {
        float snapped = shift < 1.5f ? 1f : shift < 2.5f ? 2f : 3f;
        return snapped switch
        {
            1f => [1f, 0.875f, 0.75f, 0.625f, 0.5f, 0.375f, 0.25f, 0.125f],
            2f => [1f, 0.9333333333f, 0.8571428571f, 0.7692307692f, 0.6666666666f, 0.5454545454f, 0.4f, 0.2222222222f],
            _ => [1f, 0.9545454545f, 0.9f, 0.8333333333f, 0.75f, 0.6428571428f, 0.5f, 0.3f],
        };
    }

    /// <summary>The published v1.5 turbo 2B model.</summary>
    public static AceStep15Config Turbo => new();
}
