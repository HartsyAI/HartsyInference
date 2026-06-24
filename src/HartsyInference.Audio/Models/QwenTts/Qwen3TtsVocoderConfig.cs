namespace HartsyInference.Audio.Models.QwenTts;

/// <summary>Configuration for the Qwen3-TTS custom codec DECODER (net-new Snake/ConvNeXt causal-conv vocoder).
/// Split-RVQ dequant (1 semantic + 15 acoustic) → pre_conv → 8-layer causal sliding-window transformer →
/// [2,2] ConvNeXt upsample → DAC-style decoder over upsample_rates [8,5,4,3]. Total upsample
/// 8·5·4·3·2·2 = 1920 → 12.5 Hz frames to 24 kHz. See <c>docs/Research/QWEN3_TTS_ARCHITECTURE.md</c>.</summary>
public sealed record Qwen3TtsVocoderConfig
{
    // ── Split-RVQ ───────────────────────────────────────────────────────
    /// <summary>Semantic codebook size (codebook 0).</summary>
    public int SemanticCodebookSize { get; init; } = 4_096;

    /// <summary>Semantic codebook embedding dim.</summary>
    public int SemanticCodebookDim { get; init; } = 256;

    /// <summary>Acoustic codebook size (codebooks 1..15).</summary>
    public int AcousticCodebookSize { get; init; } = 2_048;

    /// <summary>Acoustic codebook embedding dim (same 256 as semantic in the real Tokenizer-V2 codec).</summary>
    public int AcousticCodebookDim { get; init; } = 256;

    /// <summary>Number of acoustic codebooks.</summary>
    public int AcousticCodebooks { get; init; } = 15;

    /// <summary>Latent width after the quantizer output projection (== pre_conv input channels).</summary>
    public int LatentDim { get; init; } = 512;

    // ── pre_conv ────────────────────────────────────────────────────────
    public int PreConvOut { get; init; } = 1_024;
    public int PreConvKernel { get; init; } = 3;

    // ── Causal sliding-window transformer ───────────────────────────────
    public int TransformerDim { get; init; } = 512;
    public int TransformerLayers { get; init; } = 8;
    public int TransformerHeads { get; init; } = 16;
    public int TransformerHeadDim { get; init; } = 64;
    public float TransformerRopeTheta { get; init; } = 10_000f;
    public int SlidingWindow { get; init; } = 72;
    public float LayerScaleInit { get; init; } = 0.01f;
    public int TransformerFfnDim { get; init; } = 1_024;
    public float RmsNormEps { get; init; } = 1e-6f;

    // ── Upsample [2,2] ConvNeXt stage ───────────────────────────────────
    public IReadOnlyList<int> ConvNeXtUpsampleRates { get; init; } = [2, 2];
    public int ConvNeXtDim { get; init; } = 1_024;

    // ── DAC-style decoder ───────────────────────────────────────────────
    public int DecoderInKernel { get; init; } = 7;
    public int DecoderInChannels { get; init; } = 1_536;
    public IReadOnlyList<int> UpsampleRates { get; init; } = [8, 5, 4, 3];
    public IReadOnlyList<int> ResidualDilations { get; init; } = [1, 3, 9];
    public int ResidualKernel { get; init; } = 7;
    public int OutKernel { get; init; } = 7;
    public int SampleRate { get; init; } = 24_000;

    /// <summary>Total upsample factor — product of every stride in the stack (must hit 1920).</summary>
    public int TotalUpsample
    {
        get
        {
            int p = 1;
            for (int i = 0; i < ConvNeXtUpsampleRates.Count; i++) p *= ConvNeXtUpsampleRates[i];
            for (int i = 0; i < UpsampleRates.Count; i++) p *= UpsampleRates[i];
            return p;
        }
    }

    public static Qwen3TtsVocoderConfig Default => new();
}
