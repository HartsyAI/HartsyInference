namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>Architecture config for ZipVoice (k2-fsa/ZipVoice), a flow-matching zero-shot voice-cloning TTS
/// model built on a "Zipformer" (U-Net-shaped Conformer variant) backbone. Values below are read verbatim
/// from the real <c>k2-fsa/ZipVoice</c> checkpoint's <c>model.json</c> — batch is always 1 (inference-only,
/// no training/padding-mask support needed).</summary>
public sealed record ZipVoiceConfig
{
    // --- fm_decoder (the flow-matching velocity backbone) ---
    public int[] FmDecoderDownsamplingFactor { get; init; } = [1, 2, 4, 2, 1];
    public int[] FmDecoderNumLayers { get; init; } = [2, 2, 4, 4, 4];
    public int[] FmDecoderCnnModuleKernel { get; init; } = [31, 15, 7, 15, 31];
    public int FmDecoderFeedforwardDim { get; init; } = 1536;
    public int FmDecoderNumHeads { get; init; } = 4;
    public int FmDecoderDim { get; init; } = 512;

    // --- text_encoder (unconditional on the FM timestep; single stage, no downsampling) ---
    public int TextEncoderNumLayers { get; init; } = 4;
    public int TextEncoderFeedforwardDim { get; init; } = 512;
    public int TextEncoderCnnModuleKernel { get; init; } = 9;
    public int TextEncoderNumHeads { get; init; } = 4;
    public int TextEncoderDim { get; init; } = 192;

    // --- shared relative-position attention head dims (same for both backbones) ---
    public int QueryHeadDim { get; init; } = 32;
    public int ValueHeadDim { get; init; } = 12;
    public int PosHeadDim { get; init; } = 4;
    public int PosDim { get; init; } = 48;

    public int TimeEmbedDim { get; init; } = 192;
    public int TextEmbedDim { get; init; } = 192;
    public int FeatDim { get; init; } = 100;
    public int VocabSize { get; init; } = 360;

    public int SamplingRate { get; init; } = 24000;

    /// <summary>Mel/vocoder-feature scale applied around the model (upstream <c>feat_scale</c>): reference
    /// mel is multiplied by this before conditioning; predicted mel is divided by it before the vocoder.</summary>
    public float FeatScale { get; init; } = 0.1f;

    public static ZipVoiceConfig Default => new();

    /// <summary>Default sampling knobs for the base (non-distilled) <c>zipvoice</c> checkpoint.</summary>
    public static class BaseSampling
    {
        public const int NumSteps = 16;
        public const float GuidanceScale = 1.0f;
        public const float TShift = 0.5f;
    }
}
