using HartsyInference.Audio.Models.Codecs.Mimi;

namespace HartsyInference.Audio.Models.BreezeTts;

/// <summary>Configuration contract for Breeze TTS 2 (checkpoint <c>config.json</c> + <c>text_encoder_config</c>). A
/// T5Gemma2 text encoder feeds a Qwen3 backbone that predicts the first of 16 codebooks per 12.5 Hz frame, and a CSM-style
/// depth decoder predicts the other 15. Reference audio is encoded by Mimi (first 16 of 32 quantizers) and frames are
/// decoded by the Qwen3-TTS-Tokenizer-12Hz vocoder shipped in the checkpoint's <c>audio_tokenizer</c> folder.</summary>
public sealed record BreezeTts2Config
{
    public int HiddenSize { get; init; } = 2_048;
    public int NumBackboneLayers { get; init; } = 28;
    public int NumBackboneHeads { get; init; } = 16;
    public int NumBackboneKeyValueHeads { get; init; } = 8;
    public int BackboneHeadDim { get; init; } = 128;
    public int BackboneIntermediateSize { get; init; } = 6_144;
    public float BackboneRmsNormEps { get; init; } = 1e-6f;
    public float BackboneRopeTheta { get; init; } = 1_000_000f;
    public int NumDepthDecoderLayers { get; init; } = 12;
    public int DepthDecoderHiddenSize { get; init; } = 1_024;
    public int DepthDecoderIntermediateSize { get; init; } = 8_192;
    public int DepthDecoderHeads { get; init; } = 8;
    public int DepthDecoderKeyValueHeads { get; init; } = 2;
    public int DepthDecoderHeadDim { get; init; } = 128;
    public int DepthDecoderMaxPositions { get; init; } = 33;
    public float DepthDecoderRmsNormEps { get; init; } = 1e-5f;
    public float DepthDecoderRopeTheta { get; init; } = 500_000f;
    public int TextEncoderHiddenSize { get; init; } = 1_152;
    public int TextEncoderLayers { get; init; } = 26;
    public int TextEncoderVocabSize { get; init; } = 262_158;
    public T5Gemma2TextEncoderConfig TextEncoder { get; init; } = T5Gemma2TextEncoderConfig.Breeze;
    public int NumCodebooks { get; init; } = 16;
    /// <summary>Mimi encoder preset used for reference audio (32 quantizers; only the first <see cref="NumCodebooks"/> are used).</summary>
    public MimiConfig Codec { get; init; } = MimiConfig.Mimi24kHzDsm;
    public float CodecFrameRate { get; init; } = 12.5f;
    public int AudioTokenId { get; init; } = 262_144;
    public int AudioEosTokenId { get; init; } = 262_145;
    /// <summary>Per-codebook embedding/logit width: 2048 codes + EOS/PAD specials.</summary>
    public int AudioVocabSize { get; init; } = 2_051;
    public int CodebookSize { get; init; } = 2_048;
    public int CodebookEosTokenId { get; init; } = 0;
    public int CodebookPadTokenId { get; init; } = 2_050;
    public bool SupportsVoiceDesign { get; init; } = true;
    public bool SupportsVoiceDirection { get; init; } = true;
    public bool SupportsVoiceClone { get; init; } = true;
    public bool SupportsEnglishAndChinese { get; init; } = true;

    // Sampling defaults (breeze_infer/runtime.py + FastStreamingConfig).
    public float Temperature { get; init; } = 0.9f;
    public int TopK { get; init; } = 50;
    public float TopP { get; init; } = 1.0f;
    public float RepetitionPenalty { get; init; } = 1.1f;
    public int MaxNewFrames { get; init; } = 1_500;
    public int MaxSequenceLength { get; init; } = 2_048;

    public static BreezeTts2Config Default => new();
}
