namespace SharpInference.Audio.Models.Whisper;

/// <summary>Configuration for a Whisper encoder-decoder model. Covers all released
/// sizes (tiny through large-v3) plus the Sept-2024 large-v3-turbo (distilled to 4
/// decoder layers) and the HuggingFace distil-whisper variants (2-layer decoder).
/// All numbers are verified against the upstream <c>config.json</c> on HuggingFace.</summary>
public sealed record WhisperConfig
{
    /// <summary>Vocabulary size including special tokens. 51865 for &lt;=v2, 51866 for v3+
    /// (v3 added Cantonese, +1 entry).</summary>
    public int VocabSize { get; init; } = 51_865;

    /// <summary>Number of mel bins in the input spectrogram. 80 for &lt;=v2, 128 for v3+.</summary>
    public int NumMelBins { get; init; } = 80;

    /// <summary>Maximum encoder positions (audio context). Fixed at 1500 for all
    /// Whisper sizes — equals 3000 STFT frames after stride-2 conv subsampling.</summary>
    public int MaxAudioPositions { get; init; } = 1500;

    /// <summary>Maximum decoder positions (text context). Fixed at 448 for all sizes.</summary>
    public int MaxTextPositions { get; init; } = 448;

    /// <summary>Encoder layer count.</summary>
    public int EncoderLayers { get; init; } = 6;

    /// <summary>Decoder layer count. Differs from encoder for turbo (4) and distil variants (2).</summary>
    public int DecoderLayers { get; init; } = 6;

    /// <summary>Model width (d_model). Identical for encoder and decoder.</summary>
    public int HiddenSize { get; init; } = 512;

    /// <summary>Number of attention heads. Head dim = HiddenSize / NumHeads, always 64.</summary>
    public int NumHeads { get; init; } = 8;

    /// <summary>Feed-forward inner dim. Always 4 * HiddenSize for stock Whisper.</summary>
    public int IntermediateSize { get; init; } = 2048;

    /// <summary>Layer-norm epsilon. HuggingFace default 1e-5.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Convenience: head dimension. Always 64.</summary>
    public int HeadDim => HiddenSize / NumHeads;

    // ── Special token IDs (constant across all sizes) ──────────────────────────
    /// <summary>End-of-text / pad token. 50257.</summary>
    public int EndOfTextTokenId => 50_257;
    /// <summary>Start-of-transcript token. 50258.</summary>
    public int StartOfTranscriptTokenId => 50_258;
    /// <summary>First language token (English). 50259.</summary>
    public int LanguageTokenStart => 50_259;
    /// <summary>Translate task token. 50358.</summary>
    public int TranslateTokenId => 50_358;
    /// <summary>Transcribe task token. 50359.</summary>
    public int TranscribeTokenId => 50_359;
    /// <summary>No-speech token. 50362.</summary>
    public int NoSpeechTokenId => 50_362;
    /// <summary>No-timestamps token. 50363.</summary>
    public int NoTimestampsTokenId => 50_363;
    /// <summary>First timestamp token (corresponds to 0.00s). 50364. Timestamps run
    /// 50364..51864 in 0.02s steps (1501 tokens, covering 0..30s).</summary>
    public int TimestampTokenStart => 50_364;

    // ── Presets (per OpenAI / HuggingFace config.json) ─────────────────────────

    /// <summary>tiny — 39M params, 4/4 layers, d=384, 6 heads, FFN=1536.</summary>
    public static WhisperConfig Tiny => new()
    {
        VocabSize = 51_865,
        NumMelBins = 80,
        EncoderLayers = 4,
        DecoderLayers = 4,
        HiddenSize = 384,
        NumHeads = 6,
        IntermediateSize = 1_536,
    };

    /// <summary>base — 74M params, 6/6 layers, d=512, 8 heads, FFN=2048.</summary>
    public static WhisperConfig Base => new()
    {
        VocabSize = 51_865,
        NumMelBins = 80,
        EncoderLayers = 6,
        DecoderLayers = 6,
        HiddenSize = 512,
        NumHeads = 8,
        IntermediateSize = 2_048,
    };

    /// <summary>small — 244M params, 12/12 layers, d=768, 12 heads, FFN=3072.</summary>
    public static WhisperConfig Small => new()
    {
        VocabSize = 51_865,
        NumMelBins = 80,
        EncoderLayers = 12,
        DecoderLayers = 12,
        HiddenSize = 768,
        NumHeads = 12,
        IntermediateSize = 3_072,
    };

    /// <summary>medium — 769M params, 24/24 layers, d=1024, 16 heads, FFN=4096.</summary>
    public static WhisperConfig Medium => new()
    {
        VocabSize = 51_865,
        NumMelBins = 80,
        EncoderLayers = 24,
        DecoderLayers = 24,
        HiddenSize = 1_024,
        NumHeads = 16,
        IntermediateSize = 4_096,
    };

    /// <summary>large-v2 — 1.55B params, 32/32 layers, d=1280, 20 heads, FFN=5120.</summary>
    public static WhisperConfig LargeV2 => new()
    {
        VocabSize = 51_865,
        NumMelBins = 80,
        EncoderLayers = 32,
        DecoderLayers = 32,
        HiddenSize = 1_280,
        NumHeads = 20,
        IntermediateSize = 5_120,
    };

    /// <summary>large-v3 — 1.55B params, same shape as v2 but 128 mel bins and +1 vocab (Cantonese).</summary>
    public static WhisperConfig LargeV3 => LargeV2 with
    {
        VocabSize = 51_866,
        NumMelBins = 128,
    };

    /// <summary>large-v3-turbo — distilled to 4 decoder layers; otherwise identical to v3.</summary>
    public static WhisperConfig LargeV3Turbo => LargeV3 with { DecoderLayers = 4 };

    /// <summary>distil-large-v2 — 32-layer encoder, 2-layer decoder, 80 mel bins, English-only.</summary>
    public static WhisperConfig DistilLargeV2 => LargeV2 with { DecoderLayers = 2 };

    /// <summary>distil-large-v3 — 32-layer encoder, 2-layer decoder, 128 mel bins.</summary>
    public static WhisperConfig DistilLargeV3 => LargeV3 with { DecoderLayers = 2 };

    /// <summary>distil-medium.en — 24/2, 80 mel, English-only.</summary>
    public static WhisperConfig DistilMediumEn => Medium with { DecoderLayers = 2 };

    /// <summary>distil-small.en — 12/2, 80 mel, English-only.</summary>
    public static WhisperConfig DistilSmallEn => Small with { DecoderLayers = 2 };
}
