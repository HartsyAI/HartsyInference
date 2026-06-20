using HartsyInference.Audio.Models.Codecs.Dac;

namespace HartsyInference.Audio.Models.Zonos;

/// <summary>Configuration for Zyphra's Zonos-v0.1 (transformer variant). A Llama-style GQA decoder
/// (but <b>LayerNorm</b>, not RMSNorm) generates a 9-codebook DAC 44.1 kHz grid (delay pattern) conditioned
/// on a prefix of espeak phonemes + speaker + fine-grained controls (emotion / fmax / pitch / rate /
/// language). See <c>docs/Research/ZONOS_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> the attention (GQA, interleaved RoPE, no bias) and the fused-gate-up SwiGLU MLP reuse
/// <c>DiaAttention</c> / <c>DiaMlp</c>; DAC decode is the built <see cref="DacConfig.Dac44kHz"/>; delay is the
/// shared <c>MusicGenDelay</c>; sampling is <c>NucleusSampler</c> (min-p). Net-new: the LayerNorm block, the
/// conditioning prefix (Fourier/Integer/Passthrough conditioners), and the multi-codebook embed/head.</para></summary>
public sealed record ZonosConfig
{
    // ── Backbone ──
    public int Hidden { get; init; } = 2_048;
    public int NumLayers { get; init; } = 26;
    public int NumHeads { get; init; } = 16;
    public int NumKvHeads { get; init; } = 4;        // GQA 4:1
    public int HeadDim { get; init; } = 128;
    public int FfnIntermediate { get; init; } = 8_192;
    public float NormEps { get; init; } = 1e-5f;     // LayerNorm
    public float RopeTheta { get; init; } = 10_000f; // interleaved
    public int MaxPositions { get; init; } = 16_384;

    // ── Codec / codebooks ──
    public int Channels { get; init; } = 9;
    /// <summary>Per-codebook input vocab: 1024 codes + EOS(1024) + masked(1025).</summary>
    public int InputVocab { get; init; } = 1_026;
    /// <summary>Per-codebook output vocab: 1024 codes + EOS(1024).</summary>
    public int OutputVocab { get; init; } = 1_025;
    public int EosToken { get; init; } = 1_024;
    public int MaskedToken { get; init; } = 1_025;
    public DacConfig Codec { get; init; } = DacConfig.Dac44kHz;

    /// <summary>Zonos delays codebook k by k+1 → effective [1,2,…,9].</summary>
    public int[] BuildDelays()
    {
        int[] d = new int[Channels];
        for (int k = 0; k < Channels; k++) d[k] = k + 1;
        return d;
    }

    // ── Conditioning ──
    public int SpeakerDim { get; init; } = 128;
    public int EmotionDim { get; init; } = 8;
    public int NumLanguages { get; init; } = 105;
    public float FmaxMax { get; init; } = 24_000f;
    public float PitchStdMax { get; init; } = 400f;
    public float SpeakingRateMax { get; init; } = 40f;

    // ── Generation ──
    public float CfgScale { get; init; } = 2.0f;
    public float Temperature { get; init; } = 1.0f;
    public float MinP { get; init; } = 0.1f;
    public float RepetitionPenalty { get; init; } = 3.0f;
    public int RepetitionWindow { get; init; } = 2;
    public int MaxNewTokens { get; init; } = 2_580;   // 86 * 30 ≈ 30 s

    public static ZonosConfig V0_1Transformer => new();
}
