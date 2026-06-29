using HartsyInference.Audio.Models.Codecs.Dac;

namespace HartsyInference.Audio.Models.Dia;

/// <summary>Configuration for Dia-1.6B (Nari Labs) — a T5/Whisper-style encoder-decoder TTS that turns
/// byte-level text into a 9-codebook DAC code grid (delay pattern) decoded to 44.1 kHz audio. The decoder
/// cross-attends to the encoder, predicts all 9 codebooks per step (summed input embeddings + one fused
/// head), and generates with classifier-free guidance. See <c>docs/Research/DIA_TTS_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> DAC decode is the built <see cref="DacConfig.Dac44kHz"/>; delay staging is the shared
/// <c>MusicGenDelay</c> (BOS/PAD overload); sampling is <c>NucleusSampler</c>. Net-new: the non-causal
/// encoder and the decoder cross-attention sublayer (the first cross-attn transformer in the Audio
/// package).</para></summary>
public sealed record DiaConfig
{
    // ── Encoder ──
    public int EncoderLayers { get; init; } = 12;
    public int EncoderDim { get; init; } = 1_024;
    public int EncoderHeads { get; init; } = 16;
    public int EncoderKvHeads { get; init; } = 16;     // full MHA
    public int EncoderFfn { get; init; } = 4_096;

    // ── Decoder ──
    public int DecoderLayers { get; init; } = 18;
    public int DecoderDim { get; init; } = 2_048;
    public int DecoderSelfQHeads { get; init; } = 16;
    public int DecoderSelfKvHeads { get; init; } = 4;  // GQA
    public int DecoderCrossQHeads { get; init; } = 16;
    public int DecoderCrossKvHeads { get; init; } = 16;
    public int DecoderFfn { get; init; } = 8_192;

    /// <summary>Per-head dim for every attention flavor (note <c>heads*HeadDim != dim</c> — QKV up-project).</summary>
    public int HeadDim { get; init; } = 128;

    // ── Vocabulary / channels ──
    public int TextVocab { get; init; } = 256;         // raw UTF-8 bytes
    public int TextPad { get; init; } = 0;
    public int Channels { get; init; } = 9;            // DAC codebooks
    public int AudioVocab { get; init; } = 1_028;      // 1024 codes + EOS/PAD/BOS
    public int AudioEos { get; init; } = 1_024;
    public int AudioPad { get; init; } = 1_025;
    public int AudioBos { get; init; } = 1_026;

    public IReadOnlyList<int> DelayPattern { get; init; } = [0, 8, 9, 10, 11, 12, 13, 14, 15];
    public int MaxDelay => 15;

    public float RopeTheta { get; init; } = 10_000f;
    /// <summary>Minimum RoPE timescale; matches the Dia reference config (rope_min_timescale = 1).</summary>
    public int RopeMinTimescale { get; init; } = 1;
    public float NormEps { get; init; } = 1e-5f;
    public int MaxText { get; init; } = 1_024;
    public int MaxAudio { get; init; } = 3_072;

    // ── Generation ──
    public float CfgScale { get; init; } = 3.0f;
    public float Temperature { get; init; } = 1.2f;
    public float TopP { get; init; } = 0.95f;
    public int TopK { get; init; } = 45;

    // Dia decodes through the descript-native DAC-44kHz checkpoint, whose WNConvTranspose1d uses dim-0
    // weight-norm (weight_g sized C_in), so compose those over dim 0.
    public DacConfig Codec { get; init; } = DacConfig.Dac44kHz with { TransposeWeightNormDim0 = true };

    public static DiaConfig Dia1_6B => new();
}
