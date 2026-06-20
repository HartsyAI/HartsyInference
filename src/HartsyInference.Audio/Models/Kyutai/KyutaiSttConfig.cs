using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;

namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Configuration for Kyutai's delayed-streams speech-to-text models (`kyutai/stt-1b-en_fr`,
/// `kyutai/stt-2.6b-en`). A "Helium" Llama-family transformer reads <see cref="MimiConfig">Mimi</see> audio
/// codes (32 codebooks @ 12.5 Hz) and emits one text token per frame, with the audio↔text offset realized as
/// silence padding around the waveform. See <c>docs/Research/KYUTAI_DSM_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> the backbone is a plain Helium decoder → <see cref="Qwen2Model"/> with
/// <see cref="Qwen2Config.AttentionBias"/> off (driven headless via <c>ForwardEmbeds</c> because the
/// embedding is the shared text+audio table); audio encode reuses the built Mimi. The only net-new logic is
/// the shared-table summed input embedding and the silence-padding delay.</para></summary>
public sealed record KyutaiSttConfig
{
    /// <summary>Helium backbone (Llama-family; attention bias off, MHA, SwiGLU, RMSNorm, RoPE).</summary>
    public required Qwen2Config Helium { get; init; }

    /// <summary>Mimi codec config — the DSM 32-codebook variant.</summary>
    public MimiConfig Codec { get; init; } = MimiConfig.Mimi24kHzDsm;

    /// <summary>Number of Mimi codebooks consumed per frame.</summary>
    public int NumCodebooks { get; init; } = 32;

    /// <summary>Per-codebook audio vocab in the shared embedding table (2048 codes + 1 pad/BOS slot).</summary>
    public int CodebookVocab { get; init; } = 2_049;

    /// <summary>Text vocabulary size — the leading rows of the shared table and the sampling range.</summary>
    public int TextVocab { get; init; } = 4_001;

    /// <summary>Text PAD token id (emitted on frames with no word); stripped from the transcript.</summary>
    public int TextPad { get; init; } = 3;

    /// <summary>Audio BOS / initial code id fed before real codes appear.</summary>
    public int AudioBos { get; init; } = 2_048;

    /// <summary>Silence prepended before the audio, in seconds (left delay).</summary>
    public float AudioSilencePrefixSeconds { get; init; } = 1.0f;

    /// <summary>Silence appended after the audio, in seconds — the audio↔text delay.</summary>
    public float AudioDelaySeconds { get; init; } = 2.5f;

    /// <summary>Mimi frame rate in Hz (12.5 in the Moshi design).</summary>
    public float FrameRate { get; init; } = 12.5f;

    /// <summary>Row offset of codebook <paramref name="k"/>'s codes in the shared embedding table.</summary>
    public int AudioOffset(int k) => TextVocab + k * CodebookVocab;

    /// <summary>stt-2.6b-en preset (English). Helium: 2048 / 48L / 32 heads (head_dim 64) / SwiGLU 11264 /
    /// RoPE θ=100000 / RMSNorm 1e-8; text vocab 4001. <b>Reconcile on first checkpoint load:</b> the gated
    /// MLP ships as fused <c>fc1</c>/<c>fc2</c> (split into gate|up for our Qwen2 layer), the shared
    /// embedding key is <c>model.embed_tokens.embed_tokens.weight</c>, and the WORD-boundary token id.</summary>
    public static KyutaiSttConfig Stt2_6B => new()
    {
        Helium = HeliumBase with { NumHiddenLayers = 48, NumAttentionHeads = 32, NumKeyValueHeads = 32, VocabSize = 4_001, MaxPositionEmbeddings = 750 },
        TextVocab = 4_001,
        AudioSilencePrefixSeconds = 1.0f,
        AudioDelaySeconds = 2.5f,
    };

    /// <summary>stt-1b-en_fr preset (English + French). Helium: 2048 / 16L / 16 heads (head_dim 128) /
    /// SwiGLU 11264; text vocab 8001; 0.5 s audio delay, no silence prefix.</summary>
    public static KyutaiSttConfig Stt1B => new()
    {
        Helium = HeliumBase with { NumHiddenLayers = 16, NumAttentionHeads = 16, NumKeyValueHeads = 16, VocabSize = 8_001, MaxPositionEmbeddings = 375 },
        TextVocab = 8_001,
        AudioSilencePrefixSeconds = 0.0f,
        AudioDelaySeconds = 0.5f,
    };

    private static Qwen2Config HeliumBase => new()
    {
        HiddenSize = 2_048,
        NumHiddenLayers = 16,
        NumAttentionHeads = 16,
        NumKeyValueHeads = 16,
        IntermediateSize = 11_264,
        VocabSize = 8_001,
        MaxPositionEmbeddings = 750,
        RopeTheta = 100_000f,
        RmsNormEps = 1e-8f,
        AttentionBias = false,
        TieWordEmbeddings = false,
    };
}
