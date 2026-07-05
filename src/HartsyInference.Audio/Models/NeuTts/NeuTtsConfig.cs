using HartsyInference.Audio.Models.Codecs.NeuCodec;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;

namespace HartsyInference.Audio.Models.NeuTts;

/// <summary>Configuration for NeuTTS Air (`neuphonic/neutts-air`) — a Qwen2.5-0.5B LM (vocab extended to
/// 217,652 with 65,536 <c>&lt;|speech_N|&gt;</c> tokens) that emits a single NeuCodec FSQ stream, decoded to
/// 24 kHz audio. Voice cloning conditions on reference NeuCodec codes + reference text in the prompt.
///
/// <para><b>Reuse:</b> the backbone is stock Qwen2.5-0.5B → <see cref="Qwen2Model"/> with the extended vocab;
/// sampling reuses <c>NucleusSampler</c> (top-k=50, temp=1.0); audio decode is <see cref="NeuCodecDecoder"/>.
/// The only model-specific logic is the speech-token offset and the generation framing.</para></summary>
public sealed record NeuTtsConfig
{
    public required Qwen2Config Llm { get; init; }
    public NeuCodecConfig Codec { get; init; } = NeuCodecConfig.Default;

    /// <summary>LM token id of NeuCodec FSQ code 0; code <c>c</c> → token <c>SpeechTokenBase + c</c>.</summary>
    public int SpeechTokenBase { get; init; } = 151_671;
    public int CodebookSize { get; init; } = 65_536;

    public int TextPromptStart { get; init; } = 151_666;
    public int TextPromptEnd { get; init; } = 151_667;
    public int SpeechGenStart { get; init; } = 151_669;
    /// <summary>Generation stop token (<c>&lt;|SPEECH_GENERATION_END|&gt;</c>).</summary>
    public int SpeechGenEnd { get; init; } = 151_670;

    /// <summary>LM token id used as the text-replace placeholder in the prompt framing.</summary>
    public int TextReplace { get; init; } = 151_665;
    /// <summary>LM token id used as the speech-replace placeholder in the prompt framing.</summary>
    public int SpeechReplace { get; init; } = 151_668;
    /// <summary>LM token id of the checkpoint's first appended IPA phoneme token (<see cref="PhonemeTokens"/>[0]);
    /// these sit ABOVE the speech-code block and must be matched before byte-level BPE.</summary>
    public int PhonemeTokenBase { get; init; } = 217_207;
    /// <summary>The checkpoint's appended IPA phoneme tokens, in id order from <see cref="PhonemeTokenBase"/>.</summary>
    public IReadOnlyList<string> PhonemeTokens { get; init; } = NeuTtsPhonemeTokens.Air;
    /// <summary>Maximum prompt + AR generation length (upstream <c>max_context</c>, generate <c>max_length</c>).</summary>
    public int MaxContext { get; init; } = 2_048;

    // ── Sampling (matches neutts.py _infer_torch, NOT generation_config.json) ──
    public float Temperature { get; init; } = 1.0f;
    public int TopK { get; init; } = 50;
    public float TopP { get; init; } = 0f;
    /// <summary>Minimum generated tokens before EOS is allowed (suppresses early stop). Upstream applies this
    /// to a reference-conditioned sequence; on the unconditioned default-voice path a flat floor forces short
    /// clips past their natural stop (trailing babble), so callers may lower it via the Synthesize overload.</summary>
    public int MinNewTokens { get; init; } = 50;

    /// <summary>Repetition penalty over recently-emitted codes (HF semantics; 1.0 = off). Defends the
    /// unconditioned default-voice path against degenerate loops / trailing garble.</summary>
    public float RepetitionPenalty { get; init; } = 1.1f;
    /// <summary>Number of most-recent emitted tokens the repetition penalty considers.</summary>
    public int RepetitionWindow { get; init; } = 64;

    /// <summary>NeuTTS Air preset — Qwen2.5-0.5B with the 217,652 extended vocab and tied head.</summary>
    public static NeuTtsConfig Air => new()
    {
        Llm = Qwen2Config.Qwen25_0_5B with { VocabSize = 217_652 },
    };

    /// <summary>NeuTTS Nano preset (the reference DEFAULT) — a Llama-3 backbone (hidden 576 / 24L / 9 query
    /// heads / 3 KV heads MHA-via-GQA / head_dim 64 / SwiGLU 2304 / RoPE θ=500000 / no bias / tied head),
    /// vocab 194,256, with Llama-3 linear RoPE scaling (factor 32) applied through the shared
    /// <c>Core.Rope</c> path. The speech/text control tokens use the nano token block.</summary>
    public static NeuTtsConfig Nano => new()
    {
        Llm = new Qwen2Config
        {
            HiddenSize = 576,
            NumHiddenLayers = 24,
            NumAttentionHeads = 9,
            NumKeyValueHeads = 3,
            IntermediateSize = 2_304,
            VocabSize = 194_256,
            MaxPositionEmbeddings = 2_048,
            RopeTheta = 500_000f,
            RopeScaling = new HartsyInference.Core.Rope.RopeScaling
            {
                Type = HartsyInference.Core.Rope.RopeScalingType.Llama3,
                Factor = 32.0,
                LowFreqFactor = 1.0,
                HighFreqFactor = 4.0,
            },
            AttentionBias = false,
            TieWordEmbeddings = true,
        },
        SpeechTokenBase = 128_262,
        SpeechGenStart = 128_260,
        SpeechGenEnd = 128_261,
        TextPromptStart = 128_257,
        TextPromptEnd = 128_258,
        TextReplace = 128_256,
        SpeechReplace = 128_259,
        PhonemeTokenBase = 193_798,
        PhonemeTokens = NeuTtsPhonemeTokens.Nano,
    };
}
