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

    // ── Sampling (matches neutts.py _infer_torch, NOT generation_config.json) ──
    public float Temperature { get; init; } = 1.0f;
    public int TopK { get; init; } = 50;
    public float TopP { get; init; } = 0f;
    /// <summary>Minimum generated tokens before EOS is allowed (suppresses early stop).</summary>
    public int MinNewTokens { get; init; } = 50;

    /// <summary>NeuTTS Air preset — Qwen2.5-0.5B with the 217,652 extended vocab and tied head.</summary>
    public static NeuTtsConfig Air => new()
    {
        Llm = Qwen2Config.Qwen25_0_5B with { VocabSize = 217_652 },
    };
}
