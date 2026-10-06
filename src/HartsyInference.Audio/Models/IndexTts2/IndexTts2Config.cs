using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.LanguageModels.Gpt;

namespace HartsyInference.Audio.Models.IndexTts2;

/// <summary>IndexTTS-2.5 hyperparameters, hand-ported from the real <c>config.yaml</c> (no YAML dependency in
/// this solution, same posture as every other model's static preset). 2.5-only for now — 2.0 shares every one
/// of these values except <see cref="NumberTextTokens"/> and the semantic codec's downsample scale, but its
/// own GPT speaker-conditioning path (<c>condition_type: conformer_perceiver</c>, not 2.5's
/// <c>campplus</c>/<c>spk_emb_proj</c>) isn't implemented yet — see <see cref="IndexTts2T2sDecoder"/>'s own
/// remarks. Adding a <c>V2_0</c> preset before that decoder variant exists would be scaffolding nothing can
/// run, so it is deliberately not here yet.</summary>
public sealed record IndexTts2Config
{
    public required GptConfig Gpt { get; init; }
    public required int NumberTextTokens { get; init; }
    public required int MaxTextTokens { get; init; }

    /// <summary>Real <c>gpt.max_mel_tokens</c> (1815) — the GPT's own trained mel-code generation cap,
    /// distinct from <see cref="GptConfig.BlockSize"/> (2417 = this plus <see cref="MaxTextTokens"/> + 2).</summary>
    public required int MaxMelTokens { get; init; }
    public required VocosFactorizedCodecConfig SemanticCodec { get; init; }
    public required IndexTts2DitConfig S2MelDit { get; init; }
    public required IndexTts2BigVganConfig BigVgan { get; init; }

    /// <summary>Real <c>s2mel.length_regulator</c>: working channel width (512), source feature width (1024 —
    /// the w2v-bert/semantic-codec hidden size), and stage count (<c>len(sampling_ratios)</c>, 4).</summary>
    public required int LengthRegulatorChannels { get; init; }
    public required int LengthRegulatorInChannels { get; init; }
    public required int LengthRegulatorNumStages { get; init; }

    /// <summary>Output sample rate (22050 — the stock BigVGAN-22kHz vocoder's rate, NOT IndexTTS-1.5's 24000).</summary>
    public required int SampleRate { get; init; }

    /// <summary>Real <c>infer_v2_5.py</c>'s fixed semantic-codec-frames→mel-frames ratio
    /// (<c>target_lengths = S_infer.shape[1] * 1.72 * duration_factor</c>) — accounts for the semantic codec's
    /// own frame rate vs. the S2Mel stage's 22050 Hz/256-hop mel rate. Not derived from any other config field;
    /// confirmed as a literal constant in the real source.</summary>
    public required float ContentLengthRatio { get; init; }

    /// <summary>Real <c>diffusion_steps</c> (25) and <c>inference_cfg_rate</c> (0.7) — the S2Mel CFM's own
    /// defaults, confirmed from the real call site (different from <c>BASECFM</c>'s own in-code default of 0.5).</summary>
    public required int DiffusionSteps { get; init; }
    public required float InferenceCfgRate { get; init; }

    public static IndexTts2Config V2_5 => new()
    {
        Gpt = GptConfig.IndexTts2,
        NumberTextTokens = 60_509,
        MaxTextTokens = 600,
        MaxMelTokens = 1_815,
        SemanticCodec = VocosFactorizedCodecConfig.IndexTts2V5,
        S2MelDit = IndexTts2DitConfig.IndexTts2,
        BigVgan = IndexTts2BigVganConfig.Nvidia22kHz80Band,
        LengthRegulatorChannels = 512,
        LengthRegulatorInChannels = 1_024,
        LengthRegulatorNumStages = 4,
        SampleRate = 22_050,
        ContentLengthRatio = 1.72f,
        DiffusionSteps = 25,
        InferenceCfgRate = 0.7f,
    };
}
