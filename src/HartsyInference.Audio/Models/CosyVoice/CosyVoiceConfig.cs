using HartsyInference.Audio.Models.LanguageModels.Qwen2;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>Top-level configuration for the CosyVoice 2 (`FunAudioLLM/CosyVoice2-0.5B`) text-to-speech
/// pipeline. CosyVoice 2 is a four-stage system: Qwen2.5-0.5B text→speech-token LM → FSQ speech tokens →
/// chunk-aware conditional flow matching (speech-token→mel) → HiFTNet vocoder (mel→24 kHz waveform),
/// with a CAM++ 192-dim speaker embedding injected only at the flow-matching stage. See
/// <c>docs/Research/COSYVOICE_ARCHITECTURE.md</c>.
///
/// <para><b>LM structure (matches the real <c>llm.pt</c>, not the research doc's simplified
/// "single extended-vocab softmax"):</b> the Qwen2.5 backbone keeps its stock text embedding; speech
/// tokens use a <i>separate</i> <c>speech_embedding</c> on the input side and a separate
/// <c>llm_decoder</c> Linear on the output side (the Qwen <c>lm_head</c> is unused for speech). Two
/// extra control embeddings (<c>llm_embedding</c>: sos/eos + task) bracket the text/speech boundary.</para></summary>
public sealed record CosyVoiceConfig
{
    /// <summary>Qwen2.5-0.5B backbone config (24 layers / 896 hidden / 14:2 GQA / RoPE θ=1e6). The
    /// backbone keeps the stock Qwen text vocab; speech tokens are handled by the separate
    /// <c>speech_embedding</c> / <c>llm_decoder</c> heads (see <see cref="SpeechTokenSize"/>).</summary>
    public required Qwen2Config Llm { get; init; }

    /// <summary>Number of FSQ speech-token codes — <c>3^8 = 6561</c> for CosyVoice 2.</summary>
    public int SpeechTokenSize { get; init; } = 6_561;

    /// <summary>Extra speech-token vocab slots appended after the 6561 codes (sos/eos/task markers on
    /// the speech side). The <c>llm_decoder</c> output dim and <c>speech_embedding</c> rows are
    /// <c>SpeechTokenSize + SpeechTokenExtra</c>. The end-of-speech token ID is <see cref="SpeechTokenSize"/>.</summary>
    public int SpeechTokenExtra { get; init; } = 3;

    /// <summary>Speech-token frame rate (Hz). 25 Hz for CV2 — 25 tokens ≈ 1 second of audio.</summary>
    public int TokenRateHz { get; init; } = 25;

    /// <summary>Output waveform sample rate. 24 kHz for CV2.</summary>
    public int SampleRate { get; init; } = 24_000;

    /// <summary>Conditional-flow-matching (speech-token → mel) config.</summary>
    public CosyVoiceFlowConfig Flow { get; init; } = new();

    /// <summary>HiFTNet vocoder (mel → waveform) config.</summary>
    public CosyVoiceHiftConfig Hift { get; init; } = new();

    /// <summary>Default LM sampling knobs (autoregressive speech-token generation).</summary>
    public CosyVoiceSamplingConfig Sampling { get; init; } = new();

    /// <summary>Streaming text:speech interleave ratio (5 text tokens → 15 speech tokens) and the
    /// per-chunk speech-token count that drives first-packet latency. <c>0</c> chunk size selects
    /// non-streaming (emit the turn-of-speech marker only after all text).</summary>
    public int StreamingTextChunk { get; init; } = 5;
    public int StreamingSpeechChunk { get; init; } = 15;

    /// <summary>`FunAudioLLM/CosyVoice2-0.5B` preset.</summary>
    public static CosyVoiceConfig V2_0_5B => new()
    {
        // Qwen2.5-0.5B backbone. VocabSize here is the stock Qwen text vocab; the actual loaded
        // embed_tokens may carry a few appended text specials (e.g. <|endofprompt|>) — LoadWeights
        // reads the real tensor shape, so this is informational for the text side only.
        Llm = Qwen2Config.Qwen25_0_5B,
        SpeechTokenSize = 6_561,
        SpeechTokenExtra = 3,
        TokenRateHz = 25,
        SampleRate = 24_000,
    };
}

/// <summary>Conditional flow-matching (OT-CFM) config for the speech-token → mel stage. Vanilla
/// first-order Euler (no sway-sampling / omega-shift); CFG weight 0.7; 10 NFE. The estimator is a
/// small Matcha-TTS-style UNet1D wrapped in a chunk-aware causal transformer encoder.</summary>
public sealed record CosyVoiceFlowConfig
{
    /// <summary>Output mel bins. 80.</summary>
    public int MelBins { get; init; } = 80;

    /// <summary>Mel frame rate (Hz). 50 — the flow upsamples the 25 Hz token stream ~2× in time.</summary>
    public int MelFrameRateHz { get; init; } = 50;

    /// <summary>Speech-token embedding dim fed to the flow encoder (output_size of the upstream LM /
    /// the flow's input projection). 512 in CV2's flow.pt.</summary>
    public int InputSize { get; init; } = 512;

    /// <summary>UNet1D estimator channel widths per level. [256, 256] in CV2.</summary>
    public int[] UnetChannels { get; init; } = [256, 256];

    /// <summary>Attention head dim inside the estimator. 64.</summary>
    public int AttentionHeadDim { get; init; } = 64;

    /// <summary>Attention heads. 8.</summary>
    public int NumHeads { get; init; } = 8;

    /// <summary>ResNet blocks per UNet level. 4.</summary>
    public int NumBlocks { get; init; } = 4;

    /// <summary>Mid-stack blocks. 12.</summary>
    public int NumMidBlocks { get; init; } = 12;

    /// <summary>CAM++ speaker-embedding dim projected into the flow conditioning. 192 → mel.</summary>
    public int SpeakerEmbedDim { get; init; } = 192;

    /// <summary>Number of Euler ODE steps at inference.</summary>
    public int NumEulerSteps { get; init; } = 10;

    /// <summary>Classifier-free-guidance weight (<c>inference_cfg_rate</c>).</summary>
    public float CfgRate { get; init; } = 0.7f;
}

/// <summary>HiFTNet vocoder config (mel → 24 kHz waveform). Harmonic-plus-noise source filter + iSTFT
/// output stage — structurally the same family as the Kokoro iSTFTNet decoder, with an added internal
/// F0 predictor feeding the sinusoidal source.</summary>
public sealed record CosyVoiceHiftConfig
{
    public int MelBins { get; init; } = 80;
    public int SampleRate { get; init; } = 24_000;
    public int UpsampleInitialChannel { get; init; } = 512;
    public int[] UpsampleRates { get; init; } = [8, 5, 3];
    public int[] UpsampleKernelSizes { get; init; } = [16, 11, 7];
    public int[] ResBlockKernelSizes { get; init; } = [3, 7, 11];
    public int[][] ResBlockDilationSizes { get; init; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    /// <summary>iSTFT synthesis FFT size + hop for the output head.</summary>
    public int IstftNFft { get; init; } = 16;
    public int IstftHopSize { get; init; } = 4;

    /// <summary>Number of harmonics in the NSF source module.</summary>
    public int HarmonicNum { get; init; } = 8;
}

/// <summary>LM autoregressive speech-token sampling defaults (CosyVoice 2).</summary>
public sealed record CosyVoiceSamplingConfig
{
    public float Temperature { get; init; } = 0.8f;
    public int TopK { get; init; } = 25;
    public float TopP { get; init; } = 0.8f;
    public float RepetitionPenalty { get; init; } = 1.1f;

    /// <summary>Repetition-Aware Sampling: detect a degenerate speech-token loop and re-roll with a
    /// flatter distribution. Window + max-repeat thresholds mirror <c>cosyvoice/llm/llm.py</c>.</summary>
    public bool UseRas { get; init; } = true;
    public int RasWindow { get; init; } = 10;
    public int RasMaxRepeat { get; init; } = 4;
}
