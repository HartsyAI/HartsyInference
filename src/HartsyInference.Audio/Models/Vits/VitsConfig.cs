namespace HartsyInference.Audio.Models.Vits;

/// <summary>Configuration for a VITS <c>SynthesizerTrn</c> (the shared end-to-end TTS core behind Piper,
/// MeloTTS, GPT-SoVITS's SoVITS half, and OpenVoice). Inference path: phonemes → TextEncoder (relative-pos
/// MHA) → (m_p, logs_p) → duration predictor → monotonic length regulation → prior sample → normalizing
/// flow (reverse) → HiFi-GAN decoder → waveform. See <c>docs/Research/VITS_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> all convs/transposed-convs/LayerNorm via <see cref="HartsyInference.Core.Backends.IBackend"/>,
/// attention via <c>ScaledDotProductAttention</c> (+ rel-pos bias), weight-norm via <c>WeightNormFusion</c>,
/// seeded noise via <c>DeterministicRng</c>. Net-new: rel-pos attention, WN residual coupling, the monotonic
/// length regulator, and the stochastic duration predictor's spline (staged).</para></summary>
public sealed record VitsConfig
{
    public int NumVocab { get; init; } = 256;
    public int InterChannels { get; init; } = 192;     // == TextEncoder out channels / flow channels
    public int HiddenChannels { get; init; } = 192;
    public int FilterChannels { get; init; } = 768;
    public int NumHeads { get; init; } = 2;
    public int NumEncoderLayers { get; init; } = 6;
    public int EncoderKernelSize { get; init; } = 3;
    public int WindowSize { get; init; } = 4;           // relative-position clip ±4

    // ── Duration predictor ──
    public bool UseSdp { get; init; } = true;
    public int DpFilterChannels { get; init; } = 256;   // deterministic DP
    public int DpKernelSize { get; init; } = 3;
    public int SdpFlows { get; init; } = 4;
    /// <summary>State-dict prefix of the stochastic duration predictor. Real VITS/Piper checkpoints store it under
    /// <c>dp</c>; the synthetic-weight tests use <c>sdp</c>, which stays the default.</summary>
    public string SdpPrefix { get; init; } = "sdp";

    // ── Flow ──
    public int FlowLayers { get; init; } = 4;           // WN layers per coupling
    public int FlowFlows { get; init; } = 4;            // coupling+flip pairs
    public int FlowKernelSize { get; init; } = 5;
    public int FlowDilationRate { get; init; } = 1;

    // ── HiFi-GAN decoder ──
    public string ResBlock { get; init; } = "2";        // medium
    public IReadOnlyList<int> ResBlockKernelSizes { get; init; } = [3, 5, 7];
    public IReadOnlyList<IReadOnlyList<int>> ResBlockDilations { get; init; } = [[1, 2], [2, 6], [3, 12]];
    public IReadOnlyList<int> UpsampleRates { get; init; } = [8, 8, 4];     // ∏ = 256 = hop
    public int UpsampleInitialChannel { get; init; } = 256;                 // Piper "medium" (high uses 512)
    public IReadOnlyList<int> UpsampleKernelSizes { get; init; } = [16, 16, 8];

    public int GinChannels { get; init; } = 0;          // 0 = single-speaker
    public int NumSpeakers { get; init; } = 1;
    public int SampleRate { get; init; } = 22_050;

    // ── Inference scales ──
    public float NoiseScale { get; init; } = 0.667f;
    public float LengthScale { get; init; } = 1.0f;
    public float NoiseScaleW { get; init; } = 0.8f;

    public int HopLength
    {
        get { int p = 1; for (int i = 0; i < UpsampleRates.Count; i++) p *= UpsampleRates[i]; return p; }
    }

    /// <summary>Piper "medium" preset (22.05 kHz, resblock 2, 3-stage [8,8,4] upsampler, SDP). Piper stores the
    /// stochastic duration predictor under the <c>dp</c> prefix.</summary>
    public static VitsConfig PiperMedium => new() { SdpPrefix = "dp" };

    /// <summary>Piper "high" preset (resblock 1, 4-stage upsampler).</summary>
    public static VitsConfig PiperHigh => new()
    {
        SdpPrefix = "dp",
        ResBlock = "1",
        ResBlockKernelSizes = [3, 7, 11],
        ResBlockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        UpsampleRates = [8, 8, 2, 2],
        UpsampleKernelSizes = [16, 16, 4, 4],
        UpsampleInitialChannel = 512,
    };
}
