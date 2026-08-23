using HartsyInference.Audio.Preprocessing;

namespace HartsyInference.Audio.Models.Rvc;

/// <summary>Configuration for <see cref="RvcRmvpe"/>. Defaults follow the standard RMVPE: 16 kHz in, 128-mel
/// front-end (n_fft = win_length 1024 / hop 160, ~100 Hz frame rate), a 360-bin cents grid whose first bin sits
/// at 31.70 Hz (cents base 1997.3794, freq = 10 * 2^(cents/1200)) with 20-cent spacing. The UNet channel schedule
/// (16→32→64→128→256→512, 5 stages) is hardcoded in <see cref="RvcRmvpe"/> itself, matching the real
/// <c>E2E(4,1,(2,2))</c> — verified corr 1.000000 / maxAbs 9.5e-8 against real <c>rmvpe.pt</c>
/// (<c>docs/Checklists/PARITY_VERIFICATION.md</c>).</summary>
public sealed record RvcRmvpeConfig
{
    public int SampleRate { get; init; } = 16_000;
    public int MelBins { get; init; } = 128;
    public int NFft { get; init; } = 1_024;
    public int HopLength { get; init; } = 160;
    public int WinLength { get; init; } = 1_024;
    public double Fmin { get; init; } = 30.0;
    public double Fmax { get; init; } = 8_000.0;

    /// <summary>Number of pitch-classification bins.</summary>
    public int NumBins { get; init; } = 360;

    /// <summary>Frequency of the first (lowest) pitch bin (cents base 1997.3794 -> 31.70 Hz).</summary>
    public float FirstFreqHz { get; init; } = 31.70f;

    /// <summary>Cents between adjacent bins (360 bins × 20 cents ≈ 6 octaves).</summary>
    public float CentsPerBin { get; init; } = 20f;

    /// <summary>Half-width (in bins) of the local weighted-average decode window.</summary>
    public int LocalAverageWindow { get; init; } = 4;

    /// <summary>Posterior peak below which a frame is reported unvoiced (0 Hz). RMVPE uses 0.03
    /// (<c>to_local_average_cents</c> thred); 0.3 was a 10x error that silenced most voiced frames.</summary>
    public float VoicingThreshold { get; init; } = 0.03f;

    // RVC-WebUI's rmvpe.py MelSpectrogram builds its filterbank via librosa.filters.mel(..., htk=True) and
    // runs a centered STFT — omitting Scale/Center here silently fell back to the record's Slaney/no-center
    // defaults (the same bug class F5VocosConfig's doc comment already documents: "F5 output was distorted
    // before this existed"), which shifts every mel bin's center frequency and biases the network's pitch
    // estimate by a large, consistent amount rather than adding noise.
    public MelSpectrogramExtractor.Config MelConfig() => new(
        SampleRate: SampleRate,
        NFft: NFft,
        WinLength: WinLength,
        HopLength: HopLength,
        NMels: MelBins,
        Fmin: Fmin,
        Fmax: Fmax,
        Norm: MelSpectrogramExtractor.Normalization.None,
        DropLastStftFrame: false,
        LogBase: MelSpectrogramExtractor.LogBase.Natural,
        LogFloor: 1e-5f,
        DynamicRangeDb: 0f,
        NormOffset: 0f,
        NormScale: 1f,
        PowerSpectrum: false,
        Scale: MelScale.Htk,
        SlaneyNorm: false,
        Center: true);

    public static RvcRmvpeConfig Default => new();
}
