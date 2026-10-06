namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Geometry of the laion_clap audio tower ControlFoley uses for its reference-audio feature
/// (<c>CLAP_Module(enable_fusion=False, amodel='HTSAT-base')</c>): an HTS-AT Swin transformer over a 64-bin log-mel image,
/// then the <c>audio_projection</c> MLP and L2 normalisation.</summary>
public sealed record ControlFoleyClapConfig
{
    /// <summary>Patch embedding width; each stage doubles it.</summary>
    public int EmbedDim { get; init; } = 128;

    /// <summary>Swin blocks per stage.</summary>
    public IReadOnlyList<int> Depths { get; init; } = [2, 2, 12, 2];

    /// <summary>Attention heads per stage.</summary>
    public IReadOnlyList<int> Heads { get; init; } = [4, 8, 16, 32];

    /// <summary>Joint embedding width (output of <c>audio_projection</c>).</summary>
    public int JointDim { get; init; } = 512;

    /// <summary>Attention window edge in patches.</summary>
    public int WindowSize { get; init; } = 8;

    /// <summary>Edge of the square spectrogram image fed to the patch embedding.</summary>
    public int SpecSize { get; init; } = 256;

    /// <summary>Patch edge and stride of the embedding convolution.</summary>
    public int PatchSize { get; init; } = 4;

    /// <summary>Nominal rate the mel front end is parameterised for; ControlFoley feeds 16 kHz audio regardless, as the official code does.</summary>
    public int SampleRate { get; init; } = 48_000;

    /// <summary>STFT size.</summary>
    public int NFft { get; init; } = 1_024;

    /// <summary>STFT hop.</summary>
    public int Hop { get; init; } = 480;

    /// <summary>Mel bins.</summary>
    public int MelBins { get; init; } = 64;

    /// <summary>Lowest mel filter edge in Hz.</summary>
    public double Fmin { get; init; } = 50.0;

    /// <summary>Highest mel filter edge in Hz.</summary>
    public double Fmax { get; init; } = 14_000.0;

    /// <summary>Waveform length every clip is repeat-padded or truncated to (10 s at the nominal rate).</summary>
    public int ClipSamples { get; init; } = 480_000;

    /// <summary>Width of the last stage, which the pooled embedding has.</summary>
    public int FinalDim => EmbedDim << (Depths.Count - 1);

    /// <summary>The released HTSAT-base tower.</summary>
    public static ControlFoleyClapConfig HtsatBase => new();
}
