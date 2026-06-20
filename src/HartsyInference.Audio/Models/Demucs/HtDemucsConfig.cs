namespace HartsyInference.Audio.Models.Demucs;

/// <summary>Configuration for HTDemucs (Hybrid Transformer Demucs v4) music source separation — a dual-branch
/// U-Net (time 1D + spectrogram 2D, complex-as-channels) joined by a cross-domain transformer at the
/// bottleneck, decoding to 4 stereo stems (drums, bass, other, vocals). See
/// <c>docs/Research/HTDEMUCS_ARCHITECTURE.md</c>. Stereo 44.1 kHz in.</summary>
public sealed record HtDemucsConfig
{
    public int AudioChannels { get; init; } = 2;
    public IReadOnlyList<string> Sources { get; init; } = ["drums", "bass", "other", "vocals"];
    public int Channels { get; init; } = 48;     // base hidden; ×growth per depth
    public int Growth { get; init; } = 2;
    public int Depth { get; init; } = 4;
    public int NFft { get; init; } = 4_096;
    public int HopLength { get; init; } = 1_024;
    public int KernelSize { get; init; } = 8;
    public int Stride { get; init; } = 4;
    public int BottomChannels { get; init; } = 512;   // transformer width
    public int TLayers { get; init; } = 5;
    public int THeads { get; init; } = 8;
    public float HiddenScale { get; init; } = 4.0f;   // FFN = dim*4 = 2048
    public float FreqEmbScale { get; init; } = 0.2f;
    public float NormEps { get; init; } = 1e-5f;
    public int SampleRate { get; init; } = 44_100;

    public int NumSources => Sources.Count;
    public int TransformerFfn => (int)(BottomChannels * HiddenScale);
    public int TransformerHeadDim => BottomChannels / THeads;

    /// <summary>Spectrogram-branch input channels = 2 (real+imag) × audio channels.</summary>
    public int SpecInChannels => 2 * AudioChannels;

    public static HtDemucsConfig Htdemucs => new();
}
