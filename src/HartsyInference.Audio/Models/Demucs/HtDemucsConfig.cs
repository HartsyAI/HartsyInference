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

    /// <summary>Scale applied to the frequency embedding before it is added to the spectrogram features.</summary>
    public float EmbScale { get; init; } = 10f;
    /// <summary>Whether to smooth (normalize) the frequency embedding.</summary>
    public bool EmbSmooth { get; init; } = true;
    /// <summary>Stride of the time branch first convolution.</summary>
    public int TimeStride { get; init; } = 2;
    /// <summary>Context (kernel-1) of the decoder rewrite convolution.</summary>
    public int Context { get; init; } = 1;
    /// <summary>Context (kernel-1) of the encoder rewrite convolution.</summary>
    public int ContextEnc { get; init; } = 0;
    /// <summary>Encoder depth index at which group normalization starts being applied.</summary>
    public int NormStarts { get; init; } = 4;
    /// <summary>Number of groups for group normalization.</summary>
    public int NormGroups { get; init; } = 4;
    /// <summary>Whether the 1x1 rewrite convolutions are enabled in encoder/decoder blocks.</summary>
    public bool Rewrite { get; init; } = true;
    /// <summary>Number of depths over which the multi-frequency (freq-collapse) DConv is applied.</summary>
    public int MultiFreqsDepth { get; init; } = 3;
    /// <summary>Depth (number of branches) of the residual DConv blocks.</summary>
    public int DConvDepth { get; init; } = 2;
    /// <summary>Channel compression factor inside the residual DConv blocks.</summary>
    public int DConvComp { get; init; } = 8;
    /// <summary>Initial value for the DConv layer-scale parameter.</summary>
    public float DConvInit { get; init; } = 1e-3f;
    /// <summary>Whether complex-as-channels (CaC) representation is used for the spectrogram branch.</summary>
    public bool Cac { get; init; } = true;
    /// <summary>Number of Wiener-filter EM iterations applied at output (0 disables Wiener filtering).</summary>
    public int WienerIters { get; init; } = 0;
    /// <summary>Transformer positional-embedding type ("sin" for sinusoidal).</summary>
    public string TEmb { get; init; } = "sin";
    /// <summary>Maximum period used by the transformer sinusoidal positional embedding.</summary>
    public int TMaxPeriod { get; init; } = 10000;
    /// <summary>Whether the transformer applies normalization on its input.</summary>
    public bool TNormIn { get; init; } = true;
    /// <summary>Whether the transformer normalizes before (pre-norm) each sublayer.</summary>
    public bool TNormFirst { get; init; } = true;
    /// <summary>Whether the transformer applies an output normalization.</summary>
    public bool TNormOut { get; init; } = true;
    /// <summary>Whether the transformer uses per-layer learnable layer-scale.</summary>
    public bool TLayerScale { get; init; } = true;
    /// <summary>Whether the transformer feed-forward uses GELU activation.</summary>
    public bool TGelu { get; init; } = true;

    public int NumSources => Sources.Count;
    public int TransformerFfn => (int)(BottomChannels * HiddenScale);
    public int TransformerHeadDim => BottomChannels / THeads;

    /// <summary>Spectrogram-branch input channels = 2 (real+imag) × audio channels.</summary>
    public int SpecInChannels => 2 * AudioChannels;

    public static HtDemucsConfig Htdemucs => new();

    /// <summary>Fine-tuned HTDemucs (htdemucs_ft): same architecture as the default 4-source model.</summary>
    public static HtDemucsConfig HtdemucsFt => new();

    /// <summary>6-source HTDemucs (htdemucs_6s): adds guitar and piano stems.</summary>
    public static HtDemucsConfig Htdemucs6s => new()
    {
        Sources = ["drums", "bass", "other", "vocals", "guitar", "piano"],
    };
}
