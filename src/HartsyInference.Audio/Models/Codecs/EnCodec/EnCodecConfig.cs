namespace HartsyInference.Audio.Models.Codecs.EnCodec;

/// <summary>Configuration for Meta's EnCodec neural audio codec
/// (<c>facebookresearch/encodec</c>, Défossez et al. 2022). Defaults match the
/// <c>encodec_24khz</c> checkpoint that ships with Bark (24 kHz mono, 1024-entry codebooks).
/// MusicGen mono uses the distinct <c>encodec_32khz</c> codec (<see cref="EnCodec32kHz"/>) and
/// AudioGen uses <c>encodec_16khz</c> (<see cref="EnCodec16kHz"/>); both are non-causal with
/// 2048-entry codebooks and a 50 Hz token rate.
///
/// <para>The 24 kHz mono variant downsamples by <c>8*5*4*2 = 320</c> → 75 Hz latent frame
/// rate, with up to 32 residual-VQ codebooks of 1024 entries each. Operational
/// bandwidth is set per generation via <see cref="ActiveBandwidthKbps"/> which selects
/// the leading <c>N</c> codebooks (each codebook = ~0.75 kbps).</para></summary>
public sealed record EnCodecConfig
{
    /// <summary>Audio sample rate. 24000 for the 24 kHz model; 48000 for the multi-band
    /// variant (not yet supported here).</summary>
    public int SampleRate { get; init; } = 24_000;

    /// <summary>Audio channels — 1 (mono) for the Bark / MusicGen 24 kHz model.</summary>
    public int Channels { get; init; } = 1;

    /// <summary>Latent dimension between the SEANet stack and the residual VQ. 128 in
    /// the published 24 kHz checkpoint.</summary>
    public int LatentDim { get; init; } = 128;

    /// <summary>SEANet stem filter count. Channel widths follow <c>n_filters * 2^stage</c>
    /// up to <c>n_filters * 2^len(ratios) = 512</c> at the bottleneck.</summary>
    public int NFilters { get; init; } = 32;

    /// <summary>Downsample / upsample ratios in encoder-forward order. Product = 320
    /// (24 kHz → 75 Hz). The encoder reverses this list internally so the largest
    /// downsample runs LAST (highest channel count).</summary>
    public IReadOnlyList<int> Ratios { get; init; } = [8, 5, 4, 2];

    /// <summary>SEANet residual block count per downsample / upsample stage. 1 in the
    /// published checkpoints.</summary>
    public int NResidualLayers { get; init; } = 1;

    /// <summary>Kernel size for the SEANet stem and final projection conv. 7 in the
    /// published checkpoints.</summary>
    public int KernelSize { get; init; } = 7;

    /// <summary>Last conv kernel — same as <see cref="KernelSize"/> for EnCodec.</summary>
    public int LastKernelSize { get; init; } = 7;

    /// <summary>Conv1d kernel used inside the SEANet residual block. 3 in the published
    /// checkpoints.</summary>
    public int ResidualKernelSize { get; init; } = 3;

    /// <summary>Dilation base. Per-block dilation grows as <c>dilation_base^j</c> for
    /// <c>j</c> in <c>[0, n_residual_layers)</c>. With <see cref="NResidualLayers"/>=1
    /// only the j=0 dilation (=1) is used.</summary>
    public int DilationBase { get; init; } = 2;

    /// <summary>Bottleneck compression inside the SEANet residual block. The first
    /// 1d-conv reduces dim by <c>compress</c>, the second restores it. 2 in EnCodec.</summary>
    public int Compress { get; init; } = 2;

    /// <summary>LSTM bottleneck layer count between the SEANet encoder/decoder stages
    /// and the latent projection. 2 in the published checkpoints.</summary>
    public int LstmLayers { get; init; } = 2;

    /// <summary>Whether all convolutions use causal padding (left-only). True for the
    /// 24 kHz model. False for the symmetric (offline) 48 kHz multi-band variant.</summary>
    public bool Causal { get; init; } = true;

    /// <summary>Conv pad mode — <c>"constant"</c> (zero) or <c>"reflect"</c>. EnCodec
    /// uses <c>"reflect"</c> by default, but the streaming-compatible models switch to
    /// <c>"constant"</c>. We default to <c>"constant"</c> for portability.</summary>
    public string PadMode { get; init; } = "constant";

    /// <summary>Layer-norm style on the convs. <c>"weight_norm"</c> in EnCodec (we fuse
    /// at load time so this is purely informational). <c>"none"</c> for codecs that
    /// don't reparametrize.</summary>
    public string Norm { get; init; } = "weight_norm";

    /// <summary>ELU alpha. 1.0 in the published checkpoints.</summary>
    public float EluAlpha { get; init; } = 1.0f;

    /// <summary>Residual-VQ codebook size. 1024 per codebook in the published
    /// 24 kHz checkpoint.</summary>
    public int VqCodebookSize { get; init; } = 1024;

    /// <summary>Maximum residual-VQ codebook count. The 24 kHz checkpoint ships 32
    /// codebooks corresponding to a 24 kbps maximum bandwidth.</summary>
    public int VqMaxCodebooks { get; init; } = 32;

    /// <summary>Per-codebook bandwidth contribution in kbps. <c>= bins_log2 * frame_rate / 1000</c>
    /// → <c>log2(1024) * 75 / 1000 = 0.75</c> kbps per codebook.</summary>
    public double KbpsPerCodebook { get; init; } = 0.75;

    /// <summary>Active codebook count for inference. Bark uses 8 (6 kbps); MusicGen
    /// uses 4 codebooks (3 kbps) for the mono model. Override via the pipeline
    /// when loading.</summary>
    public int ActiveCodebooks { get; init; } = 8;

    /// <summary>Convenience: the latent frame rate (downsampled sample rate).
    /// 24000 / 320 = 75 Hz for the published config.</summary>
    public int FrameRate
    {
        get
        {
            int product = 1;
            for (int i = 0; i < Ratios.Count; i++) product *= Ratios[i];
            return SampleRate / product;
        }
    }

    /// <summary>EnCodec 24 kHz preset matching <c>facebook/encodec_24khz</c>.</summary>
    public static EnCodecConfig EnCodec24kHz => new();

    /// <summary>EnCodec 32 kHz preset matching <c>facebook/encodec_32khz</c> — the codec MusicGen mono
    /// consumes. Non-causal SEANet (<c>n_filters=64</c>), ratios <c>[8,5,4,4]</c> → 640× downsample → 50 Hz,
    /// 4 codebooks of 2048 entries (2.2 kbps). See AUDIO_CODECS.md "EnCodec 32k".</summary>
    public static EnCodecConfig EnCodec32kHz => new()
    {
        SampleRate = 32_000,
        NFilters = 64,
        Ratios = [8, 5, 4, 4],
        Causal = false,
        VqCodebookSize = 2_048,
        VqMaxCodebooks = 4,
        ActiveCodebooks = 4,
        KbpsPerCodebook = 0.55,
    };

    /// <summary>EnCodec 16 kHz preset matching <c>facebook/encodec_16khz</c> — the codec AudioGen consumes.
    /// Non-causal SEANet, ratios <c>[8,5,4,2]</c> → 320× downsample → 50 Hz, 4 codebooks of 2048 entries.
    /// <c>NFilters</c> is the one value to reconcile against the checkpoint on first load (64 assumed, matching
    /// the 32 kHz mono and Mimi 16 kHz codecs). See AUDIO_CODECS.md "EnCodec 16k".</summary>
    public static EnCodecConfig EnCodec16kHz => new()
    {
        SampleRate = 16_000,
        NFilters = 64,
        Ratios = [8, 5, 4, 2],
        Causal = false,
        VqCodebookSize = 2_048,
        VqMaxCodebooks = 4,
        ActiveCodebooks = 4,
        KbpsPerCodebook = 0.55,
    };
}
