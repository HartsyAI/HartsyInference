namespace HartsyInference.Audio.Models.IndexTts2;

/// <summary>Config for <see cref="IndexTts2Dit"/> — IndexTTS-2's S2Mel flow-matching DiT. Values are the real
/// <c>s2mel.DiT</c>/<c>s2mel.wavenet</c>/<c>s2mel.style_encoder</c> blocks of <c>config.yaml</c>, confirmed
/// byte-identical between the 2.0 and 2.5 repos, and cross-checked against the real <c>s2mel.pth</c>'s own
/// tensor shapes (e.g. <c>attention.wqkv.weight [1536,512]</c> confirms <c>HiddenDim=512</c>,
/// <c>feed_forward.w1.weight [1536,512]</c> confirms <c>FfnDim=1536</c>).</summary>
public sealed record IndexTts2DitConfig
{
    public required int HiddenDim { get; init; }
    public required int NumHeads { get; init; }
    public required int Depth { get; init; }
    public required int InChannels { get; init; }     // mel bands (80)
    public required int ContentDim { get; init; }      // length-regulated semantic-content width (512)
    public required int StyleDim { get; init; }        // CAM++ embedding width (192)
    public required int WavenetHiddenDim { get; init; }
    public int WavenetKernelSize { get; init; } = 5;
    public int WavenetDilationRate { get; init; } = 1;
    public int WavenetNumLayers { get; init; } = 8;
    public float RopeBase { get; init; } = 10_000f;
    public int FreqEmbedSize { get; init; } = 256;

    public int HeadDim => HiddenDim / NumHeads;

    /// <summary>The real, config-computed SwiGLU intermediate size: <c>find_multiple(2*4*dim/3, 256)</c>.
    /// Confirmed against the real checkpoint's own <c>feed_forward.w1.weight</c> shape rather than trusted
    /// blind — do not change this formula without re-checking that shape.</summary>
    public int FfnDim
    {
        get
        {
            int hidden = 4 * HiddenDim;
            int nHidden = 2 * hidden / 3;
            const int multiple = 256;
            return nHidden % multiple == 0 ? nHidden : nHidden + multiple - (nHidden % multiple);
        }
    }

    /// <summary>IndexTTS-2's real S2Mel DiT preset (both 2.0 and 2.5 — confirmed identical config.yaml blocks).</summary>
    public static IndexTts2DitConfig IndexTts2 => new()
    {
        HiddenDim = 512,
        NumHeads = 8,
        Depth = 13,
        InChannels = 80,
        ContentDim = 512,
        StyleDim = 192,
        WavenetHiddenDim = 512,
        WavenetKernelSize = 5,
        WavenetDilationRate = 1,
        WavenetNumLayers = 8,
    };
}
