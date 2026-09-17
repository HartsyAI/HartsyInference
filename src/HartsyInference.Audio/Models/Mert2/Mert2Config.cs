namespace HartsyInference.Audio.Models.Mert2;

/// <summary>Geometry of the MERT-v2 audio encoder that SheetSage2 transcribes from: a mel frontend, a ConvNeXt
/// subsampling stack that takes the frame rate down 4×, then 24 RoPE Conformer layers at width 1024. 24 kHz mono
/// input. The released model always pads its input to a full <see cref="WindowSeconds"/> window, so in production
/// every field here is a constant and <see cref="TokensPerWindow"/> is a fixed 7500.</summary>
public sealed record Mert2Config
{
    /// <summary>Encoder width, and the dimension every Conformer layer runs at.</summary>
    public int Dim { get; init; } = 1_024;

    /// <summary>Hidden width of each Conformer half-step feed-forward.</summary>
    public int Intermediate { get; init; } = 4_096;

    public int Heads { get; init; } = 16;
    public int Layers { get; init; } = 24;

    /// <summary>Width the decoder consumes, after <c>encoder_projection</c>.</summary>
    public int ProjectionDim { get; init; } = 512;

    public int SampleRate { get; init; } = 24_000;
    public int NFft { get; init; } = 2_048;
    public int HopLength { get; init; } = 240;
    public int MelBins { get; init; } = 128;

    /// <summary>Length the waveform is zero-padded to before the frontend runs — the released encoder attends over
    /// the whole window including its padding, so a shorter clip is not a cheaper forward, just a quieter one.</summary>
    public int WindowSeconds { get; init; } = 300;

    /// <summary>Channel widths the ConvNeXt stack steps through, starting at the mel bin count.</summary>
    public IReadOnlyList<int> Channels { get; init; } = [128, 128, 512, 1_024];

    /// <summary>Per-block stride of the resampling convolution; 1 means the block keeps the frame rate.</summary>
    public IReadOnlyList<int> Strides { get; init; } = [1, 2, 2];

    /// <summary>ConvNeXt layers per block.</summary>
    public IReadOnlyList<int> Depths { get; init; } = [3, 4, 5];

    /// <summary>Kernel of the resampling convolution between ConvNeXt blocks.</summary>
    public int ResampleKernel { get; init; } = 2;

    /// <summary>Depthwise kernel inside a ConvNeXt layer (symmetric padding, so the frame rate is unchanged).</summary>
    public int ConvNextKernel { get; init; } = 7;

    /// <summary>Depthwise kernel of the Conformer convolution module.</summary>
    public int ConformerConvKernel { get; init; } = 31;

    /// <summary>ConvNeXt LayerNorm epsilon; the Conformer stack uses <see cref="LayerNormEps"/> instead.</summary>
    public float ConvNextNormEps { get; init; } = 1e-6f;

    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>RoPE base. The rotation is the split-half convention: pair <c>k</c> is <c>[k]</c> with
    /// <c>[k + headDim/2]</c>, not adjacent elements.</summary>
    public float RopeTheta { get; init; } = 10_000f;

    public int HeadDim => Dim / Heads;

    /// <summary>Mel frames the frontend emits for a full window. <c>torch.stft(center=True)</c> yields
    /// <c>samples/hop + 1</c> frames and the frontend drops the last one.</summary>
    public int FramesPerWindow => WindowSeconds * SampleRate / HopLength;

    /// <summary>Encoder tokens for a full window — a fixed 7500, since the input length never varies.</summary>
    public int TokensPerWindow => SubsampledLength(FramesPerWindow);

    /// <summary>Frames left after the two stride-2 resampling convolutions.</summary>
    public int SubsampledLength(int frames)
    {
        int length = frames;
        for (int i = 0; i < Strides.Count; i++)
        {
            if (Strides[i] > 1)
            {
                length = (length - ResampleKernel) / Strides[i] + 1;
            }
        }
        return length;
    }
}
