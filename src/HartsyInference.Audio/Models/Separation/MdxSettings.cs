namespace HartsyInference.Audio.Models.Separation;

/// <summary>Per-model constants of an MDX-Net vocal separator and the separator-level defaults of the reference
/// (<c>audio-separator</c> 0.30.2 as LongCat-Video drives it). The model constants come from the entry of
/// <c>mdx_model_data.json</c> keyed by the MD5 of the file's last 10,240,000 bytes.</summary>
public sealed record MdxSettings
{
    /// <summary>Kim_Vocal_2: MD5 <c>970b3f9492014d18fefeedfe4773cb42</c> maps to n_fft 7680, dim_f 3072, dim_t 2^8, compensate 1.009.</summary>
    public static MdxSettings KimVocal2 { get; } = new();

    /// <summary>STFT size; also twice the trim at both ends of a chunk.</summary>
    public int NFft { get; init; } = 7680;

    /// <summary>Samples between STFT frames.</summary>
    public int Hop { get; init; } = 1024;

    /// <summary>Frequency bins the network sees (<c>mdx_dim_f_set</c>); the rest of the one-sided spectrum is dropped.</summary>
    public int DimF { get; init; } = 3072;

    /// <summary>Frames per chunk (<c>segment_size</c>, equal to <c>2^mdx_dim_t_set</c>).</summary>
    public int Frames { get; init; } = 256;

    /// <summary>Gain applied to the separated stem (<c>compensate</c>).</summary>
    public float Compensate { get; init; } = 1.009f;

    /// <summary>Fraction of a chunk shared with the next (<c>overlap</c>, 0.25 by default).</summary>
    public double Overlap { get; init; } = 0.25;

    /// <summary>Peak the mix and the stem are normalized to when they exceed it (<c>normalization_threshold</c>).</summary>
    public float NormalizationThreshold { get; init; } = 0.9f;

    /// <summary>Sample rate the network was trained at; input is expected here.</summary>
    public int SampleRate { get; init; } = 44_100;

    /// <summary>Lowest frequency bins zeroed before the network (<c>spek[:, :, :3, :] = 0</c>).</summary>
    public int ZeroLowBins { get; init; } = 3;

    /// <summary>Audio samples per chunk: <c>hop × (frames − 1)</c>.</summary>
    public int ChunkSize => Hop * (Frames - 1);

    /// <summary>Samples trimmed at each end of a chunk: <c>n_fft / 2</c>.</summary>
    public int Trim => NFft / 2;
}
