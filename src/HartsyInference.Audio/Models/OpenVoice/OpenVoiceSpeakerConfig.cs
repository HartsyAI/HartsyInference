namespace HartsyInference.Audio.Models.OpenVoice;

/// <summary>Configuration for <see cref="OpenVoiceSpeakerEncoder"/> (the OpenVoice V2 tone-color
/// <c>ReferenceEncoder</c>). Defaults match <c>myshell-ai/OpenVoiceV2/converter</c>: a LayerNorm over the
/// linear-spectrogram bins, a six-layer weight-normed Conv2d stack (filters 32/32/64/64/128/128, kernel 3×3,
/// stride 2×2, pad 1), a single-layer GRU (hidden 128), and a Linear to the 256-d (gin) speaker vector.</summary>
public sealed record OpenVoiceSpeakerConfig
{
    public int SampleRate { get; init; } = 22_050;

    /// <summary>Linear-spectrogram bin count fed to the encoder (<c>n_fft/2 + 1</c>; 513 for n_fft 1024).</summary>
    public int SpecChannels { get; init; } = 513;

    /// <summary>Per-layer output channels for the Conv2d stack.</summary>
    public IReadOnlyList<int> Channels { get; init; } = [32, 32, 64, 64, 128, 128];

    public int KernelSize { get; init; } = 3;
    public int Stride { get; init; } = 2;
    public int Padding { get; init; } = 1;

    /// <summary>GRU hidden size (<c>gin / 2</c> in the upstream model).</summary>
    public int GruHidden { get; init; } = 128;

    /// <summary>Speaker-conditioning channel count (the tone converter's <c>g</c> width).</summary>
    public int Gin { get; init; } = 256;

    public static OpenVoiceSpeakerConfig Default => new();
}
