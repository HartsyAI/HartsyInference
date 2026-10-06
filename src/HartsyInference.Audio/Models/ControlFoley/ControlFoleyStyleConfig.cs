using HartsyInference.Audio.Models.Hubert;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Geometry of the MusicGen-Style conditioner ControlFoley uses as its timbre encoder (<c>StyleConditioner</c> of
/// <c>facebook/musicgen-style</c> with the MERT-v1-95M feature extractor): MERT features, a linear embedding, a pre-norm
/// transformer, batch norm, residual vector quantisation truncated to <see cref="EvalCodebooks"/>, a fixed downsample and the
/// projection to the language model's width.</summary>
public sealed record ControlFoleyStyleConfig
{
    /// <summary>The MERT feature extractor (a HuBERT-base shaped network).</summary>
    public HubertConfig Mert { get; init; } = new();

    /// <summary>Internal width of the conditioner (embedding, transformer, quantiser).</summary>
    public int Dim { get; init; } = 512;

    /// <summary>Transformer layers.</summary>
    public int Layers { get; init; } = 8;

    /// <summary>Attention heads.</summary>
    public int Heads { get; init; } = 8;

    /// <summary>Codebook entries per quantiser.</summary>
    public int Bins { get; init; } = 1_024;

    /// <summary>Codebooks stored in the checkpoint.</summary>
    public int Codebooks { get; init; } = 6;

    /// <summary>Codebooks used at inference; ControlFoley sets <c>eval_q=1</c>.</summary>
    public int EvalCodebooks { get; init; } = 1;

    /// <summary>Keep every n-th frame of the quantised sequence (75 Hz MERT frames to 5 Hz).</summary>
    public int Downsample { get; init; } = 15;

    /// <summary>Width of the returned style tokens.</summary>
    public int OutputDim { get; init; } = 1_536;

    /// <summary>Rate the clip is expected at.</summary>
    public int SampleRate { get; init; } = 32_000;

    /// <summary>Rate MERT consumes; the clip is resampled to it with the julius resampler audiocraft uses.</summary>
    public int MertSampleRate { get; init; } = 24_000;

    /// <summary>The released MusicGen-Style conditioner.</summary>
    public static ControlFoleyStyleConfig MusicGenStyle => new();
}
