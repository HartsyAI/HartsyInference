using HartsyInference.Audio.Models.Whisper;

namespace HartsyInference.Audio.Models.QwenOmni;

/// <summary>Qwen2.5-Omni thinker audio-tower configuration plus the audio front-end constants, defaulting to the 3B checkpoint.</summary>
public sealed record QwenOmniConfig
{
    /// <summary>Mel bins of the Whisper-style front-end.</summary>
    public int NumMelBins { get; init; } = 128;

    /// <summary>Tower width (d_model).</summary>
    public int DModel { get; init; } = 1_280;

    /// <summary>Transformer layers in the tower.</summary>
    public int Layers { get; init; } = 32;

    /// <summary>Attention heads.</summary>
    public int Heads { get; init; } = 20;

    /// <summary>Feed-forward inner dim.</summary>
    public int FfnDim { get; init; } = 5_120;

    /// <summary>Half the mel-chunk length: each chunk is 2 * NWindow mel frames and 1 * NWindow conv frames.</summary>
    public int NWindow { get; init; } = 100;

    /// <summary>Projection width, equal to the thinker LLM hidden size.</summary>
    public int OutputDim { get; init; } = 2_048;

    /// <summary>Layer-norm epsilon (torch default).</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Sample rate expected by the feature extractor.</summary>
    public int SampleRate { get; init; } = 16_000;

    /// <summary>Feature-extractor FFT size.</summary>
    public int NFft { get; init; } = 400;

    /// <summary>Feature-extractor hop in samples.</summary>
    public int HopLength { get; init; } = 160;

    /// <summary>Feature-extractor padded length in samples (300 s), the max-8 dB clamp is taken over this window.</summary>
    public int MaxSamples { get; init; } = 4_800_000;

    /// <summary>Token id of <c>&lt;|AUDIO|&gt;</c>.</summary>
    public int AudioTokenId { get; init; } = 151_646;

    /// <summary>Token id of <c>&lt;|audio_bos|&gt;</c>.</summary>
    public int AudioBosTokenId { get; init; } = 151_647;

    /// <summary>Token id of <c>&lt;|audio_eos|&gt;</c>.</summary>
    public int AudioEosTokenId { get; init; } = 151_648;

    /// <summary>Head dimension.</summary>
    public int HeadDim => DModel / Heads;

    /// <summary>Mel frames per independently convolved chunk.</summary>
    public int ChunkFrames => 2 * NWindow;

    /// <summary>The Qwen2.5-Omni-3B thinker audio tower.</summary>
    public static QwenOmniConfig Default { get; } = new();

    /// <summary>The Whisper layer configuration the shared encoder block is built from.</summary>
    public WhisperConfig ToWhisperLayerConfig() => new()
    {
        NumMelBins = NumMelBins,
        HiddenSize = DModel,
        NumHeads = Heads,
        IntermediateSize = FfnDim,
        EncoderLayers = Layers,
        LayerNormEps = LayerNormEps,
    };

    /// <summary>Throws when the dimensions cannot form a valid tower.</summary>
    public void Validate()
    {
        if (NumMelBins <= 0 || DModel <= 0 || Layers <= 0 || Heads <= 0 || FfnDim <= 0 || NWindow <= 0 || OutputDim <= 0)
        {
            throw new ArgumentException("All QwenOmniConfig dimensions must be positive.");
        }
        if (DModel % Heads != 0 || DModel % 2 != 0)
        {
            throw new ArgumentException($"DModel {DModel} must be even and divisible by Heads {Heads}.");
        }
        if (DModel / 2 < 2)
        {
            throw new ArgumentException("DModel must be at least 4 for the sinusoidal positions.");
        }
    }
}
