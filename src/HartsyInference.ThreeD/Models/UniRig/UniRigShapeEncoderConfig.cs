using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Hyper-parameters of the Michelangelo-style point encoder both UniRig stages use (<c>michelangelo_encoder</c> in the model configs).</summary>
public sealed record UniRigShapeEncoderConfig
{
    /// <summary>Token width of the cross-attention and self-attention blocks.</summary>
    public required int Width { get; init; }

    /// <summary>Attention heads; <see cref="Width"/> divides evenly by it.</summary>
    public required int Heads { get; init; }

    /// <summary>Number of residual self-attention blocks after the cross-attention.</summary>
    public required int Layers { get; init; }

    /// <summary>Frequencies <c>2^0 … 2^(NumFreqs-1)</c> of the Fourier point embedding (no π factor).</summary>
    public int NumFreqs { get; init; } = 8;

    /// <summary>Per-point feature channels appended to the embedding; 3 for surface normals.</summary>
    public int PointFeats { get; init; } = 3;

    /// <summary>Output token count: the FPS size. Upstream first draws <c>4 * TokenNum</c> points at random.</summary>
    public int TokenNum { get; init; } = 1024;

    /// <summary>Per-point input width of <c>input_proj</c>: coordinates, their sin/cos bands, and the features.</summary>
    public int InputWidth => 3 * (2 * NumFreqs + 1) + PointFeats;

    /// <summary>The shipped articulation-xl encoder: width 512, 8 heads, 16 blocks, 1,024 tokens.</summary>
    public static UniRigShapeEncoderConfig Default => new() { Width = 512, Heads = 8, Layers = 16 };

    /// <summary>Throws for a config the encoder cannot build.</summary>
    public void Validate()
    {
        if (Width <= 0 || Heads <= 0 || Width % Heads != 0 || Layers < 0 || NumFreqs < 0 || PointFeats < 0 || TokenNum <= 0)
        {
            throw new HartsyInferenceException(
                $"UniRig shape encoder config is invalid: width {Width}, heads {Heads}, layers {Layers}, freqs {NumFreqs}, feats {PointFeats}, tokens {TokenNum}.");
        }
    }
}
