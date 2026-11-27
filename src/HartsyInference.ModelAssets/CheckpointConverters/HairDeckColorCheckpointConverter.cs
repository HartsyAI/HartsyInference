using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Typed weights of the colour model (<c>hairdeck/models/colormodel.py</c>, route <c>reg1</c>): the image DiT under
/// <c>dit.</c>, the seven-channel patch statistics projection, the colour head and the Lab standardisation.</summary>
public static class HairDeckColorCheckpointConverter
{
    /// <summary>Colours per card along its length.</summary>
    public const int Bins = 16;

    /// <summary>The colour model. <see cref="Mu"/> and <see cref="Sd"/> hold the Lab standardisation (L, a, b).</summary>
    public sealed record Weights(
        HairDeckImageCheckpointConverter.Weights Dit,
        HairDeckSetDitCheckpointConverter.Linear Patch,
        HairDeckSetDitCheckpointConverter.Linear ColorAda,
        HairDeckSetDitCheckpointConverter.Linear ColorOut,
        Tensor Mu,
        Tensor Sd);

    /// <summary>Reads the colour model. The <c>dit.</c> prefix is stripped here only; the other keys are read as they are.</summary>
    public static Weights Convert(IReadOnlyDictionary<string, Tensor> all)
    {
        ArgumentNullException.ThrowIfNull(all);
        HairDeckImageCheckpointConverter.Weights dit = HairDeckImageCheckpointConverter.Convert(all, "dit.");
        long dim = dit.Dit.Dim;
        return new Weights(
            Dit: dit,
            Patch: HairDeckSetDitCheckpointConverter.Lin(all, "patch", dim, 7),
            ColorAda: HairDeckSetDitCheckpointConverter.Lin(all, "col_ada", 2 * dim, dim),
            ColorOut: HairDeckSetDitCheckpointConverter.Lin(all, "col_out", 3 * Bins, dim),
            Mu: HairDeckTensorReader.Take(all, "mu", 3),
            Sd: HairDeckTensorReader.Take(all, "sd", 3));
    }
}
