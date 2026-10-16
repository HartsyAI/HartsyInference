using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The raw conditioning one FLUX.3 pass reads: the text-encoder features <c>[1, T, 20480]</c> and, for edits,
/// every reference image's packed BN-normalized latent tokens concatenated <c>[1, Σh·w, 128]</c> with their grids.
/// The tensors stay owned by the caller.</summary>
public sealed record Flux3ConditionInput(Tensor TextFeatures, Tensor? ReferenceTokens, IReadOnlyList<(int Height, int Width)> ReferenceShapes)
{
    /// <summary>Text-only conditioning.</summary>
    public static Flux3ConditionInput TextOnly(Tensor textFeatures) => new(textFeatures, null, []);

    /// <summary>Total reference tokens (zero for text-only).</summary>
    public int ReferenceTokenCount
    {
        get
        {
            int total = 0;
            foreach ((int h, int w) in ReferenceShapes) total += h * w;
            return total;
        }
    }
}
