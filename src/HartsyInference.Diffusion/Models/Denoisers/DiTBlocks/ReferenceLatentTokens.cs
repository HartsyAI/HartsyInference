using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>One reference image as packed latent tokens <c>[1, Height·Width, C]</c> in raster order, with its token grid.
/// The tokens are owned by whoever built them.</summary>
public sealed record ReferenceLatentTokens(Tensor Tokens, int Height, int Width)
{
    /// <summary>Tokens in this reference.</summary>
    public int TokenCount => Height * Width;

    /// <summary>Throws unless <see cref="Tokens"/> is <c>[1, Height·Width, channels]</c>.</summary>
    public void Validate(int channels, string family)
    {
        ArgumentNullException.ThrowIfNull(Tokens);
        if (Height < 1 || Width < 1 || Tokens.Shape.Rank != 3 || Tokens.Shape[0] != 1 || Tokens.Shape[1] != TokenCount
            || Tokens.Shape[2] != channels)
        {
            throw new ArgumentException(
                $"{family}: a reference needs packed tokens [1, {Height}x{Width}, {channels}], got {Tokens.Shape}.");
        }
    }
}
