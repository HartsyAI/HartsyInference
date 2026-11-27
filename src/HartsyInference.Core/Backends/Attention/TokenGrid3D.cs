namespace HartsyInference.Core.Backends.Attention;

/// <summary>A <c>Frames × Height × Width</c> box of tokens in <c>(T, H, W)</c> raster order, width fastest: a whole latent token grid
/// of <see cref="IBackend.BlockSparseAttention"/>, or one block of it.</summary>
public readonly record struct TokenGrid3D(int Frames, int Height, int Width)
{
    /// <summary>Tokens in the box.</summary>
    public long Tokens => (long)Frames * Height * Width;
}
