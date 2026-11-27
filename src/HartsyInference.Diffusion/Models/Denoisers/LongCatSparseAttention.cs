using HartsyInference.Core.Backends.Attention;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The block-sparse self-attention the base model's 720p refinement runs (the reference's <c>bsa_params</c>): the share of
/// key blocks a query block does not read, and the query and key block in latent tokens. The grids come from each forward.</summary>
public sealed record LongCatSparseAttention(double Sparsity, TokenGrid3D QueryBlock, TokenGrid3D KeyBlock)
{
    /// <summary>The released base DiT's values: sparsity 0.9375 and 4×4×4 blocks.</summary>
    public static LongCatSparseAttention Released { get; } = new(0.9375, new TokenGrid3D(4, 4, 4), new TokenGrid3D(4, 4, 4));

    /// <summary>Why a config's <c>bsa_params</c> cannot be run, or null. Only the block-sparse forward reads the parameters, so a
    /// config that holds others still loads, and <see cref="Options"/> refuses it.</summary>
    public string? Unsupported { get; init; }

    /// <summary>The options of one self-attention segment: queries of <paramref name="queryFrames"/> frames against keys of
    /// <paramref name="keyFrames"/> frames, each frame <paramref name="gridH"/> × <paramref name="gridW"/> tokens.</summary>
    /// <exception cref="ArgumentException">A grid extent is not a multiple of its block's.</exception>
    /// <exception cref="InvalidDataException">The config's parameters are <see cref="Unsupported"/>.</exception>
    public BlockSparseAttentionOptions Options(int queryFrames, int keyFrames, int gridH, int gridW)
    {
        if (Unsupported is not null)
            throw new InvalidDataException(Unsupported);
        Require(queryFrames, gridH, gridW, QueryBlock, "query");
        Require(keyFrames, gridH, gridW, KeyBlock, "key");
        return new BlockSparseAttentionOptions
        {
            QueryGrid = new TokenGrid3D(queryFrames, gridH, gridW), KeyGrid = new TokenGrid3D(keyFrames, gridH, gridW),
            QueryBlock = QueryBlock, KeyBlock = KeyBlock, Sparsity = Sparsity,
        };
    }

    private static void Require(int frames, int gridH, int gridW, TokenGrid3D block, string side)
    {
        if (frames % block.Frames != 0 || gridH % block.Height != 0 || gridW % block.Width != 0)
            throw new ArgumentException($"LongCat's block-sparse attention needs {side} token grids that are multiples of "
                + $"{block.Frames}x{block.Height}x{block.Width} (frames, height, width); got {frames}x{gridH}x{gridW}.");
    }
}
