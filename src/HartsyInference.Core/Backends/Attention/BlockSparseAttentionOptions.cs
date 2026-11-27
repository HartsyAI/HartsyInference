namespace HartsyInference.Core.Backends.Attention;

/// <summary>The token layout and block selection of a <see cref="IBackend.BlockSparseAttention"/> call. Queries and keys have their own
/// grid and block; the two grids may differ on any axis.</summary>
/// <remarks>Blocks are numbered in <c>(T, H, W)</c> raster order of the block grid, and the tokens inside one in raster order of the
/// block. <see cref="BlockSparseAttentionContract.Validate"/> checks an options object against its operands; the derived counts and
/// rows are meaningful only for options it accepts.</remarks>
public sealed record BlockSparseAttentionOptions
{
    /// <summary>The queries' token grid; its token count is the query sequence length.</summary>
    public required TokenGrid3D QueryGrid { get; init; }

    /// <summary>The keys' and values' token grid; its token count is the key sequence length.</summary>
    public required TokenGrid3D KeyGrid { get; init; }

    /// <summary>The query block; every extent of <see cref="QueryGrid"/> is a multiple of its extent.</summary>
    public TokenGrid3D QueryBlock { get; init; } = new(4, 4, 4);

    /// <summary>The key block; every extent of <see cref="KeyGrid"/> is a multiple of its extent.</summary>
    public TokenGrid3D KeyBlock { get; init; } = new(4, 4, 4);

    /// <summary>The share of key blocks a query block does not read, in <c>[0, 1]</c>. A double, because the selected count truncates
    /// <c>(1 − Sparsity) × KeyBlocks</c> in double precision.</summary>
    public required double Sparsity { get; init; }

    /// <summary>The softmax score multiplier; null is <c>headDim^-1/2</c>. Block selection does not use it.</summary>
    public float? Scale { get; init; }

    /// <summary>Query blocks in the query grid.</summary>
    public int QueryBlocks => (int)(QueryGrid.Tokens / QueryBlock.Tokens);

    /// <summary>Key blocks in the key grid.</summary>
    public int KeyBlocks => (int)(KeyGrid.Tokens / KeyBlock.Tokens);

    /// <summary>Key blocks every query block reads: <c>(1 − Sparsity) × KeyBlocks</c>, truncated. It can be zero.</summary>
    public int SelectedBlocks => (int)((1.0 - Sparsity) * KeyBlocks);

    /// <summary>The raster row in the query sequence of token <paramref name="offset"/> of query block <paramref name="block"/>.</summary>
    public int QueryRow(int block, int offset) => Row(QueryGrid, QueryBlock, block, offset);

    /// <summary>The raster row in the key sequence of token <paramref name="offset"/> of key block <paramref name="block"/>.</summary>
    public int KeyRow(int block, int offset) => Row(KeyGrid, KeyBlock, block, offset);

    private static int Row(TokenGrid3D grid, TokenGrid3D box, int block, int offset)
    {
        int blocksHigh = grid.Height / box.Height, blocksWide = grid.Width / box.Width;
        int frame = block / (blocksHigh * blocksWide) * box.Frames + offset / (box.Height * box.Width);
        int row = block / blocksWide % blocksHigh * box.Height + offset / box.Width % box.Height;
        int column = block % blocksWide * box.Width + offset % box.Width;
        return (frame * grid.Height + row) * grid.Width + column;
    }
}
