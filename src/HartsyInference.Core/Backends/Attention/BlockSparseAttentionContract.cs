using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>The operand and option checks every <see cref="IBackend.BlockSparseAttention"/> implementation shares.</summary>
public static class BlockSparseAttentionContract
{
    /// <summary>Checks the options, the operands' shapes against their grids and their dtypes (Q, K and V one of F32 or BF16, the
    /// output F32 or BF16), aliasing and the selection report. It reads no tensor's data.</summary>
    /// <exception cref="ArgumentException">An operand or option does not fit the call.</exception>
    public static void Validate(Tensor output, Tensor query, Tensor key, Tensor value, BlockSparseAttentionOptions options,
        Tensor? selected = null)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(value);
        ArgumentNullException.ThrowIfNull(options);
        if (query.Shape.Rank != 4 || key.Shape.Rank != 4 || value.Shape != key.Shape || output.Shape != query.Shape
            || query.ElementCount == 0 || key.ElementCount == 0)
            throw new ArgumentException(
                "Block-sparse attention takes head-major [B, H, S, D] operands of positive extents, V in K's shape and the output in Q's; "
                + $"got Q={query.Shape}, K={key.Shape}, V={value.Shape}, O={output.Shape}.");
        if (key.Shape[0] != query.Shape[0] || key.Shape[1] != query.Shape[1] || key.Shape[3] != query.Shape[3])
            throw new ArgumentException($"Q and K share batch, heads and head dim; got Q={query.Shape}, K={key.Shape}.", nameof(key));
        if (!IsOperandType(query.DType) || key.DType != query.DType || value.DType != query.DType || !IsOperandType(output.DType))
            throw new ArgumentException(
                "Block-sparse attention reads Q, K and V in one of F32 or BF16 and writes F32 or BF16; "
                + $"got Q={query.DType}, K={key.DType}, V={value.DType}, O={output.DType}.");
        CheckGrid(options.QueryGrid, options.QueryBlock, query.Shape[2], "query");
        CheckGrid(options.KeyGrid, options.KeyBlock, key.Shape[2], "key");
        if (!(options.Sparsity is >= 0.0 and <= 1.0))
            throw new ArgumentException($"The sparsity is in [0, 1]; got {options.Sparsity}.", nameof(options));
        if (options.Scale is { } scale && !float.IsFinite(scale))
            throw new ArgumentException("The scale must be finite.", nameof(options));
        // By reference only: reading a pointer would sync a device-resident operand to the host.
        if (ReferenceEquals(output, query) || ReferenceEquals(output, key) || ReferenceEquals(output, value))
            throw new ArgumentException("The output must not alias an input.", nameof(output));
        if (selected is null) return;
        TensorShape report = new(query.Shape[0], query.Shape[1], options.QueryBlocks, options.SelectedBlocks);
        if (selected.DType != DType.I32 || selected.Shape != report)
            throw new ArgumentException($"The selection report is I32 {report}; got {selected.DType} {selected.Shape}.", nameof(selected));
        if (ReferenceEquals(selected, output) || ReferenceEquals(selected, query) || ReferenceEquals(selected, key)
            || ReferenceEquals(selected, value))
            throw new ArgumentException("The selection report must be its own tensor.", nameof(selected));
    }

    private static bool IsOperandType(DType type) => type == DType.F32 || type == DType.BF16;

    private static void CheckGrid(TokenGrid3D grid, TokenGrid3D block, long tokens, string side)
    {
        if (grid.Frames <= 0 || grid.Height <= 0 || grid.Width <= 0 || block.Frames <= 0 || block.Height <= 0 || block.Width <= 0)
            throw new ArgumentException($"The {side} grid and block need positive extents; got grid {grid}, block {block}.", "options");
        if (grid.Frames % block.Frames != 0 || grid.Height % block.Height != 0 || grid.Width % block.Width != 0)
            throw new ArgumentException(
                $"The {side} grid {grid} is not a multiple of its block {block} on every axis; the caller pads it.", "options");
        if (grid.Tokens != tokens)
            throw new ArgumentException($"The {side} grid {grid} holds {grid.Tokens} tokens; the sequence has {tokens}.", "options");
    }
}
