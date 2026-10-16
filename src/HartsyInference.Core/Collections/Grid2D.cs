namespace HartsyInference.Core.Collections;

/// <summary>Conversions between a rectangular <c>T[,]</c> grid and the jagged rows or columns it is carried as in JSON and in
/// per-step callbacks.</summary>
public static class Grid2D
{
    /// <summary>The rows of <paramref name="grid"/>: element <c>[r][c]</c> is <c>grid[r, c]</c>.</summary>
    public static T[][] ToRows<T>(T[,] grid)
    {
        ArgumentNullException.ThrowIfNull(grid);
        T[][] rows = new T[grid.GetLength(0)][];
        for (int r = 0; r < rows.Length; r++)
        {
            T[] row = new T[grid.GetLength(1)];
            for (int c = 0; c < row.Length; c++) row[c] = grid[r, c];
            rows[r] = row;
        }
        return rows;
    }

    /// <summary>The grid whose rows are <paramref name="rows"/>, which must all have one length.</summary>
    /// <exception cref="ArgumentException">The rows differ in length.</exception>
    public static T[,] FromRows<T>(IReadOnlyList<T[]> rows)
    {
        ArgumentNullException.ThrowIfNull(rows);
        int width = rows.Count == 0 ? 0 : rows[0]?.Length ?? 0;
        T[,] grid = new T[rows.Count, width];
        for (int r = 0; r < rows.Count; r++)
        {
            T[] row = rows[r];
            if (row is null || row.Length != width)
            {
                throw new ArgumentException($"Row {r} has {row?.Length ?? 0} elements; every row must have {width}.", nameof(rows));
            }
            for (int c = 0; c < width; c++) grid[r, c] = row[c];
        }
        return grid;
    }

    /// <summary>The grid whose columns are <paramref name="columns"/>, which must all have one length: element <c>[r, c]</c> is
    /// <c>columns[c][r]</c>.</summary>
    /// <exception cref="ArgumentException">The columns differ in length.</exception>
    public static T[,] FromColumns<T>(IReadOnlyList<T[]> columns)
    {
        ArgumentNullException.ThrowIfNull(columns);
        int height = columns.Count == 0 ? 0 : columns[0]?.Length ?? 0;
        T[,] grid = new T[height, columns.Count];
        for (int c = 0; c < columns.Count; c++)
        {
            T[] column = columns[c];
            if (column is null || column.Length != height)
            {
                throw new ArgumentException($"Column {c} has {column?.Length ?? 0} elements; every column must have {height}.", nameof(columns));
            }
            for (int r = 0; r < height; r++) grid[r, c] = column[r];
        }
        return grid;
    }
}
