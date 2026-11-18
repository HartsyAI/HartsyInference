namespace HartsyInference.Audio.Dsp.Alignment;

/// <summary>Monotonic dynamic-time-warping alignment of an ordered symbol sequence (rows) onto time (columns).</summary>
public static class DynamicTimeWarping
{
    private const sbyte Diagonal = 0;
    private const sbyte Down = 1;
    private const sbyte Right = 2;

    /// <summary>Finds the lowest-cost monotonic path from <c>(0, 0)</c> to <c>(rows - 1, cols - 1)</c> through a
    /// row-major <c>[rows, cols]</c> cost matrix, stepping diagonally, down a row or right a column. A tie goes to
    /// the rightward step unless the diagonal or downward one is strictly cheaper, which is the rule OpenAI's Whisper
    /// aligner uses.</summary>
    /// <returns>The visited cells in order: <c>Rows[i]</c> and <c>Cols[i]</c> are the i-th cell's row and column.</returns>
    public static (int[] Rows, int[] Cols) Align(ReadOnlySpan<float> cost, int rows, int cols)
    {
        if (rows <= 0 || cols <= 0)
        {
            throw new ArgumentOutOfRangeException(rows <= 0 ? nameof(rows) : nameof(cols), "The cost matrix needs at least one row and one column.");
        }
        if (cost.Length != rows * cols)
        {
            throw new ArgumentException($"Cost has {cost.Length} entries but {rows} x {cols} needs {rows * cols}.", nameof(cost));
        }

        int stride = cols + 1;
        float[] total = new float[(rows + 1) * stride];
        sbyte[] trace = new sbyte[(rows + 1) * stride];
        Array.Fill(total, float.PositiveInfinity);
        total[0] = 0f;
        for (int i = 1; i <= rows; i++)
        {
            for (int j = 1; j <= cols; j++)
            {
                float diagonal = total[(i - 1) * stride + j - 1];
                float down = total[(i - 1) * stride + j];
                float right = total[i * stride + j - 1];
                float best;
                sbyte step;
                if (diagonal < down && diagonal < right)
                {
                    best = diagonal;
                    step = Diagonal;
                }
                else if (down < diagonal && down < right)
                {
                    best = down;
                    step = Down;
                }
                else
                {
                    best = right;
                    step = Right;
                }
                total[i * stride + j] = best + cost[(i - 1) * cols + j - 1];
                trace[i * stride + j] = step;
            }
        }

        // The border has no predecessor of its own: the first row can only be entered from the left and the first
        // column only from above, so a path that reaches an edge slides along it to the origin.
        for (int j = 0; j <= cols; j++)
        {
            trace[j] = Right;
        }
        for (int i = 0; i <= rows; i++)
        {
            trace[i * stride] = Down;
        }

        List<int> pathRows = new List<int>(rows + cols);
        List<int> pathCols = new List<int>(rows + cols);
        int row = rows;
        int col = cols;
        while (row > 0 || col > 0)
        {
            pathRows.Add(row - 1);
            pathCols.Add(col - 1);
            switch (trace[row * stride + col])
            {
                case Diagonal:
                    row--;
                    col--;
                    break;
                case Down:
                    row--;
                    break;
                default:
                    col--;
                    break;
            }
        }
        pathRows.Reverse();
        pathCols.Reverse();
        return (pathRows.ToArray(), pathCols.ToArray());
    }
}
