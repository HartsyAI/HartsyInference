namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Which stacked output heads a MiniMax-H3-Image sampler step uses, and how much of each. Mirrors ComfyUI's
/// <c>_pdd_head</c>: block 0 is always the base head with weight 1, and block n (n &gt;= 1) is an offset weighted by the
/// width of its own interval, normalized by the total width of the span. Block 0's interval is part of that total
/// even though it carries no offset weight, so the offset weights do not sum to 1 when the span starts at 0.</summary>
public sealed class StackedOutputHeads
{
    /// <summary>Grid of heads + 1 points, decreasing from 1 to 0.</summary>
    private readonly double[] _grid;

    public StackedOutputHeads(IReadOnlyList<double> grid)
    {
        ArgumentNullException.ThrowIfNull(grid);
        if (grid.Count < 2)
        {
            throw new ArgumentException("A stacked output-head grid needs at least two points.", nameof(grid));
        }

        _grid = new double[grid.Count];
        for (int i = 0; i < grid.Count; i++)
        {
            _grid[i] = grid[i];
            if (i > 0 && !(grid[i - 1] > grid[i]))
            {
                throw new ArgumentException("The output-head grid must be strictly decreasing.", nameof(grid));
            }
        }
    }

    /// <summary>Number of heads (grid intervals).</summary>
    public int HeadCount => _grid.Length - 1;

    /// <summary>Resolves the step from <paramref name="sigma"/> to <paramref name="sigmaNext"/> into its span and
    /// per-block weights. Each sigma snaps to the nearest grid point; the span is clamped so it covers at least one
    /// interval and stays inside the grid.</summary>
    public OutputHeadStep Resolve(double sigma, double sigmaNext)
    {
        int heads = HeadCount;
        int start = Math.Min(NearestIndex(sigma), heads - 1);
        int stop = Math.Max(NearestIndex(sigmaNext), start + 1);

        double total = 0.0;
        for (int k = start; k < stop; k++)
        {
            total += _grid[k] - _grid[k + 1];
        }

        int first = Math.Max(start, 1);
        double[] weights = new double[heads];
        for (int n = first; n < stop; n++)
        {
            weights[n] = (_grid[n] - _grid[n + 1]) / total;
        }

        return new OutputHeadStep(start, stop, first, weights);
    }

    /// <summary>Index of the grid point nearest <paramref name="sigma"/>; the lowest index wins a tie, as in ComfyUI's
    /// <c>min(range, key=...)</c>.</summary>
    private int NearestIndex(double sigma)
    {
        int best = 0;
        double bestDistance = Math.Abs(_grid[0] - sigma);
        for (int k = 1; k < _grid.Length; k++)
        {
            double distance = Math.Abs(_grid[k] - sigma);
            if (distance < bestDistance)
            {
                best = k;
                bestDistance = distance;
            }
        }

        return best;
    }
}

/// <summary>A resolved step: the grid span it covers, the first offset block, and each block's weight (length = head
/// count). Blocks outside <see cref="FirstOffset"/> .. <see cref="Stop"/> - 1 have weight 0.</summary>
public sealed record OutputHeadStep(int Start, int Stop, int FirstOffset, IReadOnlyList<double> Weights);
