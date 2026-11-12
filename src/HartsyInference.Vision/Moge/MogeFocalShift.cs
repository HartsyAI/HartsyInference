namespace HartsyInference.Vision.Moge;

/// <summary>
/// MoGe's recovery of the focal length and the z shift of an affine-invariant point map (upstream <c>recover_focal_shift</c>, <c>solve_optimal_focal_shift</c> and
/// <c>solve_optimal_shift</c>): find the shift <c>s</c> and focal <c>f</c> that minimise <c>| f · xy / (z + s) - uv |</c> over the valid points of a 64 x 64 nearest-neighbour subsample,
/// where <c>uv</c> is the view-plane grid of <see cref="MogeHead.ViewPlaneUv"/>. For a free focal, the best <c>f</c> of a given shift is closed form, so the search is one-dimensional.
/// Upstream runs SciPy's Levenberg-Marquardt (<c>least_squares(..., x0=0, ftol=1e-3, method='lm')</c>), which stops at a loose tolerance; this runs the same iteration to a tight one,
/// so the shift agrees with upstream to the looseness of that tolerance, not bit for bit.
/// </summary>
public static class MogeFocalShift
{
    private const int MaxIterations = 200;

    /// <summary>Minimises over the shift (and, when <paramref name="focal"/> is null, the focal length).</summary>
    /// <param name="uv"><c>n</c> pairs of view-plane coordinates.</param>
    /// <param name="xyz"><c>n</c> triples of point coordinates.</param>
    /// <param name="focal">A known focal length relative to the half diagonal, or null to solve for it.</param>
    public static (double Shift, double Focal) Solve(double[] uv, double[] xyz, double? focal = null)
    {
        int n = uv.Length / 2;
        if (uv.Length != 2 * n || xyz.Length != 3 * n || n < 2)
            throw new ArgumentException("The solver needs at least two points, each with a UV pair and an XYZ triple.");
        double[] residual = new double[2 * n], plus = new double[2 * n], minus = new double[2 * n];

        // Fills the residuals of a shift (with its best focal) and returns their squared sum; NaN when a point is at or behind the camera.
        double Evaluate(double shift, double[] output)
        {
            double numerator = 0, denominator = 0;
            for (int i = 0; i < n; i++)
            {
                double depth = xyz[3 * i + 2] + shift;
                if (!(depth > 1e-9))
                    return double.NaN;
                double px = xyz[3 * i] / depth, py = xyz[3 * i + 1] / depth;
                numerator += px * uv[2 * i] + py * uv[2 * i + 1];
                denominator += px * px + py * py;
            }

            double f = focal ?? numerator / denominator;
            double cost = 0;
            for (int i = 0; i < n; i++)
            {
                double depth = xyz[3 * i + 2] + shift;
                double rx = f * xyz[3 * i] / depth - uv[2 * i], ry = f * xyz[3 * i + 1] / depth - uv[2 * i + 1];
                output[2 * i] = rx;
                output[2 * i + 1] = ry;
                cost += rx * rx + ry * ry;
            }

            return cost;
        }

        double shiftNow = 0, cost = Evaluate(shiftNow, residual), lambda = 1e-3;
        if (double.IsNaN(cost))
            throw new ArgumentException("A point is at or behind the camera for a zero shift; the solver needs positive depths.");
        for (int iteration = 0; iteration < MaxIterations; iteration++)
        {
            double h = 1e-6 * Math.Max(1.0, Math.Abs(shiftNow));
            double up = Evaluate(shiftNow + h, plus), down = Evaluate(shiftNow - h, minus);
            if (double.IsNaN(up) || double.IsNaN(down))
                break;
            double g = 0, curvature = 0;
            for (int i = 0; i < residual.Length; i++)
            {
                double j = (plus[i] - minus[i]) / (2 * h);
                g += j * residual[i];
                curvature += j * j;
            }

            if (curvature <= 0)
                break;
            bool improved = false;
            for (int attempt = 0; attempt < 30 && !improved; attempt++)
            {
                double step = -g / (curvature * (1 + lambda));
                double trial = Evaluate(shiftNow + step, plus);
                if (!double.IsNaN(trial) && trial < cost)
                {
                    double reduction = (cost - trial) / Math.Max(cost, 1e-300);
                    shiftNow += step;
                    cost = trial;
                    Array.Copy(plus, residual, residual.Length);
                    lambda = Math.Max(lambda * 0.3, 1e-12);
                    improved = true;
                    if (reduction < 1e-12 || Math.Abs(step) < 1e-10 * (1 + Math.Abs(shiftNow)))
                        return (shiftNow, FocalOf(shiftNow));
                }
                else
                {
                    lambda *= 10;
                }
            }

            if (!improved)
                break;
        }

        return (shiftNow, FocalOf(shiftNow));

        double FocalOf(double shift)
        {
            if (focal is double known)
                return known;
            double numerator = 0, denominator = 0;
            for (int i = 0; i < n; i++)
            {
                double depth = xyz[3 * i + 2] + shift;
                double px = xyz[3 * i] / depth, py = xyz[3 * i + 1] / depth;
                numerator += px * uv[2 * i] + py * uv[2 * i + 1];
                denominator += px * px + py * py;
            }

            return numerator / denominator;
        }
    }

    /// <summary>
    /// <c>recover_focal_shift</c> of one point map: subsample the points, the UV grid and the mask to <paramref name="downsample"/> squared with nearest-neighbour indices
    /// (<c>floor(i · in / out)</c>), keep the valid ones and <see cref="Solve"/>. Fewer than two valid points give focal 1 and shift 0, as upstream.
    /// </summary>
    /// <param name="points">Planar <c>[3, height, width]</c>.</param>
    /// <param name="valid"><c>height · width</c> flags of the points to use; null uses all.</param>
    /// <param name="focal">A known focal length (<c>solve_optimal_shift</c>), or null to solve for it too.</param>
    public static (double Focal, double Shift) Recover(ReadOnlySpan<float> points, ReadOnlySpan<bool> valid, int height, int width, double? focal = null, int downsample = 64)
    {
        if (points.Length != 3 * height * width || (!valid.IsEmpty && valid.Length != height * width))
            throw new ArgumentException($"A point map of {height} x {width} holds {3 * height * width} values and {height * width} flags; got {points.Length} and {valid.Length}.");
        float[] uGrid = new float[height * width], vGrid = new float[height * width];
        MogeHead.ViewPlaneUv(uGrid, vGrid, height, width, (double)width / height);

        List<double> uv = [], xyz = [];
        int plane = height * width;
        for (int oy = 0; oy < downsample; oy++)
        {
            int y = Math.Min((int)(oy * ((float)height / downsample)), height - 1);
            for (int ox = 0; ox < downsample; ox++)
            {
                int x = Math.Min((int)(ox * ((float)width / downsample)), width - 1);
                int index = y * width + x;
                if (!valid.IsEmpty && !valid[index])
                    continue;
                uv.Add(uGrid[index]);
                uv.Add(vGrid[index]);
                xyz.Add(points[index]);
                xyz.Add(points[plane + index]);
                xyz.Add(points[2 * plane + index]);
            }
        }

        if (uv.Count / 2 < 2)
            return (1.0, 0.0);
        (double shift, double solved) = Solve([.. uv], [.. xyz], focal);
        return (solved, shift);
    }
}
