namespace HartsyInference.ThreeD.Geometry.HairRig;

/// <summary>
/// Geometric skinning of hair points to spring chains, ported from HairDeck's <c>hairdeck/spring/sim.py</c> (Apache-2.0):
/// the chain a card follows, the segment and arc length of each point on it, and the blend across joints. Thresholds are in
/// face widths (<see cref="FaceWidth"/>), the reference's constants.
/// </summary>
public static class ChainBinding
{
    /// <summary>Mean distance of the lower 60% of a card, in face widths, under which the card follows a chain.</summary>
    public const double Assign = 0.4;

    /// <summary>Arc length, in face widths, over which the chain weight ramps from 0 at the root to 1.</summary>
    public const double Ramp = 0.05;

    /// <summary>Samples per card centreline used to choose its chain (the reference uses 16).</summary>
    public const int CentrelineSamples = 16;

    /// <summary>
    /// Face width in metres. The reference takes it from the fitted VRM body, which this export does not have, so it is a fixed
    /// estimate of an adult head width. The assignment and ramp scale with it.
    /// </summary>
    public const double FaceWidth = 0.15;

    /// <summary>Per point: distance to the polyline, segment index and arc length from the polyline's first joint.</summary>
    public readonly record struct Projection(double Distance, int Segment, double Arc);

    /// <summary>The polyline with zero-length segments removed (the reference's <c>clean</c>): a point is kept when it moves
    /// more than 1e-6 from the point before it in the input.</summary>
    public static double[] Clean(double[] joints)
    {
        int n = joints.Length / 3;
        List<double> kept = new(joints.Length);
        for (int i = 0; i < n; i++)
        {
            bool keep = i == 0;
            if (!keep)
            {
                double dx = joints[3 * i] - joints[3 * (i - 1)];
                double dy = joints[3 * i + 1] - joints[3 * (i - 1) + 1];
                double dz = joints[3 * i + 2] - joints[3 * (i - 1) + 2];
                keep = Math.Sqrt(dx * dx + dy * dy + dz * dz) > 1e-6;
            }

            if (keep)
            {
                kept.Add(joints[3 * i]);
                kept.Add(joints[3 * i + 1]);
                kept.Add(joints[3 * i + 2]);
            }
        }

        return kept.ToArray();
    }

    /// <summary>Projects each point onto the polyline <paramref name="joints"/> (xyz triples, at least two joints).</summary>
    public static Projection[] Project(double[] points, double[] joints)
    {
        int m = points.Length / 3;
        int segments = joints.Length / 3 - 1;
        double[] segLength = new double[segments];
        double[] cum = new double[segments + 1];
        for (int k = 0; k < segments; k++)
        {
            double dx = joints[3 * (k + 1)] - joints[3 * k];
            double dy = joints[3 * (k + 1) + 1] - joints[3 * k + 1];
            double dz = joints[3 * (k + 1) + 2] - joints[3 * k + 2];
            segLength[k] = Math.Sqrt(dx * dx + dy * dy + dz * dz);
            cum[k + 1] = cum[k] + segLength[k];
        }

        Projection[] result = new Projection[m];
        for (int i = 0; i < m; i++)
        {
            double px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
            double best = double.PositiveInfinity;
            int bestSeg = 0;
            double bestArc = 0;
            for (int k = 0; k < segments; k++)
            {
                double ax = joints[3 * k], ay = joints[3 * k + 1], az = joints[3 * k + 2];
                double abx = joints[3 * (k + 1)] - ax, aby = joints[3 * (k + 1) + 1] - ay, abz = joints[3 * (k + 1) + 2] - az;
                double l2 = abx * abx + aby * aby + abz * abz;
                double u = ((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / l2;
                u = Math.Clamp(u, 0.0, 1.0);
                double qx = px - (ax + u * abx), qy = py - (ay + u * aby), qz = pz - (az + u * abz);
                double dist = Math.Sqrt(qx * qx + qy * qy + qz * qz);
                if (dist < best)
                {
                    best = dist;
                    bestSeg = k;
                    bestArc = cum[k] + u * segLength[k];
                }
            }

            result[i] = new Projection(best, bestSeg, bestArc);
        }

        return result;
    }

    /// <summary>The chain a card follows (its mean distance over the lower 60% of the points, under <see cref="Assign"/> face
    /// widths, is smallest), or -1 for a card that stays with the head.</summary>
    public static int AssignChain(double[] points, IReadOnlyList<double[]> chains, double faceWidth)
    {
        int m = points.Length / 3;
        int from = (int)(0.4 * m);
        double[] lower = points[(3 * from)..];
        int best = -1;
        double bestDistance = Assign * faceWidth;
        for (int c = 0; c < chains.Count; c++)
        {
            Projection[] projections = Project(lower, chains[c]);
            double mean = 0;
            foreach (Projection p in projections)
            {
                mean += p.Distance;
            }

            mean /= projections.Length;
            if (mean < bestDistance)
            {
                best = c;
                bestDistance = mean;
            }
        }

        return best;
    }

    /// <summary>
    /// Binding of each point to the segments of one chain: the segment it lies on, its weight on the chain (0 at the root,
    /// rising to 1 over <see cref="Ramp"/> face widths of arc length), and the neighbouring segment with its blend factor.
    /// A point at parameter u on segment k follows k with alpha and the neighbour with 1 - alpha, where alpha = 0.5 + u toward
    /// k - 1 for u below 0.5 and 1.5 - u toward k + 1 above 0.5, so the binding is continuous across joints.
    /// </summary>
    public static (int[] Segment, double[] Weight, int[] Segment2, double[] Alpha) Bind(double[] points, double[] joints, double faceWidth)
    {
        Projection[] projections = Project(points, joints);
        int m = projections.Length;
        int[] segment = new int[m];
        double[] weight = new double[m];
        int[] segment2 = new int[m];
        double[] alpha = new double[m];
        int segments = joints.Length / 3 - 1;
        for (int i = 0; i < m; i++)
        {
            Projection p = projections[i];
            segment[i] = p.Segment;
            weight[i] = Math.Clamp(p.Arc / (Ramp * faceWidth), 0.0, 1.0);
            segment2[i] = p.Segment;
            alpha[i] = 1.0;

            double px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
            int k = p.Segment;
            double ax = joints[3 * k], ay = joints[3 * k + 1], az = joints[3 * k + 2];
            double abx = joints[3 * (k + 1)] - ax, aby = joints[3 * (k + 1) + 1] - ay, abz = joints[3 * (k + 1) + 2] - az;
            double u = Math.Clamp(((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / (abx * abx + aby * aby + abz * abz), 0.0, 1.0);
            if (u < 0.5 && k > 0)
            {
                segment2[i] = k - 1;
                alpha[i] = 0.5 + u;
            }
            else if (u > 0.5 && k < segments - 1)
            {
                segment2[i] = k + 1;
                alpha[i] = 1.5 - u;
            }
        }

        return (segment, weight, segment2, alpha);
    }

    /// <summary>Centreline of a card: <see cref="CentrelineSamples"/> arc-length samples of its control points (xyz triples).</summary>
    public static double[] Centreline(double[] controlPoints)
    {
        int n = controlPoints.Length / 3;
        double[] cum = new double[n];
        for (int i = 1; i < n; i++)
        {
            double dx = controlPoints[3 * i] - controlPoints[3 * (i - 1)];
            double dy = controlPoints[3 * i + 1] - controlPoints[3 * (i - 1) + 1];
            double dz = controlPoints[3 * i + 2] - controlPoints[3 * (i - 1) + 2];
            cum[i] = cum[i - 1] + Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }

        double total = cum[n - 1];
        double[] result = new double[CentrelineSamples * 3];
        int j = 0;
        for (int s = 0; s < CentrelineSamples; s++)
        {
            double t = total * s / (CentrelineSamples - 1);
            while (j < n - 2 && cum[j + 1] < t)
            {
                j++;
            }

            double span = cum[j + 1] - cum[j];
            double u = span > 0 ? Math.Clamp((t - cum[j]) / span, 0.0, 1.0) : 0.0;
            for (int c = 0; c < 3; c++)
            {
                result[3 * s + c] = controlPoints[3 * j + c] + u * (controlPoints[3 * (j + 1) + c] - controlPoints[3 * j + c]);
            }
        }

        return result;
    }
}
