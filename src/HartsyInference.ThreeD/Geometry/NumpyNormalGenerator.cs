using HartsyInference.Core.Numerics;

namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// NumPy's <c>np.random.default_rng(seed).normal(loc, scale)</c> stream for a single 32-bit seed: SeedSequence, PCG64
/// (XSL-RR output) and the 256-layer ziggurat normal sampler, ported from numpy 2.5 so draws are bit-identical.
/// </summary>
internal sealed class NumpyNormalGenerator
{
    private const double ZigguratNorR = 3.6541528853610087963519472518;
    private const double ZigguratNorInvR = 0.27366123732975827203338247596;

    private readonly NumpyPcg64 _bits;

    public NumpyNormalGenerator(uint seed) => _bits = new NumpyPcg64(seed);

    /// <summary>One draw of <c>Generator.normal(loc, scale)</c>.</summary>
    public double Normal(double loc, double scale) => loc + scale * StandardNormal();

    private ulong NextUInt64() => _bits.NextUInt64();

    private double NextDouble() => _bits.NextDouble();

    private double StandardNormal()
    {
        while (true)
        {
            ulong r = NextUInt64();
            int idx = (int)(r & 0xff);
            r >>= 8;
            int sign = (int)(r & 1);
            ulong rabs = (r >> 1) & 0x000fffffffffffff;
            double x = rabs * NumpyZigguratTables.Wi[idx];
            if (sign != 0)
                x = -x;
            if (rabs < NumpyZigguratTables.Ki[idx])
                return x;
            if (idx == 0)
            {
                while (true)
                {
                    double xx = -ZigguratNorInvR * Math.Log(1.0 - NextDouble());
                    double yy = -Math.Log(1.0 - NextDouble());
                    if (yy + yy > xx * xx)
                        return ((rabs >> 8) & 1) != 0 ? -(ZigguratNorR + xx) : ZigguratNorR + xx;
                }
            }
            double[] fi = NumpyZigguratTables.Fi;
            if ((fi[idx - 1] - fi[idx]) * NextDouble() + fi[idx] < Math.Exp(-0.5 * x * x))
                return x;
        }
    }
}
