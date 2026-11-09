using System.Numerics;

namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// NumPy's <c>np.random.default_rng(seed).normal(loc, scale)</c> stream for a single 32-bit seed: SeedSequence, PCG64
/// (XSL-RR output) and the 256-layer ziggurat normal sampler, ported from numpy 2.5 so draws are bit-identical.
/// </summary>
internal sealed class NumpyNormalGenerator
{
    private const uint InitA = 0x43b0d7e5;
    private const uint MultA = 0x931e8875;
    private const uint InitB = 0x8b51f9dd;
    private const uint MultB = 0x58f38ded;
    private const uint MixMultL = 0xca01f9dd;
    private const uint MixMultR = 0x4973f715;
    private const int XShift = 16;
    private const double ZigguratNorR = 3.6541528853610087963519472518;
    private const double ZigguratNorInvR = 0.27366123732975827203338247596;
    private const double TwoToMinus53 = 1.0 / 9007199254740992.0;
    private static readonly UInt128 PcgMultiplier = ((UInt128)2549297995355413924UL << 64) | 4865540595714422341UL;

    private UInt128 _state;
    private readonly UInt128 _inc;

    public NumpyNormalGenerator(uint seed)
    {
        ulong[] words = SeedWords(seed);
        UInt128 initState = ((UInt128)words[0] << 64) | words[1];
        UInt128 initSeq = ((UInt128)words[2] << 64) | words[3];
        _inc = (initSeq << 1) | 1;
        Step();
        _state += initState;
        Step();
    }

    /// <summary>One draw of <c>Generator.normal(loc, scale)</c>.</summary>
    public double Normal(double loc, double scale) => loc + scale * StandardNormal();

    private void Step() => _state = _state * PcgMultiplier + _inc;

    private ulong NextUInt64()
    {
        Step();
        ulong hi = (ulong)(_state >> 64);
        ulong lo = (ulong)_state;
        return BitOperations.RotateRight(hi ^ lo, (int)(hi >> 58));
    }

    private double NextDouble() => (NextUInt64() >> 11) * TwoToMinus53;

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

    /// <summary>SeedSequence(seed).generate_state(4, uint64) for an entropy of one 32-bit word.</summary>
    private static ulong[] SeedWords(uint seed)
    {
        uint[] pool = new uint[4];
        uint hashConst = InitA;
        for (int i = 0; i < pool.Length; i++)
            pool[i] = HashMix(i == 0 ? seed : 0u, ref hashConst);
        for (int src = 0; src < pool.Length; src++)
            for (int dst = 0; dst < pool.Length; dst++)
                if (src != dst)
                    pool[dst] = Mix(pool[dst], HashMix(pool[src], ref hashConst));

        uint[] words32 = new uint[8];
        uint constB = InitB;
        for (int i = 0; i < words32.Length; i++)
        {
            uint v = pool[i % pool.Length] ^ constB;
            constB *= MultB;
            v *= constB;
            words32[i] = v ^ (v >> XShift);
        }
        ulong[] words64 = new ulong[4];
        for (int j = 0; j < words64.Length; j++)
            words64[j] = words32[2 * j] | ((ulong)words32[2 * j + 1] << 32);
        return words64;
    }

    private static uint HashMix(uint value, ref uint hashConst)
    {
        value ^= hashConst;
        hashConst *= MultA;
        value *= hashConst;
        return value ^ (value >> XShift);
    }

    private static uint Mix(uint x, uint y)
    {
        uint result = MixMultL * x - MixMultR * y;
        return result ^ (result >> XShift);
    }
}
