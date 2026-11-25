using System.Numerics;

namespace HartsyInference.Core.Numerics;

/// <summary>
/// NumPy's <c>np.random.default_rng(seed)</c> bit stream for a single 32-bit seed: SeedSequence into PCG64 (XSL-RR output), ported from
/// numpy 2.5 so draws are bit-identical. Models whose checkpoints depend on a NumPy-drawn constant (hash multipliers, noise) read it from here.
/// </summary>
public sealed class NumpyPcg64
{
    private const uint InitA = 0x43b0d7e5;
    private const uint MultA = 0x931e8875;
    private const uint InitB = 0x8b51f9dd;
    private const uint MultB = 0x58f38ded;
    private const uint MixMultL = 0xca01f9dd;
    private const uint MixMultR = 0x4973f715;
    private const int XShift = 16;
    private const double TwoToMinus53 = 1.0 / 9007199254740992.0;
    private static readonly UInt128 PcgMultiplier = ((UInt128)2549297995355413924UL << 64) | 4865540595714422341UL;

    private readonly UInt128 _inc;
    private UInt128 _state;

    /// <summary>Seeds like <c>default_rng(seed)</c>.</summary>
    public NumpyPcg64(uint seed)
    {
        ulong[] words = SeedWords(seed);
        UInt128 initState = ((UInt128)words[0] << 64) | words[1];
        UInt128 initSeq = ((UInt128)words[2] << 64) | words[3];
        _inc = (initSeq << 1) | 1;
        Step();
        _state += initState;
        Step();
    }

    /// <summary>The next 64 random bits (<c>bit_generator.random_raw()</c>).</summary>
    public ulong NextUInt64()
    {
        Step();
        ulong hi = (ulong)(_state >> 64);
        ulong lo = (ulong)_state;
        return BitOperations.RotateRight(hi ^ lo, (int)(hi >> 58));
    }

    /// <summary>The next double in [0, 1) (<c>Generator.random()</c>).</summary>
    public double NextDouble() => (NextUInt64() >> 11) * TwoToMinus53;

    /// <summary>One draw of <c>Generator.integers(low, highExclusive, dtype=np.int64)</c> by Lemire's method.</summary>
    /// <exception cref="NotSupportedException">The range is 2^32 or narrower, which NumPy draws through a buffered 32-bit path this port does not carry.</exception>
    public long NextInt64(long low, long highExclusive)
    {
        if (highExclusive <= low) throw new ArgumentOutOfRangeException(nameof(highExclusive), "The range must hold at least one value.");
        ulong rng = unchecked((ulong)(highExclusive - low - 1));
        if (rng == 0) return low;
        if (rng <= uint.MaxValue)
            throw new NotSupportedException("Ranges up to 2^32 take NumPy's buffered 32-bit path, which is not ported.");
        if (rng == ulong.MaxValue) return unchecked(low + (long)NextUInt64());
        ulong exclusive = rng + 1;
        UInt128 m = (UInt128)NextUInt64() * exclusive;
        ulong leftover = (ulong)m;
        if (leftover < exclusive)
        {
            ulong threshold = (0UL - exclusive) % exclusive;
            while (leftover < threshold)
            {
                m = (UInt128)NextUInt64() * exclusive;
                leftover = (ulong)m;
            }
        }
        return unchecked(low + (long)(ulong)(m >> 64));
    }

    private void Step() => _state = _state * PcgMultiplier + _inc;

    // SeedSequence(seed).generate_state(4, uint64) for an entropy of one 32-bit word.
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
