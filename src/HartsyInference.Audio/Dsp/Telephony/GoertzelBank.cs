using System.Numerics;

namespace HartsyInference.Audio.Dsp.Telephony;

/// <summary>Goertzel power at a fixed set of arbitrary frequencies over one block of samples, with the bins spread
/// across SIMD lanes. Allocation-free after construction; every lane runs the same arithmetic, so the result does not
/// depend on the hardware vector width.</summary>
/// <remarks>The power of a pure tone of amplitude <c>A</c> exactly at a bin's frequency over <c>N</c> samples is
/// <c>(A·N/2)²</c> with a rectangular block, so <c>2·P/N</c> is the energy that tone contributes. Not thread-safe.</remarks>
public sealed class GoertzelBank
{
    private readonly float[] _coefficients;
    private readonly float[] _power;
    private readonly int _count;

    /// <summary>A bank over <paramref name="frequenciesHz"/> at <paramref name="sampleRate"/> Hz.</summary>
    public GoertzelBank(int sampleRate, ReadOnlySpan<double> frequenciesHz)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(sampleRate, 1);
        if (frequenciesHz.IsEmpty)
        {
            throw new ArgumentException("A bank needs at least one frequency.", nameof(frequenciesHz));
        }
        _count = frequenciesHz.Length;
        int width = Vector<float>.Count;
        int padded = (_count + width - 1) / width * width;
        _coefficients = new float[padded];
        _power = new float[padded];
        for (int i = 0; i < _count; i++)
        {
            if (frequenciesHz[i] <= 0 || frequenciesHz[i] >= sampleRate / 2.0)
            {
                throw new ArgumentOutOfRangeException(nameof(frequenciesHz), frequenciesHz[i], "Frequencies must lie in (0, sampleRate/2).");
            }
            _coefficients[i] = (float)(2.0 * Math.Cos(2.0 * Math.PI * frequenciesHz[i] / sampleRate));
        }
    }

    /// <summary>Number of frequencies.</summary>
    public int Count => _count;

    /// <summary>Writes the power at every frequency over <paramref name="block"/> into <paramref name="power"/>
    /// (at least <see cref="Count"/> long).</summary>
    public void Compute(ReadOnlySpan<float> block, Span<float> power)
    {
        if (power.Length < _count)
        {
            throw new ArgumentException($"Power span holds {power.Length}, the bank has {_count} frequencies.", nameof(power));
        }
        int width = Vector<float>.Count;
        for (int offset = 0; offset < _coefficients.Length; offset += width)
        {
            Vector<float> c = new(_coefficients, offset);
            Vector<float> s1 = Vector<float>.Zero;
            Vector<float> s2 = Vector<float>.Zero;
            for (int i = 0; i < block.Length; i++)
            {
                Vector<float> s0 = new Vector<float>(block[i]) + c * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            (s1 * s1 + s2 * s2 - c * s1 * s2).CopyTo(_power, offset);
        }
        _power.AsSpan(0, _count).CopyTo(power);
    }
}
