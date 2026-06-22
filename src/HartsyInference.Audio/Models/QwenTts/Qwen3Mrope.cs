using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.QwenTts;

/// <summary>3D interleaved multimodal RoPE (mRoPE) for the Qwen3-TTS talker. Unlike plain 1D RoPE there are three
/// position axes (text/time/extra). The <c>head_dim/2</c> frequency pairs are partitioned into bands by
/// <c>mrope_section</c> (e.g. [24,20,20] summing to 64 = head_dim/2); each band reads its angle from one position
/// axis. <c>interleaved=true</c> means the per-pair rotation is the GPT-J convention over adjacent dims
/// <c>(2i, 2i+1)</c> — matching <see cref="HartsyInference.Audio.Models.Moonshine.RotaryEmbedding"/> — rather than
/// the split-halves form.
///
/// <para>For pure-text TTS the time/extra axes default to the text-position axis (the talker advances one codec
/// frame per step; without explicit per-second alignment all three axes share the running text position). The
/// <see cref="ForPositions"/> factory builds the per-position cos/sin tables once; <see cref="ApplyInPlace"/>
/// rotates a <c>[1, heads, t, headDim]</c> tensor in place using the band → axis mapping baked into the tables.</para></summary>
public sealed unsafe class Qwen3Mrope
{
    private readonly int _headDim;
    private readonly int _half;
    private readonly double[] _invFreq;     // [half]
    private readonly int[] _bandAxis;       // [half] axis id (0..2) for each frequency pair

    /// <summary>Frames per second used to derive the time axis from a frame index (position_id_per_seconds 13).</summary>
    public int PositionIdPerSeconds { get; }

    public int HeadDim => _headDim;

    public Qwen3Mrope(int headDim, float ropeTheta, ReadOnlySpan<int> mropeSection, int positionIdPerSeconds = 13)
    {
        if (headDim % 2 != 0) throw new ArgumentException($"head_dim must be even, got {headDim}.", nameof(headDim));
        _headDim = headDim;
        _half = headDim / 2;
        PositionIdPerSeconds = positionIdPerSeconds;

        int sum = 0;
        for (int i = 0; i < mropeSection.Length; i++) sum += mropeSection[i];
        if (sum != _half)
            throw new ArgumentException($"mrope_section must sum to head_dim/2 ({_half}), got {sum}.", nameof(mropeSection));

        _invFreq = new double[_half];
        for (int i = 0; i < _half; i++)
            _invFreq[i] = 1.0 / Math.Pow(ropeTheta, (double)(2 * i) / headDim);

        // Interleaved band assignment: section s owns a contiguous run of frequency pairs starting at its offset.
        _bandAxis = new int[_half];
        int off = 0;
        for (int s = 0; s < mropeSection.Length; s++)
        {
            for (int j = 0; j < mropeSection[s]; j++) _bandAxis[off + j] = s;
            off += mropeSection[s];
        }
    }

    /// <summary>Builds cos/sin tables of shape <c>[t, half]</c> from explicit 3-axis position ids. Each
    /// frequency pair reads its angle from the axis its band maps to. <paramref name="posAxis0"/>/1/2 are the
    /// per-step position ids for the text/time/extra axes (length <paramref name="t"/>).</summary>
    public (float[] Cos, float[] Sin) BuildTables(ReadOnlySpan<long> posAxis0, ReadOnlySpan<long> posAxis1, ReadOnlySpan<long> posAxis2, int t)
    {
        if (posAxis0.Length < t || posAxis1.Length < t || posAxis2.Length < t)
            throw new ArgumentException("position-id spans must have at least t entries.");
        float[] cos = new float[t * _half];
        float[] sin = new float[t * _half];
        for (int s = 0; s < t; s++)
        {
            int row = s * _half;
            for (int i = 0; i < _half; i++)
            {
                long pos = _bandAxis[i] switch
                {
                    0 => posAxis0[s],
                    1 => posAxis1[s],
                    _ => posAxis2[s],
                };
                double angle = pos * _invFreq[i];
                cos[row + i] = (float)Math.Cos(angle);
                sin[row + i] = (float)Math.Sin(angle);
            }
        }
        return (cos, sin);
    }

    /// <summary>Pure-text default: all three axes share the running text position <c>posStart + s</c>. Builds
    /// cos/sin tables of shape <c>[t, half]</c>.</summary>
    public (float[] Cos, float[] Sin) ForPositions(int t, int posStart)
    {
        long[] axis = new long[t];
        for (int s = 0; s < t; s++) axis[s] = posStart + s;
        return BuildTables(axis, axis, axis, t);
    }

    /// <summary>Applies interleaved mRoPE in place to <c>[1, heads, t, headDim]</c>. Adjacent dims
    /// <c>(2i, 2i+1)</c> rotate by <c>cos[s][i]/sin[s][i]</c>; trailing dims (none when half·2 == head_dim)
    /// pass through.</summary>
    public void ApplyInPlace(Tensor x, int heads, int t, float[] cos, float[] sin)
    {
        float* p = (float*)x.DataPointer;
        for (int h = 0; h < heads; h++)
            for (int s = 0; s < t; s++)
            {
                long baseOff = ((long)h * t + s) * _headDim;
                int tableOff = s * _half;
                for (int i = 0; i < _half; i++)
                {
                    int dEven = 2 * i;
                    int dOdd = 2 * i + 1;
                    float xe = p[baseOff + dEven];
                    float xo = p[baseOff + dOdd];
                    float c = cos[tableOff + i];
                    float si = sin[tableOff + i];
                    p[baseOff + dEven] = xe * c - xo * si;
                    p[baseOff + dOdd] = xo * c + xe * si;
                }
            }
    }
}
