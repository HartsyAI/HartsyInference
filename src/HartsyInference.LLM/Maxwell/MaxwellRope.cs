namespace HartsyInference.LLM.Maxwell;

/// <summary>Rotate-half RoPE over a whole head, with cos and sin tabulated once per position.</summary>
internal sealed class MaxwellRope
{
    private readonly int _half;
    private readonly float[] _cos;
    private readonly float[] _sin;

    /// <param name="headDim">Even width of the rotated vector.</param>
    /// <param name="positions">Positions tabulated: <c>[0, positions)</c>.</param>
    /// <param name="theta">Frequency base.</param>
    public MaxwellRope(int headDim, int positions, double theta)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(headDim, 2);
        if (headDim % 2 != 0) throw new ArgumentException("The rotated width must be even.", nameof(headDim));
        ArgumentOutOfRangeException.ThrowIfLessThan(positions, 1);
        _half = headDim / 2;
        _cos = new float[checked(positions * _half)];
        _sin = new float[_cos.Length];
        float[] inverse = new float[_half];
        for (int i = 0; i < _half; i++) inverse[i] = (float)(1.0 / Math.Pow(theta, 2.0 * i / headDim));
        for (int p = 0; p < positions; p++)
        {
            for (int i = 0; i < _half; i++)
            {
                float angle = inverse[i] * p;
                _cos[p * _half + i] = MathF.Cos(angle);
                _sin[p * _half + i] = MathF.Sin(angle);
            }
        }
    }

    /// <summary>Positions the table covers.</summary>
    public int Positions => _cos.Length / _half;

    /// <summary>Rotates one head vector in place for position <paramref name="position"/>.</summary>
    public void Apply(Span<float> head, int position)
    {
        if (head.Length != 2 * _half) throw new ArgumentException($"A head must hold {2 * _half} values.", nameof(head));
        ReadOnlySpan<float> cos = _cos.AsSpan(position * _half, _half), sin = _sin.AsSpan(position * _half, _half);
        for (int i = 0; i < _half; i++)
        {
            float a = head[i], b = head[i + _half];
            head[i] = a * cos[i] - b * sin[i];
            head[i + _half] = b * cos[i] + a * sin[i];
        }
    }
}
