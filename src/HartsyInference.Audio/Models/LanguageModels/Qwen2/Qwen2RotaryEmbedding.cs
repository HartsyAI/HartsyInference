using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.LanguageModels.Qwen2;

/// <summary>Precomputed Llama-style rotary positional embedding (split-half convention).
/// Same math as the rest of the Llama / Qwen family — for each head vector
/// <c>x ∈ R^D</c> at absolute position <c>p</c>:
/// <code>
///   half = D / 2
///   freqs[k] = 1 / theta^(2k / D),  k ∈ [0, half)
///   cos[p, k] = cos(p · freqs[k])
///   sin[p, k] = sin(p · freqs[k])
///   y[k]        = x[k]      · cos[p, k] - x[k+half] · sin[p, k]
///   y[k+half]   = x[k+half] · cos[p, k] + x[k]      · sin[p, k]
/// </code>
///
/// <para>For AR generation we apply RoPE at the absolute position currently being
/// decoded — <c>posStart..posStart + T - 1</c> where <c>posStart</c> is the KV-cache
/// length before this step. The table is lazily grown to cover the longest position
/// seen so far, rounded up to a power of two to keep growth amortized.</para></summary>
internal sealed unsafe class Qwen2RotaryEmbedding
{
    private readonly int _headDim;
    private readonly float _theta;
    private readonly int _hardMax;
    private float[]? _cos;
    private float[]? _sin;
    private int _builtFor;

    public int HeadDim => _headDim;

    public Qwen2RotaryEmbedding(int headDim, float theta, int maxPositionEmbeddings)
    {
        if ((headDim & 1) != 0) throw new ArgumentException($"head_dim must be even, got {headDim}.");
        _headDim = headDim;
        _theta = theta;
        _hardMax = maxPositionEmbeddings;
    }

    /// <summary>Ensure cos/sin are built for at least <paramref name="posEnd"/> positions
    /// (inclusive of any in-flight position the caller is about to write into).</summary>
    public void EnsureBuilt(int posEnd)
    {
        if (posEnd > _hardMax) throw new ArgumentException($"Position {posEnd} exceeds max_position_embeddings={_hardMax}.");
        if (_cos is not null && _builtFor >= posEnd) return;

        int half = _headDim / 2;
        int targetLen = Math.Max(posEnd, _builtFor);
        int rounded = 1;
        while (rounded < targetLen) rounded <<= 1;
        targetLen = Math.Min(rounded, _hardMax);

        float[] cos = new float[targetLen * half];
        float[] sin = new float[targetLen * half];

        // freqs[k] = 1 / theta^(2k / D) — Llama / Qwen convention.
        for (int p = 0; p < targetLen; p++)
        {
            for (int k = 0; k < half; k++)
            {
                double freq = 1.0 / Math.Pow(_theta, (double)(2 * k) / _headDim);
                double angle = p * freq;
                cos[p * half + k] = (float)Math.Cos(angle);
                sin[p * half + k] = (float)Math.Sin(angle);
            }
        }

        _cos = cos;
        _sin = sin;
        _builtFor = targetLen;
    }

    /// <summary>Apply RoPE in place to a <c>[B, H, T, D]</c> multi-head tensor whose
    /// <c>T</c> rows are positioned at <c>posStart, posStart+1, ..., posStart+T-1</c>.
    /// Caller must have invoked <see cref="EnsureBuilt"/> for <c>posStart + T</c>.</summary>
    public void Apply(Tensor mh, int batch, int heads, int t, int posStart)
    {
        if (_cos is null || _sin is null) throw new InvalidOperationException("RoPE table not built. Call EnsureBuilt first.");
        float[] cos = _cos;
        float[] sin = _sin;
        int half = _headDim / 2;
        float* p = (float*)mh.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < heads; h++)
            {
                for (int s = 0; s < t; s++)
                {
                    long off = (((long)b * heads + h) * t + s) * _headDim;
                    int posOff = (posStart + s) * half;
                    for (int i = 0; i < half; i++)
                    {
                        float c = cos[posOff + i];
                        float si = sin[posOff + i];
                        float a = p[off + i];
                        float bb = p[off + i + half];
                        p[off + i] = a * c - bb * si;
                        p[off + i + half] = bb * c + a * si;
                    }
                }
            }
        }
    }
}
