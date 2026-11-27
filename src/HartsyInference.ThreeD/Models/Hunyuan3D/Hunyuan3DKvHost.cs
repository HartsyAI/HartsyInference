using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Host copy of the geo-decoder's head-major keys and values, for picking the latents a group of queries attends to.</summary>
internal sealed unsafe class Hunyuan3DKvHost
{
    private readonly float[] _k, _v;
    private readonly int _heads, _latents, _headDim;

    internal Hunyuan3DKvHost(Tensor kP, Tensor vP)
    {
        if (kP.Shape.Rank != 4 || kP.Shape[0] != 1 || !kP.Shape.Equals(vP.Shape))
            throw new ArgumentException($"Keys and values must both be [1,H,N,D]; got {kP.Shape} and {vP.Shape}.");
        _heads = (int)kP.Shape[1]; _latents = (int)kP.Shape[2]; _headDim = (int)kP.Shape[3];
        _k = Copy(kP); _v = Copy(vP);
    }

    internal int Latents => _latents;

    /// <summary>Per head, the <paramref name="topK"/> keys best matching the mean of every <paramref name="stride"/>-th row of <paramref name="q"/> <c>[1,H,rows,D]</c>.</summary>
    /// <returns>The keys and their values as <c>[1,H,topK,D]</c> tensors the caller owns.</returns>
    /// <remarks>Mean-of-dot-products equals the dot product with the mean, which is why the mean is formed first.</remarks>
    internal (Tensor K, Tensor V) SelectTopK(Tensor q, int rows, int stride, int topK)
    {
        if (stride <= 0 || topK <= 0 || topK > _latents)
            throw new ArgumentOutOfRangeException(nameof(topK), topK, $"Need 0 < topK <= {_latents} and a positive stride.");
        int d = _headDim;
        float* qp = (float*)q.DataPointer;
        float[] mean = new float[d];
        float[] sim = new float[_latents];
        int[] order = new int[_latents];
        Tensor kSel = new(new TensorShape(1, _heads, topK, d), DType.F32), vSel = new(new TensorShape(1, _heads, topK, d), DType.F32);
        float* kd = (float*)kSel.DataPointer, vd = (float*)vSel.DataPointer;
        int sampled = (rows + stride - 1) / stride;
        for (int h = 0; h < _heads; h++)
        {
            Array.Clear(mean);
            for (int s = 0; s < rows; s += stride)
            {
                float* row = qp + ((long)h * rows + s) * d;
                for (int j = 0; j < d; j++) mean[j] += row[j];
            }

            for (int j = 0; j < d; j++) mean[j] /= sampled;
            for (int n = 0; n < _latents; n++)
            {
                ReadOnlySpan<float> key = _k.AsSpan(((h * _latents) + n) * d, d);
                float dot = 0f;
                for (int j = 0; j < d; j++) dot += mean[j] * key[j];
                sim[n] = dot; order[n] = n;
            }

            Array.Sort(order, (a, b) => sim[b].CompareTo(sim[a]));
            for (int t = 0; t < topK; t++)
            {
                int n = order[t];
                _k.AsSpan(((h * _latents) + n) * d, d).CopyTo(new Span<float>(kd + ((long)h * topK + t) * d, d));
                _v.AsSpan(((h * _latents) + n) * d, d).CopyTo(new Span<float>(vd + ((long)h * topK + t) * d, d));
            }
        }

        return (kSel, vSel);
    }

    private static float[] Copy(Tensor t)
    {
        float[] copy = new float[t.ElementCount];
        new ReadOnlySpan<float>((float*)t.DataPointer, copy.Length).CopyTo(copy);
        return copy;
    }
}
