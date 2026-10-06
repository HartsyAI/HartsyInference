using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Layer primitives of the ControlFoley network on F32 <c>[B, T, C]</c> tensors: linear and channel-last
/// convolutions, SwiGLU feed-forwards, adaLN modulation, interleaved RoPE and the resampling used by the condition
/// preprocessing. Matrix work runs on the backend; the cheap element-wise and layout steps run on the host.</summary>
internal static unsafe class ControlFoleyOps
{
    internal const float SeluScale = 1.0507009873554805f;
    internal const float SeluAlpha = 1.6732632423543772f;

    /// <summary>The epsilon <c>nn.RMSNorm(eps=None)</c> resolves to for float32 inputs.</summary>
    internal const float RmsNormEps = 1.1920929e-07f;

    internal const float LayerNormEps = 1e-5f;

    internal static Tensor New(long b, long t, long d) => new(new TensorShape(b, t, d), DType.F32);

    internal static Tensor Clone(Tensor x)
    {
        Tensor o = new(x.Shape, DType.F32);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)o.DataPointer, x.ElementCount * 4, x.ElementCount * 4);
        return o;
    }

    internal static Tensor Linear(IBackend backend, Tensor x, Tensor weight, Tensor? bias)
    {
        Tensor o = New(x.Shape[0], x.Shape[1], weight.Shape[0]);
        backend.Linear(o, x, weight, bias);
        return o;
    }

    /// <summary>Port of <c>ChannelLastConv1d</c>: stride-1 convolution over the time axis of <c>[B, T, C]</c> with
    /// symmetric <c>(k-1)/2</c> padding (the only padding the released model uses).</summary>
    internal static Tensor ConvLastChannel(IBackend backend, Tensor x, Tensor weight, Tensor? bias)
    {
        if (weight.DType == DType.BF16)
        {
            return ConvUnfolded(backend, x, weight, bias);
        }

        int b = (int)x.Shape[0], t = (int)x.Shape[1], c = (int)x.Shape[2];
        int co = (int)weight.Shape[0], k = (int)weight.Shape[2];
        if (weight.Shape[1] != c || k % 2 == 0)
        {
            throw new ArgumentException($"Conv weight {weight.Shape} does not fit input {x.Shape} with 'same' padding.");
        }

        int pad = (k - 1) / 2;
        using Tensor channelsFirst = new(new TensorShape(b, c, t), DType.F32);
        float* src = (float*)x.DataPointer, dst = (float*)channelsFirst.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                for (int ci = 0; ci < c; ci++)
                {
                    dst[((long)bi * c + ci) * t + ti] = src[((long)bi * t + ti) * c + ci];
                }
            }
        }

        using Tensor conv = new(new TensorShape(b, co, t), DType.F32);
        backend.Conv1d(conv, channelsFirst, weight, bias, 1, pad, pad, 1, 1);
        Tensor o = New(b, t, co);
        float* cp = (float*)conv.DataPointer, op = (float*)o.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            for (int ci = 0; ci < co; ci++)
            {
                for (int ti = 0; ti < t; ti++)
                {
                    op[((long)bi * t + ti) * co + ci] = cp[((long)bi * co + ci) * t + ti];
                }
            }
        }

        return o;
    }

    /// <summary>Converts a checkpoint tensor for use: a BF16 conv weight <c>[out, in, k]</c> becomes <c>[out, k, in]</c> so
    /// the convolution runs as one Linear over the unfolded input (half the memory of F32, no conv kernel for BF16);
    /// everything else is widened to F32 except BF16 matrices, which stay as stored. Created tensors go to <paramref name="owned"/>.</summary>
    internal static Tensor PrepareWeight(Tensor t, List<Tensor> owned)
    {
        if (t.DType == DType.BF16 && t.Shape.Rank == 2)
        {
            return t;
        }

        if (t.DType == DType.BF16 && t.Shape.Rank == 3)
        {
            int co = (int)t.Shape[0], ci = (int)t.Shape[1], k = (int)t.Shape[2];
            Tensor o = new(new TensorShape(co, k, ci), DType.BF16);
            ushort* src = (ushort*)t.DataPointer, dst = (ushort*)o.DataPointer;
            for (int a = 0; a < co; a++)
            {
                for (int j = 0; j < k; j++)
                {
                    for (int i = 0; i < ci; i++)
                    {
                        dst[((long)a * k + j) * ci + i] = src[((long)a * ci + i) * k + j];
                    }
                }
            }

            owned.Add(o);
            return o;
        }

        Tensor f = WhisperOps.EnsureF32(t);
        if (!ReferenceEquals(f, t))
        {
            owned.Add(f);
        }

        return f;
    }

    /// <summary>Same-padded convolution with a BF16 weight laid out <c>[out, k, in]</c>: unfold k shifted copies of the
    /// input along channels and apply one Linear.</summary>
    private static Tensor ConvUnfolded(IBackend backend, Tensor x, Tensor weight, Tensor? bias)
    {
        int b = (int)x.Shape[0], t = (int)x.Shape[1], c = (int)x.Shape[2];
        int co = (int)weight.Shape[0], k = (int)weight.Shape[1];
        if (weight.Shape[2] != c || k % 2 == 0)
        {
            throw new ArgumentException($"Conv weight {weight.Shape} does not fit input {x.Shape} with 'same' padding.");
        }

        int pad = (k - 1) / 2;
        using Tensor unfolded = new(new TensorShape(b, t, k * c), DType.F32);
        float* src = (float*)x.DataPointer, dst = (float*)unfolded.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                float* row = dst + ((long)bi * t + ti) * k * c;
                for (int j = 0; j < k; j++)
                {
                    int s = ti + j - pad;
                    if (s < 0 || s >= t)
                    {
                        new Span<float>(row + (long)j * c, c).Clear();
                    }
                    else
                    {
                        Buffer.MemoryCopy(src + ((long)bi * t + s) * c, row + (long)j * c, c * 4L, c * 4L);
                    }
                }
            }
        }

        using Tensor flat = weight.Reshape(new TensorShape(co, k * c));
        Tensor o = New(b, t, co);
        backend.Linear(o, unfolded, flat, bias);
        return o;
    }

    /// <summary>Linear when the weight is 2-D, <see cref="ConvLastChannel"/> when it is 3-D.</summary>
    internal static Tensor Project(IBackend backend, Tensor x, Tensor weight, Tensor? bias)
        => weight.Shape.Rank == 3 ? ConvLastChannel(backend, x, weight, bias) : Linear(backend, x, weight, bias);

    /// <summary>SwiGLU feed-forward (<c>MLP</c> / <c>ConvMLP</c>): <c>w2(silu(w1 x) * w3 x)</c>; the weight rank picks
    /// linear or convolutional projections.</summary>
    internal static Tensor SwiGlu(IBackend backend, Tensor x, Tensor w1, Tensor w2, Tensor w3)
    {
        using Tensor gate = Project(backend, x, w1, null);
        using Tensor up = Project(backend, x, w3, null);
        backend.Silu(gate, gate);
        backend.Mul(gate, gate, up);
        return Project(backend, gate, w2, null);
    }

    internal static void Selu(Tensor x)
    {
        float* p = (float*)x.DataPointer;
        for (long i = 0; i < x.ElementCount; i++)
        {
            float v = p[i];
            p[i] = SeluScale * (v > 0f ? v : SeluAlpha * (MathF.Exp(v) - 1f));
        }
    }

    internal static Tensor Silu(IBackend backend, Tensor x)
    {
        Tensor o = new(x.Shape, DType.F32);
        backend.Silu(o, x);
        return o;
    }

    /// <summary>Elementwise <c>a += b</c> where <paramref name="b"/> has either the same shape or a time length of 1
    /// (broadcast over time); batch and channel counts must match.</summary>
    internal static void AddBroadcastTime(Tensor a, Tensor b)
    {
        int bs = (int)a.Shape[0], t = (int)a.Shape[1], d = (int)a.Shape[2];
        int tb = (int)b.Shape[1];
        if (b.Shape[0] != bs || b.Shape[2] != d || (tb != t && tb != 1))
        {
            throw new ArgumentException($"Cannot add {b.Shape} to {a.Shape}.");
        }

        float* ap = (float*)a.DataPointer, bp = (float*)b.DataPointer;
        for (int bi = 0; bi < bs; bi++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                float* ar = ap + ((long)bi * t + ti) * d;
                float* br = bp + ((long)bi * tb + (tb == 1 ? 0 : ti)) * d;
                for (int c = 0; c < d; c++)
                {
                    ar[c] += br[c];
                }
            }
        }
    }

    /// <summary>A strided window over a modulation tensor <c>[B, Tc, n * D]</c>: chunk <c>index</c> of its last axis,
    /// broadcast over time when <c>Tc == 1</c>.</summary>
    internal readonly struct Chunk(float* data, int timeLength, int rowWidth, int offset, int width)
    {
        internal float* Row(int batch, int t) => data + ((long)batch * timeLength + (timeLength == 1 ? 0 : t)) * rowWidth + offset;

        internal int Width => width;
    }

    internal static Chunk ChunkOf(Tensor m, int index, int chunks)
    {
        int width = (int)m.Shape[2] / chunks;
        return new Chunk((float*)m.DataPointer, (int)m.Shape[1], (int)m.Shape[2], index * width, width);
    }

    /// <summary><c>LayerNorm(elementwise_affine=False)(x) * (1 + scale) + shift</c> (the official <c>modulate</c>).</summary>
    internal static Tensor NormModulate(IBackend backend, Tensor x, Chunk shift, Chunk scale)
    {
        Tensor o = new(x.Shape, DType.F32);
        backend.LayerNormNoAffine(o, x, LayerNormEps);
        int b = (int)x.Shape[0], t = (int)x.Shape[1], d = (int)x.Shape[2];
        float* op = (float*)o.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                float* row = op + ((long)bi * t + ti) * d;
                float* sh = shift.Row(bi, ti), sc = scale.Row(bi, ti);
                for (int c = 0; c < d; c++)
                {
                    row[c] = row[c] * (1f + sc[c]) + sh[c];
                }
            }
        }

        return o;
    }

    /// <summary><c>x += y * gate</c>; consumes nothing, <paramref name="y"/> keeps its owner.</summary>
    internal static void GatedAdd(Tensor x, Tensor y, Chunk gate)
    {
        int b = (int)x.Shape[0], t = (int)x.Shape[1], d = (int)x.Shape[2];
        float* xp = (float*)x.DataPointer, yp = (float*)y.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                long at = ((long)bi * t + ti) * d;
                float* g = gate.Row(bi, ti);
                for (int c = 0; c < d; c++)
                {
                    xp[at + c] += yp[at + c] * g[c];
                }
            }
        }
    }

    /// <summary>F.interpolate(mode='linear', align_corners=False) over the time axis of <c>[B, T, C]</c>.</summary>
    internal static Tensor InterpolateLinear(Tensor x, int outLength)
    {
        int b = (int)x.Shape[0], t = (int)x.Shape[1], c = (int)x.Shape[2];
        Tensor o = New(b, outLength, c);
        float scale = (float)t / outLength;
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int i = 0; i < outLength; i++)
        {
            float src = MathF.Max(scale * (i + 0.5f) - 0.5f, 0f);
            int i0 = Math.Min((int)src, t - 1);
            int i1 = Math.Min(i0 + 1, t - 1);
            float lambda1 = src - i0, lambda0 = 1f - lambda1;
            for (int bi = 0; bi < b; bi++)
            {
                float* a = xp + ((long)bi * t + i0) * c, a1 = xp + ((long)bi * t + i1) * c;
                float* r = op + ((long)bi * outLength + i) * c;
                for (int ci = 0; ci < c; ci++)
                {
                    r[ci] = lambda0 * a[ci] + lambda1 * a1[ci];
                }
            }
        }

        return o;
    }

    /// <summary>F.interpolate(mode='nearest-exact') over the time axis of <c>[B, T, C]</c>.</summary>
    internal static Tensor InterpolateNearestExact(Tensor x, int outLength)
    {
        int b = (int)x.Shape[0], t = (int)x.Shape[1], c = (int)x.Shape[2];
        Tensor o = New(b, outLength, c);
        float scale = (float)t / outLength;
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int i = 0; i < outLength; i++)
        {
            int src = outLength == t ? i : Math.Min((int)MathF.Floor((i + 0.5f) * scale), t - 1);
            for (int bi = 0; bi < b; bi++)
            {
                Buffer.MemoryCopy(xp + ((long)bi * t + src) * c, op + ((long)bi * outLength + i) * c, c * 4L, c * 4L);
            }
        }

        return o;
    }

    /// <summary>Mean over the time axis: <c>[B, T, C]</c> to <c>[B, 1, C]</c>.</summary>
    internal static Tensor MeanOverTime(Tensor x)
    {
        int b = (int)x.Shape[0], t = (int)x.Shape[1], c = (int)x.Shape[2];
        Tensor o = New(b, 1, c);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            for (int ci = 0; ci < c; ci++)
            {
                float sum = 0f;
                for (int ti = 0; ti < t; ti++)
                {
                    sum += xp[((long)bi * t + ti) * c + ci];
                }

                op[(long)bi * c + ci] = sum / t;
            }
        }

        return o;
    }

    /// <summary>Repeats a batch-1 tensor to <paramref name="batch"/> rows.</summary>
    internal static Tensor ExpandBatch(Tensor x, int batch)
    {
        if (x.Shape[0] != 1)
        {
            throw new ArgumentException($"Only a batch-1 tensor can be expanded, got {x.Shape}.");
        }

        Tensor o = New(batch, x.Shape[1], x.Shape[2]);
        long bytes = x.ElementCount * 4;
        for (int bi = 0; bi < batch; bi++)
        {
            Buffer.MemoryCopy((void*)x.DataPointer, (byte*)o.DataPointer + bi * bytes, bytes, bytes);
        }

        return o;
    }

    internal static float[] ToHost(Tensor x) => new ReadOnlySpan<float>((void*)x.DataPointer, (int)x.ElementCount).ToArray();
}
