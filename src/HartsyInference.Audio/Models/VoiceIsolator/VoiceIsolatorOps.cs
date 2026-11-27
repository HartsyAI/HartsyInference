using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>The isolator's layout and elementwise steps: backend ops wherever one exists, and host loops only for the steps no
/// backend op covers (the exclusive attention, the frame reversal, the causal depthwise convolution's carry).</summary>
internal static unsafe class VoiceIsolatorOps
{
    /// <summary>Requires a tensor read through a raw pointer to be F32 with exactly <paramref name="count"/> elements.</summary>
    internal static void RequireF32(Tensor tensor, string name, long count)
    {
        if (tensor.DType != DType.F32 || tensor.ElementCount != count)
        {
            throw new InvalidDataException($"{name} must be F32 with {count} elements, got {tensor.DType.Name} {tensor.ElementCount}.");
        }
    }

    /// <summary>Swaps the two leading axes of <c>[a, b, w]</c> into <c>[b, a, w]</c>.</summary>
    public static void SwapLeading(IBackend backend, Tensor output, Tensor input, int a, int b, int w)
        => backend.Permute0213(output, input, a, b, w);

    /// <summary>Splits fused projections <c>[groups * seq, 3 * heads * hd]</c> (q, k, v concatenated per token) into head-major
    /// <c>[groups, heads, seq, hd]</c> tensors.</summary>
    public static void SplitQkvToHeads(IBackend backend, Tensor qkv, Tensor q, Tensor k, Tensor v, int groups, int seq, int heads, int hd)
    {
        int width = heads * hd;
        TensorShape chunk = qkv.Shape.Rank == 3 ? new TensorShape(1, groups * seq, width) : new TensorShape(groups * seq, width);
        Tensor qt = new(chunk, DType.F32);
        Tensor kt = new(chunk, DType.F32);
        Tensor vt = new(chunk, DType.F32);
        backend.Split([qt, kt, vt], qkv, qkv.Shape.Rank - 1);
        backend.Permute0213(q, qt, seq, heads, hd);
        backend.Permute0213(k, kt, seq, heads, hd);
        backend.Permute0213(v, vt, seq, heads, hd);
        qt.Dispose();
        kt.Dispose();
        vt.Dispose();
    }

    /// <summary>Gathers head-major <c>[groups, heads, seq, hd]</c> into tokens <c>[groups * seq, heads * hd]</c>.</summary>
    public static void HeadsToTokens(IBackend backend, Tensor output, Tensor input, int groups, int seq, int heads, int hd)
        => backend.Permute0213(output, input, heads, seq, hd);

    /// <summary>Rotates head-major <c>[groups, heads, seq, hd]</c> in place by the rotate-half RoPE of position <c>firstPosition + s</c>.
    /// The tables are the half-width rows of <see cref="RopeTables"/>, widened to the element layout the backend rotary takes.</summary>
    public static void Rope(IBackend backend, Tensor x, int seq, int hd, int firstPosition, float[] cos, float[] sin)
    {
        Tensor cosRows = WidenedRows(cos, firstPosition, seq, hd);
        Tensor sinRows = WidenedRows(sin, firstPosition, seq, hd);
        backend.Rotary(RotaryCall.HeadMajor(x, cosRows, sinRows, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf));
        cosRows.Dispose();
        sinRows.Dispose();
    }

    /// <summary>Rotates every row of <paramref name="x"/> (<c>[..., seq, hd]</c>) by the one table position <paramref name="position"/>.
    /// Used by the cache compaction, where every kept slot moves by the same distance.</summary>
    public static void RopeFixed(IBackend backend, Tensor x, int seq, int hd, int position, float[] cos, float[] sin)
    {
        int h = hd / 2;
        Tensor cosRows = new(new TensorShape(seq, hd), DType.F32);
        Tensor sinRows = new(new TensorShape(seq, hd), DType.F32);
        float* cp = (float*)cosRows.DataPointer;
        float* sp = (float*)sinRows.DataPointer;
        for (int s = 0; s < seq; s++)
        {
            for (int i = 0; i < h; i++)
            {
                float c = cos[(long)position * h + i];
                float sn = sin[(long)position * h + i];
                cp[(long)s * hd + i] = c;
                cp[(long)s * hd + h + i] = c;
                sp[(long)s * hd + i] = sn;
                sp[(long)s * hd + h + i] = sn;
            }
        }
        backend.Rotary(RotaryCall.HeadMajor(x, cosRows, sinRows, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf));
        cosRows.Dispose();
        sinRows.Dispose();
    }

    /// <summary>The cos and sin tables <c>[positions, hd / 2]</c> of the reference's <c>rope_tables</c> (base 10000).</summary>
    public static (float[] Cos, float[] Sin) RopeTables(int positions, int hd)
    {
        int half = hd / 2;
        float[] cos = new float[(long)positions * half];
        float[] sin = new float[(long)positions * half];
        for (int p = 0; p < positions; p++)
        {
            for (int i = 0; i < half; i++)
            {
                float inv = MathF.Pow(10000f, -(float)i / half);
                float angle = p * inv;
                cos[(long)p * half + i] = MathF.Cos(angle);
                sin[(long)p * half + i] = MathF.Sin(angle);
            }
        }
        return (cos, sin);
    }

    /// <summary>Exclusive attention: removes from every output row (last axis <paramref name="hd"/>) its component along the matching value row.</summary>
    public static void ExcludeValue(Tensor output, Tensor value, long rows, int hd)
    {
        float* o = (float*)output.DataPointer;
        float* v = (float*)value.DataPointer;
        CpuParallel.For((int)rows, rows * hd, r =>
        {
            float* orow = o + (long)r * hd;
            float* vrow = v + (long)r * hd;
            float dot = 0f;
            float norm = 0f;
            for (int i = 0; i < hd; i++)
            {
                dot += orow[i] * vrow[i];
                norm += vrow[i] * vrow[i];
            }
            float coefficient = dot / MathF.Max(norm, 1e-6f);
            for (int i = 0; i < hd; i++)
            {
                orow[i] -= coefficient * vrow[i];
            }
        });
    }

    /// <summary>Splits a <c>[.., 2 * inner]</c> tensor into its two halves along the last axis.</summary>
    public static void SplitHalves(IBackend backend, Tensor input, Tensor first, Tensor second)
        => backend.Split([first, second], input, input.Shape.Rank - 1);

    /// <summary>Splits <c>[.., dtRank + 2 * states]</c> into the step input <c>[.., dtRank]</c>, B and C <c>[.., states]</c>.</summary>
    public static void SplitDtBc(IBackend backend, Tensor input, Tensor dt, Tensor b, Tensor c)
        => backend.Split([dt, b, c], input, input.Shape.Rank - 1);

    /// <summary>Causal depthwise convolution plus SiLU over <c>[1, rows * frames, channels]</c>, on the backend. The result is a new
    /// tensor of the same shape. <paramref name="history"/> holds the <c>taps - 1</c> frames before the first one per row
    /// (<c>[rows, taps - 1, channels]</c>) and is rewritten with the last ones; null means zeros before the start and keeps nothing.
    /// The frames are gathered frame-major, convolved with the backend's depthwise op, biased with a broadcast copy and activated.</summary>
    public static Tensor CausalDepthwiseSiluOn(IBackend backend, Tensor x, Tensor weight, Tensor bias, Tensor? history,
        int rows, int frames, int channels, int taps)
    {
        int carry = taps - 1;
        int total = carry + frames;
        // Frame-major input [total, rows, channels]: the carried frames, then the new ones. Zeros fill whatever no history covers.
        Tensor padded = new(new TensorShape(total, rows, channels), DType.F32);
        backend.Fill(padded, 0f);
        if (history is not null)
        {
            backend.CopyStrided(padded, history, new StridedCopySpec(
                StridedView.Of(padded.Shape).Slice(0, 0, carry),
                StridedView.Of(history.Shape).Permute([1, 0, 2])));
        }
        backend.CopyStrided(padded, x, new StridedCopySpec(
            StridedView.Of(padded.Shape).Slice(0, carry, frames),
            StridedView.Of(x.Shape).Reshape([rows, frames, channels]).Permute([1, 0, 2])));
        // Output t uses padded frames t .. t + taps - 1, so a zero-padded convolution with no extra pad gives the causal result.
        Tensor convolved = new(padded.Shape, DType.F32);
        backend.DepthwiseConvFrames(convolved, padded, weight, total, rows, channels, taps, 0);
        Tensor biasFull = new(padded.Shape, DType.F32);
        backend.CopyStrided(biasFull, bias, new StridedCopySpec(StridedView.Of(biasFull.Shape),
            StridedView.Of(bias.Shape).Broadcast(0, rows).Broadcast(0, total)));
        Tensor biased = new(padded.Shape, DType.F32);
        backend.Add(biased, convolved, biasFull);
        Tensor activated = new(padded.Shape, DType.F32);
        backend.Silu(activated, biased);
        // The carry for the next call is the last taps - 1 padded frames, per row.
        if (history is not null)
        {
            backend.CopyStrided(history, padded, new StridedCopySpec(
                StridedView.Of(history.Shape),
                StridedView.Of(padded.Shape).Slice(0, frames, carry).Permute([1, 0, 2])));
        }
        Tensor output = new(x.Shape, DType.F32);
        backend.CopyStrided(output, activated, new StridedCopySpec(
            StridedView.Of(output.Shape).Reshape([rows, frames, channels]),
            StridedView.Of(activated.Shape).Slice(0, 0, frames).Permute([1, 0, 2])));
        padded.Dispose();
        convolved.Dispose();
        biasFull.Dispose();
        biased.Dispose();
        activated.Dispose();
        return output;
    }

    /// <summary>Causal depthwise convolution plus SiLU in place over <c>[rows, frames, channels]</c>. <paramref name="history"/> holds
    /// the <c>taps - 1</c> frames before the first one per row (<c>[rows, taps - 1, channels]</c>) and is rewritten with the last
    /// ones; null means zeros before the start and keeps nothing.</summary>
    public static void CausalDepthwiseSilu(Tensor x, Tensor weight, Tensor bias, Tensor? history, int rows, int frames, int channels, int taps)
    {
        float* p = (float*)x.DataPointer;
        float* w = (float*)weight.DataPointer;
        float* b = (float*)bias.DataPointer;
        float* h = history is null ? null : (float*)history.DataPointer;
        int carry = taps - 1;
        CpuParallel.For(rows, (long)rows * frames * channels * taps, r =>
        {
            float* rowX = p + (long)r * frames * channels;
            float* rowH = h is null ? null : h + (long)r * carry * channels;
            float* padded = (float*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)((long)(carry + frames) * channels * 4));
            try
            {
                for (int i = 0; i < carry * channels; i++)
                {
                    padded[i] = rowH is null ? 0f : rowH[i];
                }
                Buffer.MemoryCopy(rowX, padded + (long)carry * channels, (long)frames * channels * 4, (long)frames * channels * 4);
                for (int t = 0; t < frames; t++)
                {
                    float* dst = rowX + (long)t * channels;
                    for (int c = 0; c < channels; c++)
                    {
                        float acc = b[c];
                        for (int k = 0; k < taps; k++)
                        {
                            acc += w[c * taps + k] * padded[(long)(t + k) * channels + c];
                        }
                        dst[c] = acc / (1f + MathF.Exp(-acc));
                    }
                }
                if (rowH is not null)
                {
                    Buffer.MemoryCopy(padded + (long)frames * channels, rowH, (long)carry * channels * 4, (long)carry * channels * 4);
                }
            }
            finally
            {
                System.Runtime.InteropServices.NativeMemory.Free(padded);
            }
        });
    }

    /// <summary>Reverses the frame order of every row of <c>[rows, frames, width]</c> into <paramref name="output"/>.</summary>
    public static void FlipFrames(Tensor output, Tensor input, int rows, int frames, int width)
    {
        float* src = (float*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        CpuParallel.For(rows * frames, (long)rows * frames * width, token =>
        {
            int r = token / frames;
            int t = token - r * frames;
            Buffer.MemoryCopy(src + ((long)r * frames + t) * width, dst + ((long)r * frames + (frames - 1 - t)) * width, width * 4L, width * 4L);
        });
    }

    /// <summary>Concatenates <c>first + residual</c> and <c>second + residual</c> along the last axis into <paramref name="output"/>
    /// <c>[.., 2 * width]</c>.</summary>
    public static void ConcatPlusResidual(IBackend backend, Tensor output, Tensor first, Tensor second, Tensor residual)
    {
        Tensor a = new(first.Shape, DType.F32);
        Tensor b = new(second.Shape, DType.F32);
        backend.Add(a, first, residual);
        backend.Add(b, second, residual);
        backend.Concat(output, [a, b], output.Shape.Rank - 1);
        a.Dispose();
        b.Dispose();
    }

    /// <summary>Copies rows <c>[firstPosition, firstPosition + count)</c> of a half-width table into a <c>[count, hd]</c> tensor whose
    /// two halves both hold the row, the layout the rotate-half backend rotary reads.</summary>
    private static Tensor WidenedRows(float[] half, int firstPosition, int count, int hd)
    {
        int h = hd / 2;
        Tensor rows = new(new TensorShape(count, hd), DType.F32);
        float* p = (float*)rows.DataPointer;
        for (int s = 0; s < count; s++)
        {
            for (int i = 0; i < h; i++)
            {
                float v = half[(long)(firstPosition + s) * h + i];
                p[(long)s * hd + i] = v;
                p[(long)s * hd + h + i] = v;
            }
        }
        return rows;
    }
}
