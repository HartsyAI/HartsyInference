using HartsyInference.Audio.Layers;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>Layout and elementwise steps of the isolator that no backend op covers, as parallel host loops over F32 tensors.</summary>
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
    public static void SwapLeading(Tensor output, Tensor input, int a, int b, int w)
    {
        float* src = (float*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        CpuParallel.For(a * b, (long)a * b * w, row =>
        {
            int i = row / b;
            int j = row - i * b;
            Buffer.MemoryCopy(src + ((long)i * b + j) * w, dst + ((long)j * a + i) * w, w * 4L, w * 4L);
        });
    }

    /// <summary>Splits fused projections <c>[groups * seq, 3 * heads * hd]</c> (q, k, v concatenated per token) into head-major
    /// <c>[groups, heads, seq, hd]</c> tensors.</summary>
    public static void SplitQkvToHeads(Tensor qkv, Tensor q, Tensor k, Tensor v, int groups, int seq, int heads, int hd)
    {
        float* src = (float*)qkv.DataPointer;
        float* outQ = (float*)q.DataPointer;
        float* outK = (float*)k.DataPointer;
        float* outV = (float*)v.DataPointer;
        int width = heads * hd;
        CpuParallel.For(groups * seq, (long)groups * seq * width * 3, token =>
        {
            int g = token / seq;
            int s = token - g * seq;
            float* row = src + (long)token * 3 * width;
            for (int h = 0; h < heads; h++)
            {
                long dst = (((long)g * heads + h) * seq + s) * hd;
                Buffer.MemoryCopy(row + h * hd, outQ + dst, hd * 4L, hd * 4L);
                Buffer.MemoryCopy(row + width + h * hd, outK + dst, hd * 4L, hd * 4L);
                Buffer.MemoryCopy(row + 2 * width + h * hd, outV + dst, hd * 4L, hd * 4L);
            }
        });
    }

    /// <summary>Gathers head-major <c>[groups, heads, seq, hd]</c> into tokens <c>[groups * seq, heads * hd]</c>.</summary>
    public static void HeadsToTokens(Tensor output, Tensor input, int groups, int seq, int heads, int hd)
    {
        float* src = (float*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        CpuParallel.For(groups * seq, (long)groups * seq * heads * hd, token =>
        {
            int g = token / seq;
            int s = token - g * seq;
            for (int h = 0; h < heads; h++)
            {
                Buffer.MemoryCopy(src + (((long)g * heads + h) * seq + s) * hd, dst + (long)token * heads * hd + h * hd, hd * 4L, hd * 4L);
            }
        });
    }

    /// <summary>Rotates head-major <c>[groups, heads, seq, hd]</c> in place by the rotate-half RoPE of position <c>firstPosition + s</c>.</summary>
    public static void Rope(Tensor x, int groups, int heads, int seq, int hd, int firstPosition, float[] cos, float[] sin)
    {
        int half = hd / 2;
        float* p = (float*)x.DataPointer;
        fixed (float* cosP = cos)
        fixed (float* sinP = sin)
        {
            float* c = cosP;
            float* sn = sinP;
            CpuParallel.For(groups * heads, (long)groups * heads * seq * hd, gh =>
            {
                float* head = p + (long)gh * seq * hd;
                for (int s = 0; s < seq; s++)
                {
                    float* row = head + (long)s * hd;
                    float* cr = c + (long)(firstPosition + s) * half;
                    float* sr = sn + (long)(firstPosition + s) * half;
                    for (int i = 0; i < half; i++)
                    {
                        float x1 = row[i];
                        float x2 = row[half + i];
                        row[i] = x1 * cr[i] - x2 * sr[i];
                        row[half + i] = x1 * sr[i] + x2 * cr[i];
                    }
                }
            });
        }
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

    /// <summary>Erf-form GELU in place.</summary>
    public static void Gelu(Tensor x) => Activations.ErfGelu(x);

    /// <summary>Adds <paramref name="bias"/> (length <paramref name="width"/>) to every row, applies softplus, in place.</summary>
    public static void BiasSoftplus(Tensor x, Tensor bias, long rows, int width)
    {
        float* p = (float*)x.DataPointer;
        float* b = (float*)bias.DataPointer;
        CpuParallel.For((int)rows, rows * width * 8, r =>
        {
            float* row = p + (long)r * width;
            for (int i = 0; i < width; i++)
            {
                row[i] = SoftplusReference.Scalar(row[i] + b[i]);
            }
        });
    }

    /// <summary>Splits <c>[rows, 2 * inner]</c> into the two halves <c>[rows, inner]</c>.</summary>
    public static void SplitHalves(Tensor input, Tensor first, Tensor second, long rows, int inner)
    {
        float* src = (float*)input.DataPointer;
        float* a = (float*)first.DataPointer;
        float* b = (float*)second.DataPointer;
        CpuParallel.For((int)rows, rows * inner * 2, r =>
        {
            Buffer.MemoryCopy(src + (long)r * 2 * inner, a + (long)r * inner, inner * 4L, inner * 4L);
            Buffer.MemoryCopy(src + (long)r * 2 * inner + inner, b + (long)r * inner, inner * 4L, inner * 4L);
        });
    }

    /// <summary>Splits <c>[rows, dtRank + 2 * states]</c> into the step input <c>[rows, dtRank]</c>, B and C <c>[rows, states]</c>.</summary>
    public static void SplitDtBc(Tensor input, Tensor dt, Tensor b, Tensor c, long rows, int dtRank, int states)
    {
        float* src = (float*)input.DataPointer;
        float* d = (float*)dt.DataPointer;
        float* bp = (float*)b.DataPointer;
        float* cp = (float*)c.DataPointer;
        int stride = dtRank + 2 * states;
        CpuParallel.For((int)rows, rows * stride, r =>
        {
            float* row = src + (long)r * stride;
            Buffer.MemoryCopy(row, d + (long)r * dtRank, dtRank * 4L, dtRank * 4L);
            Buffer.MemoryCopy(row + dtRank, bp + (long)r * states, states * 4L, states * 4L);
            Buffer.MemoryCopy(row + dtRank + states, cp + (long)r * states, states * 4L, states * 4L);
        });
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

    /// <summary>Concatenates two <c>[rows, width]</c> tensors, each plus <paramref name="residual"/>, into <c>[rows, 2 * width]</c>.</summary>
    public static void ConcatPlusResidual(Tensor output, Tensor first, Tensor second, Tensor residual, long rows, int width)
    {
        float* a = (float*)first.DataPointer;
        float* b = (float*)second.DataPointer;
        float* res = (float*)residual.DataPointer;
        float* dst = (float*)output.DataPointer;
        CpuParallel.For((int)rows, rows * width * 2, r =>
        {
            float* d = dst + (long)r * 2 * width;
            long at = (long)r * width;
            for (int i = 0; i < width; i++)
            {
                d[i] = a[at + i] + res[at + i];
                d[width + i] = b[at + i] + res[at + i];
            }
        });
    }
}
