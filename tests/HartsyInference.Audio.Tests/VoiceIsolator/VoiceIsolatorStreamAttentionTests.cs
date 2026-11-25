using HartsyInference.Audio.Models.VoiceIsolator;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests.VoiceIsolator;

/// <summary>The cached time attention against a dense reference that applies the window rule directly: each query sees the keys
/// <c>j</c> with <c>p - S &lt; j &lt;= p</c>, where <c>S = min(window, cache)</c> (or <c>cache</c> when the window is 0), its own key
/// included.</summary>
public sealed class VoiceIsolatorStreamAttentionTests
{
    private const int Bands = 2;
    private const int Heads = 2;
    private const int HeadDim = 8;
    private const int Frames = 40;
    private const int Block = 4;

    [Theory]
    [InlineData(6, 0)]    // the cache alone bounds the window
    [InlineData(6, 6)]    // cache equals the checkpoint window (the default)
    [InlineData(12, 6)]   // cache longer than the window
    [InlineData(8, 20)]   // cache shorter than the window: the cache bounds each query
    public void CachedAttentionMatchesTheDenseWindowRule(int cache, int window)
    {
        using CpuBackend cpu = new();
        IBackend backend = cpu;
        int capacity = cache + Math.Max(cache / 2, Block);
        int slidingWindow = window > 0 ? Math.Min(window, cache) : cache;
        Random rng = new(5);
        float[] q = Values(rng, Bands * Heads * Frames * HeadDim);
        float[] k = Values(rng, Bands * Heads * Frames * HeadDim);
        float[] v = Values(rng, Bands * Heads * Frames * HeadDim);

        float[] streamed = new float[q.Length];
        VoiceIsolatorStreamAttention attention = new(cache, capacity, window, HeadDim);
        using VoiceIsolatorAttentionLane lane = new();
        for (int start = 0; start < Frames; start += Block)
        {
            Tensor qb = Block4(backend, q, start);
            Tensor kb = Block4(backend, k, start);
            Tensor vb = Block4(backend, v, start);
            Tensor output = attention.Attend(backend, lane, qb, kb, vb, Bands, Heads, Block, HeadDim);
            float[] values = Read(output);
            for (int b = 0; b < Bands * Heads; b++)
            {
                Array.Copy(values, b * Block * HeadDim, streamed, (b * Frames + start) * HeadDim, Block * HeadDim);
            }
            qb.Dispose();
            kb.Dispose();
            vb.Dispose();
            output.Dispose();
        }

        float[] expected = Dense(backend, q, k, v, slidingWindow);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(expected[i] - streamed[i]));
        }
        Assert.True(worst < 1e-4f, $"cache {cache}, window {window}: max abs difference {worst}");
    }

    /// <summary>Rotates q and k at their absolute positions, then attends each query over the keys its window allows.</summary>
    private static float[] Dense(IBackend backend, float[] q, float[] k, float[] v, int slidingWindow)
    {
        (float[] cos, float[] sin) = VoiceIsolatorOps.RopeTables(Frames, HeadDim);
        float[] qr = RotateHalf(q, cos, sin);
        float[] kr = RotateHalf(k, cos, sin);

        float scale = 1f / MathF.Sqrt(HeadDim);
        float[] output = new float[q.Length];
        for (int bh = 0; bh < Bands * Heads; bh++)
        {
            for (int p = 0; p < Frames; p++)
            {
                int first = Math.Max(0, p - slidingWindow + 1);
                float max = float.NegativeInfinity;
                float[] score = new float[p - first + 1];
                for (int j = first; j <= p; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < HeadDim; d++)
                    {
                        dot += qr[(bh * Frames + p) * HeadDim + d] * kr[(bh * Frames + j) * HeadDim + d];
                    }
                    score[j - first] = dot * scale;
                    max = Math.Max(max, score[j - first]);
                }
                float total = 0f;
                for (int i = 0; i < score.Length; i++)
                {
                    score[i] = MathF.Exp(score[i] - max);
                    total += score[i];
                }
                for (int j = first; j <= p; j++)
                {
                    float weight = score[j - first] / total;
                    for (int d = 0; d < HeadDim; d++)
                    {
                        output[(bh * Frames + p) * HeadDim + d] += weight * v[(bh * Frames + j) * HeadDim + d];
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Rotate-half RoPE of every row <c>[bands * heads, frames, hd]</c> at its absolute position.</summary>
    private static float[] RotateHalf(float[] x, float[] cos, float[] sin)
    {
        int half = HeadDim / 2;
        float[] y = new float[x.Length];
        for (int row = 0; row < Bands * Heads * Frames; row++)
        {
            int p = row % Frames;
            for (int i = 0; i < half; i++)
            {
                float x1 = x[row * HeadDim + i];
                float x2 = x[row * HeadDim + half + i];
                float c = cos[p * half + i];
                float sn = sin[p * half + i];
                y[row * HeadDim + i] = x1 * c - x2 * sn;
                y[row * HeadDim + half + i] = x1 * sn + x2 * c;
            }
        }
        return y;
    }

    private static float[] Values(Random rng, int n)
    {
        float[] values = new float[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = (float)rng.NextDouble() * 2f - 1f;
        }
        return values;
    }

    /// <summary>The block of <see cref="Block"/> frames starting at <paramref name="start"/>, head-major <c>[bands, heads, block, hd]</c>.</summary>
    private static Tensor Block4(IBackend backend, float[] full, int start)
    {
        float[] block = new float[Bands * Heads * Block * HeadDim];
        for (int bh = 0; bh < Bands * Heads; bh++)
        {
            Array.Copy(full, (bh * Frames + start) * HeadDim, block, bh * Block * HeadDim, Block * HeadDim);
        }
        return Make(new long[] { Bands, Heads, Block, HeadDim }, block);
    }

    private static unsafe Tensor Make(long[] shape, float[] values)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        fixed (float* p = values)
        {
            Buffer.MemoryCopy(p, (void*)t.DataPointer, values.Length * 4L, values.Length * 4L);
        }
        return t;
    }

    private static unsafe float[] Read(Tensor t)
    {
        float[] values = new float[(int)t.ElementCount];
        fixed (float* p = values)
        {
            Buffer.MemoryCopy((void*)t.DataPointer, p, values.Length * 4L, values.Length * 4L);
        }
        return values;
    }
}
