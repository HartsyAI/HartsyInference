using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.CrossFrame;
using Xunit;

namespace HartsyInference.Vision.Tests.CrossFrame;

/// <summary>The unbuffered ring against a plain host model of itself: the pooled prefix is the mean of the visible frames' prefixes, the key and
/// value caches hold the pooled rows first and then one slot per frame (oldest overwritten once the ring wraps), and <c>LiveCopy</c> is the live
/// prefix of every head. The ring writes through backend ops only, so this also holds on a backend that keeps the caches on the device.</summary>
public sealed class CrossFrameKvRingTests : IDisposable
{
    private const int Window = 3, Frames = 7, PrefixRows = 2, Width = 5, Patches = 3, Heads = 2, Dim = 4;

    private readonly CpuBackend _backend = new();

    [Fact]
    public void PooledPrefixIsTheMeanOfTheVisibleFramesAndTheCachesFollowTheSlots()
    {
        Random rng = new(5);
        using CrossFrameKvRing ring = new(Window);
        List<float[]> prefixes = [];
        List<float[]> keys = [], values = [];
        for (int t = 0; t < Frames; t++)
        {
            float[] prefix = Random(rng, PrefixRows * Width), key = Random(rng, Patches * Heads * Dim), value = Random(rng, Patches * Heads * Dim);
            prefixes.Add(prefix);
            keys.Add(key);
            values.Add(value);

            using Tensor prefixTensor = Make(prefix, 1, PrefixRows, Width);
            using Tensor pooled = ring.UpdatePrefix(_backend, 0, prefixTensor);
            float[] expectedMean = Mean(prefixes.Skip(Math.Max(0, t + 1 - Window)).ToList());
            Assert.True(MaxAbs(pooled.AsReadOnlySpan<float>(), expectedMean) <= 1e-6f, $"frame {t}: pooled prefix");

            float[] pooledKeyRows = Random(rng, PrefixRows * Heads * Dim), pooledValueRows = Random(rng, PrefixRows * Heads * Dim);
            using Tensor keyTensor = Make(key, 1, Patches, Heads, Dim), valueTensor = Make(value, 1, Patches, Heads, Dim);
            using Tensor pooledKey = Make(pooledKeyRows, 1, PrefixRows, Heads, Dim), pooledValue = Make(pooledValueRows, 1, PrefixRows, Heads, Dim);
            int live = ring.Update(_backend, 0, keyTensor, valueTensor, pooledKey, pooledValue);
            Assert.Equal(PrefixRows + Math.Min(t + 1, Window) * Patches, live);
            Assert.Equal(live, ring.LiveRows(0));

            // Slot s holds frame (t - ((t - s) mod Window)); rows before the first slot are the pooled rows.
            int capacity = PrefixRows + Window * Patches;
            ReadOnlySpan<float> cacheK = ring.Keys(0).AsReadOnlySpan<float>(), cacheV = ring.Values(0).AsReadOnlySpan<float>();
            for (int h = 0; h < Heads; h++)
            {
                for (int r = 0; r < PrefixRows; r++)
                    for (int d = 0; d < Dim; d++)
                    {
                        Assert.Equal(pooledKeyRows[(r * Heads + h) * Dim + d], cacheK[(h * capacity + r) * Dim + d]);
                        Assert.Equal(pooledValueRows[(r * Heads + h) * Dim + d], cacheV[(h * capacity + r) * Dim + d]);
                    }

                for (int s = 0; s < Math.Min(t + 1, Window); s++)
                {
                    int frame = t - ((t - s) % Window + Window) % Window;
                    for (int p = 0; p < Patches; p++)
                        for (int d = 0; d < Dim; d++)
                        {
                            Assert.Equal(keys[frame][(p * Heads + h) * Dim + d], cacheK[(h * capacity + PrefixRows + s * Patches + p) * Dim + d]);
                            Assert.Equal(values[frame][(p * Heads + h) * Dim + d], cacheV[(h * capacity + PrefixRows + s * Patches + p) * Dim + d]);
                        }
                }
            }

            using Tensor liveKeys = CrossFrameKvRing.LiveCopy(_backend, ring.Keys(0), live);
            Assert.Equal(new TensorShape(1, Heads, live, Dim), liveKeys.Shape);
            ReadOnlySpan<float> copied = liveKeys.AsReadOnlySpan<float>();
            for (int h = 0; h < Heads; h++)
                for (int i = 0; i < live * Dim; i++)
                    Assert.Equal(cacheK[h * capacity * Dim + i], copied[h * live * Dim + i]);
        }
    }

    [Fact]
    public void LiveCopyRefusesMoreRowsThanTheCacheHolds()
    {
        using Tensor cache = new(new TensorShape(1, 2, 6, 4), DType.F32);
        Assert.Throws<ArgumentOutOfRangeException>(() => CrossFrameKvRing.LiveCopy(_backend, cache, 7));
        Assert.Throws<ArgumentOutOfRangeException>(() => CrossFrameKvRing.LiveCopy(_backend, cache, 0));
    }

    public void Dispose() => _backend.Dispose();

    private static float[] Random(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }

    private static float[] Mean(List<float[]> rows)
    {
        float[] mean = new float[rows[0].Length];
        foreach (float[] row in rows)
            for (int i = 0; i < mean.Length; i++) mean[i] += row[i];
        for (int i = 0; i < mean.Length; i++) mean[i] /= rows.Count;
        return mean;
    }

    private static Tensor Make(float[] values, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static float MaxAbs(ReadOnlySpan<float> a, float[] b)
    {
        float max = 0;
        for (int i = 0; i < b.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }
}
