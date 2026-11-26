using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.CrossFrame;
using Xunit;

namespace HartsyInference.Vision.Tests.CrossFrame;

/// <summary>The unbuffered ring against a plain host model of itself: the pooled prefix is the mean of the visible frames' prefixes, and
/// <c>Read</c> gives the pooled rows first and then one slot per frame (oldest replaced once the ring wraps), head-major. The ring takes what
/// <c>Update</c> is given, whether it stores or refuses it.</summary>
public sealed class CrossFrameKvRingTests : IDisposable
{
    private const int Window = 3, Frames = 7, PrefixRows = 2, Width = 5, Patches = 3, Heads = 2, Dim = 4;

    private readonly CpuBackend _backend = new();

    [Fact]
    public void PooledPrefixIsTheMeanOfTheVisibleFramesAndTheRowsFollowTheSlots()
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

            float[] pooledKey = Random(rng, PrefixRows * Heads * Dim), pooledValue = Random(rng, PrefixRows * Heads * Dim);
            int live = ring.Update(0, Rows(key, Patches), Rows(value, Patches), Rows(pooledKey, PrefixRows), Rows(pooledValue, PrefixRows));
            Assert.Equal(PrefixRows + Math.Min(t + 1, Window) * Patches, live);
            Assert.Equal(live, ring.LiveRows(0));

            // Slot s holds frame (t - ((t - s) mod Window)); rows before the first slot are the pooled rows.
            (Tensor readKeys, Tensor readValues) = ring.Read(0);
            using (readKeys)
            using (readValues)
            {
                Assert.Equal(new TensorShape(1, Heads, live, Dim), readKeys.Shape);
                Assert.Equal(readKeys.Shape, readValues.Shape);
                List<float[]> keyRows = [pooledKey], valueRows = [pooledValue];
                for (int s = 0; s < Math.Min(t + 1, Window); s++)
                {
                    int frame = t - ((t - s) % Window + Window) % Window;
                    keyRows.Add(keys[frame]);
                    valueRows.Add(values[frame]);
                }

                Assert.Equal(HeadMajor(keyRows), readKeys.AsReadOnlySpan<float>().ToArray());
                Assert.Equal(HeadMajor(valueRows), readValues.AsReadOnlySpan<float>().ToArray());
            }
        }

        Assert.Equal(Frames, ring.Seen(0));
    }

    [Fact]
    public void TheRingTakesWhatUpdateIsGivenAndRefusesUseOnceDisposed()
    {
        CrossFrameKvRing ring = new(Window);
        List<Tensor> given = [];
        for (int t = 0; t < Window + 1; t++) given.AddRange(Step(ring, Patches));

        Assert.All(given, t => Assert.False(Alive(t)));
        (Tensor keys, Tensor values) = ring.Read(0);
        using (keys)
        using (values)
        {
            ring.Dispose();
            ring.Dispose();

            // What Read gave stays the caller's.
            Assert.True(Alive(keys) && Alive(values));
        }

        Assert.Throws<ObjectDisposedException>(() => ring.Read(0));
    }

    [Fact]
    public void ARefusedUpdateFreesItsTensorsAndKeepsTheRows()
    {
        using CrossFrameKvRing ring = new(Window);
        Step(ring, Patches);
        (Tensor before, Tensor beforeValues) = ring.Read(0);
        using (before)
        using (beforeValues)
        {
            Tensor[] other = Four(Patches + 1);
            Assert.Throws<ArgumentException>(() => ring.Update(0, other[0], other[1], other[2], other[3]));
            Tensor[] unseen = Four(Patches);
            Assert.Throws<InvalidOperationException>(() => ring.Update(1, unseen[0], unseen[1], unseen[2], unseen[3]));
            Tensor[] heads = Four(Patches);
            Tensor oneHeadMore = Make(new float[PrefixRows * (Heads + 1) * Dim], 1, PrefixRows, Heads + 1, Dim);
            Assert.Throws<ArgumentException>(() => ring.Update(0, heads[0], heads[1], oneHeadMore, heads[3]));
            Tensor[] rows = Four(Patches);
            Tensor oneRowMore = Make(new float[(PrefixRows + 1) * Heads * Dim], 1, PrefixRows + 1, Heads, Dim);
            Assert.Throws<ArgumentException>(() => ring.Update(0, rows[0], rows[1], oneRowMore, rows[3]));
            Tensor[] missing = Four(Patches);
            Assert.Throws<ArgumentNullException>(() => ring.Update(0, missing[0], null!, missing[2], missing[3]));

            Tensor[] refused = [.. other, .. unseen, heads[0], heads[1], heads[3], oneHeadMore, rows[0], rows[1], rows[3], oneRowMore,
                missing[0], missing[2], missing[3]];
            Assert.All(refused, t => Assert.False(Alive(t)));
            Assert.Equal(1, ring.Seen(0));
            (Tensor after, Tensor afterValues) = ring.Read(0);
            using (after)
            using (afterValues)
            {
                Assert.Equal(before.AsReadOnlySpan<float>().ToArray(), after.AsReadOnlySpan<float>().ToArray());
                Assert.Equal(beforeValues.AsReadOnlySpan<float>().ToArray(), afterValues.AsReadOnlySpan<float>().ToArray());
            }
        }
    }

    [Fact]
    public void ABlockRefusesReadsBeforeItsFirstUpdateAndAnotherBackendAfterItsFirstPrefix()
    {
        using CrossFrameKvRing ring = new(Window);
        using Tensor prefix = Make(new float[PrefixRows * Width], 1, PrefixRows, Width);
        ring.UpdatePrefix(_backend, 0, prefix).Dispose();
        using CpuBackend other = new();

        Assert.Throws<InvalidOperationException>(() => ring.Read(0));
        Assert.Throws<ArgumentException>(() => ring.UpdatePrefix(other, 0, prefix));
    }

    public void Dispose() => _backend.Dispose();

    // One frame through block 0; returns the four tensors the ring was given.
    private Tensor[] Step(CrossFrameKvRing ring, int patches)
    {
        using Tensor prefix = Make(new float[PrefixRows * Width], 1, PrefixRows, Width);
        ring.UpdatePrefix(_backend, 0, prefix).Dispose();
        Tensor[] four = Four(patches);
        ring.Update(0, four[0], four[1], four[2], four[3]);
        return four;
    }

    private static Tensor[] Four(int patches)
    {
        Random rng = new(patches);
        return [Rows(Random(rng, patches * Heads * Dim), patches), Rows(Random(rng, patches * Heads * Dim), patches),
            Rows(Random(rng, PrefixRows * Heads * Dim), PrefixRows), Rows(Random(rng, PrefixRows * Heads * Dim), PrefixRows)];
    }

    private static Tensor Rows(float[] values, int rows) => Make(values, 1, rows, Heads, Dim);

    // Token-major segments of [rows, Heads, Dim], one after another, as one head-major [Heads, rows, Dim] array.
    private static float[] HeadMajor(List<float[]> segments)
    {
        float[] tokens = segments.SelectMany(s => s).ToArray();
        int rows = tokens.Length / (Heads * Dim);
        float[] result = new float[tokens.Length];
        for (int r = 0; r < rows; r++)
            for (int h = 0; h < Heads; h++)
                Array.Copy(tokens, (r * Heads + h) * Dim, result, (h * rows + r) * Dim, Dim);
        return result;
    }

    private static bool Alive(Tensor t)
    {
        try
        {
            _ = t.AsReadOnlySpan<byte>().Length;
            return true;
        }
        catch (ObjectDisposedException)
        {
            return false;
        }
    }

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
