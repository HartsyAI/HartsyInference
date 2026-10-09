using HartsyInference.Core.Moe;
using HartsyInference.Cpu.Moe;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Cpu.Tests.Moe;

public sealed class CpuExpertPoolTests
{
    private const int Items = 10_000;

    [Fact]
    public void EveryItemRunsExactlyOnce()
    {
        int[] counts = new int[Items];
        Action<int> body = i => Interlocked.Increment(ref counts[i]);
        using CpuExpertPool pool = new(4);

        for (int round = 0; round < 20; round++)
        {
            pool.Run(Items, body);
        }

        for (int i = 0; i < Items; i++)
        {
            Assert.Equal(20, counts[i]);
        }
    }

    [Fact]
    public void CallerOnlyPoolCompletes()
    {
        int[] counts = new int[257];
        Action<int> body = i => Interlocked.Increment(ref counts[i]);
        using CpuExpertPool pool = new(1);

        Assert.Equal(1, pool.Parallelism);
        pool.Run(counts.Length, body);
        pool.Run(0, body);

        Assert.All(counts, c => Assert.Equal(1, c));
    }

    [Fact]
    public void ThreadCapOfOneLeavesOnlyTheCaller()
    {
        int parallelism = CpuSchedules.WithCpuThreads(1, () =>
        {
            using CpuExpertPool pool = new();
            return pool.Parallelism;
        });

        Assert.Equal(1, parallelism);
    }

    [Fact]
    public void ParallelismIsClampedToTheCoreCeiling()
    {
        using CpuExpertPool pool = new(int.MaxValue);

        Assert.InRange(pool.Parallelism, 1, Environment.ProcessorCount);
    }

    [Fact]
    public void NonPositiveParticipantCountIsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CpuExpertPool(0));
    }

    [Fact]
    public void JobExceptionPropagatesAndPoolStaysUsable()
    {
        int[] counts = new int[100];
        Action<int> failing = i =>
        {
            if (i == 37)
            {
                throw new InvalidOperationException("boom");
            }
        };
        Action<int> counting = i => Interlocked.Increment(ref counts[i]);
        using CpuExpertPool pool = new(4);

        InvalidOperationException thrown = Assert.Throws<InvalidOperationException>(() => pool.Run(100, failing));
        Assert.Equal("boom", thrown.Message);

        pool.Run(counts.Length, counting);
        Assert.All(counts, c => Assert.Equal(1, c));
    }

    [Fact]
    public void JobExceptionOnCallerOnlyPoolPropagates()
    {
        using CpuExpertPool pool = new(1);

        Assert.Throws<ArgumentException>(() => pool.Run(8, i => throw new ArgumentException("bad", "x")));
        pool.Run(1, _ => { });
    }

    [Fact]
    public void RunAfterDisposeThrows()
    {
        CpuExpertPool pool = new(4);
        pool.Dispose();

        Assert.Throws<ObjectDisposedException>(() => pool.Run(1, _ => { }));
    }

    [Fact]
    public void DoubleDisposeIsSafe()
    {
        CpuExpertPool pool = new(4);
        pool.Dispose();

        Exception? second = Record.Exception(pool.Dispose);

        Assert.Null(second);
    }

    [Fact]
    public void ReentrantRunIsRejected()
    {
        using CpuExpertPool pool = new(1);
        Action<int> nested = _ => pool.Run(1, _ => { });

        Assert.Throws<InvalidOperationException>(() => pool.Run(1, nested));
        pool.Run(1, _ => { });
    }

    [Fact]
    public void SplitExpertReferenceMatchesSingleThreadBitForBit()
    {
        const int hidden = 32;
        const int intermediate = 48;
        const int rows = 64;
        Random random = new(7);
        float[] gate = RandomValues(random, intermediate * hidden);
        float[] up = RandomValues(random, intermediate * hidden);
        float[] down = RandomValues(random, hidden * intermediate);
        float[] x = RandomValues(random, rows * hidden);
        F32ExpertWeights weights = new(hidden, intermediate, gate, up, down);
        ExpertProgram program = ExpertProgram.SwigluClamped(4f);

        float[] expected = new float[rows * hidden];
        ExpertProgramReference.Apply(program, weights, x, rows, expected);

        float[] actual = new float[rows * hidden];
        Action<int> row = r => ExpertProgramReference.Apply(program, weights, x.AsSpan(r * hidden, hidden), 1,
            actual.AsSpan(r * hidden, hidden));
        using CpuExpertPool pool = new(4);
        pool.Run(rows, row);

        AssertBitEqual(expected, actual);
    }

    [Fact]
    public void SteadyStateRunAllocatesNothing()
    {
        int[] counts = new int[256];
        Action<int> body = i => Interlocked.Increment(ref counts[i]);
        using CpuExpertPool pool = new(4);

        for (int warm = 0; warm < 200; warm++)
        {
            pool.Run(256, body);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int round = 0; round < 1000; round++)
        {
            pool.Run(256, body);
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(0, allocated);
    }

    private static float[] RandomValues(Random random, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = random.NextSingle() * 2f - 1f;
        }
        return values;
    }

    private static void AssertBitEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"element {i}: expected {expected[i]:R} got {actual[i]:R}");
        }
    }
}
