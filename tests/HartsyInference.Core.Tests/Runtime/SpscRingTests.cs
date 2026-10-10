using HartsyInference.Core.Runtime;
using Xunit;

namespace HartsyInference.Core.Tests.Runtime;

/// <summary>The ring between an ordinary producer and a real-time consumer. A wrong index here is not a crash but
/// a click in the audio, so the tests pin ordering across the wrap point, the drop-newest policy on overflow and,
/// most importantly, that two threads hammering it without a lock still hand every element across exactly once.</summary>
public sealed class SpscRingTests
{
    [Fact]
    public void Constructor_RejectsNonPowerOfTwoCapacity()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SpscRing<short>(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SpscRing<short>(6));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SpscRing<short>(-8));
        Assert.Equal(8, new SpscRing<short>(8).Capacity);
    }

    [Fact]
    public void WrapAroundTheEnd_KeepsOrder()
    {
        SpscRing<int> ring = new(8);
        ring.Write([1, 2, 3, 4, 5, 6]);
        int[] scratch = new int[4];
        ring.Read(scratch);
        Assert.Equal(new[] { 1, 2, 3, 4 }, scratch);

        // Write index is at 6; four more elements straddle the end of the backing array.
        Assert.Equal(4, ring.Write([7, 8, 9, 10]));
        int[] output = new int[6];
        Assert.Equal(6, ring.Read(output));
        Assert.Equal(new[] { 5, 6, 7, 8, 9, 10 }, output);
    }

    [Fact]
    public void FullRing_DropsTheNewestAndCountsIt()
    {
        SpscRing<short> ring = new(4);
        Assert.Equal(4, ring.Write([1, 2, 3, 4, 5, 6]));
        Assert.Equal(2, ring.DroppedSamples);
        Assert.Equal(0, ring.Write([7]));
        Assert.Equal(3, ring.DroppedSamples);

        short[] output = new short[4];
        Assert.Equal(4, ring.Read(output));
        // The four that fit are the OLDEST of the burst; nothing already queued was overwritten.
        Assert.Equal(new short[] { 1, 2, 3, 4 }, output);
    }

    [Fact]
    public void ConcurrentProducerAndConsumer_HandOverEveryElementOnce()
    {
        const int Total = 2_000_000;
        SpscRing<int> ring = new(1024);
        long producerSum = 0, consumerSum = 0;
        long consumed = 0;
        int lastSeen = -1;
        bool ordered = true;

        Thread producer = new(() =>
        {
            int[] batch = new int[97];
            int next = 0;
            while (next < Total)
            {
                // Offer only what fits, so nothing is ever dropped and every element crosses exactly once.
                int n = Math.Min(Math.Min(batch.Length, Total - next), ring.FreeSpace);
                if (n == 0)
                {
                    Thread.SpinWait(20);
                    continue;
                }
                for (int i = 0; i < n; i++)
                {
                    batch[i] = next + i;
                }
                int written = ring.Write(batch.AsSpan(0, n));
                Assert.Equal(n, written);
                for (int i = 0; i < written; i++)
                {
                    producerSum += batch[i];
                }
                next += written;
            }
        });
        Thread consumer = new(() =>
        {
            int[] batch = new int[131];
            while (Volatile.Read(ref consumed) < Total)
            {
                int read = ring.Read(batch);
                if (read == 0)
                {
                    Thread.SpinWait(20);
                    continue;
                }
                for (int i = 0; i < read; i++)
                {
                    if (batch[i] != lastSeen + 1)
                    {
                        ordered = false;
                    }
                    lastSeen = batch[i];
                    consumerSum += batch[i];
                }
                Volatile.Write(ref consumed, consumed + read);
            }
        });
        producer.Start();
        consumer.Start();
        Assert.True(producer.Join(TimeSpan.FromSeconds(60)), "producer did not finish");
        Assert.True(consumer.Join(TimeSpan.FromSeconds(60)), "consumer did not finish");

        Assert.Equal(0, ring.DroppedSamples);
        Assert.True(ordered, "elements arrived out of order or were duplicated");
        Assert.Equal(Total, consumed);
        Assert.Equal((long)Total * (Total - 1) / 2, consumerSum);
        Assert.Equal(producerSum, consumerSum);
    }

    [Fact]
    public void DiscardAll_WhileTheProducerWrites_NeverCorruptsOrder()
    {
        const int Total = 500_000;
        SpscRing<int> ring = new(256);
        int lastSeen = -1;
        bool ordered = true;
        long discarded = 0;
        long consumed = 0;
        bool producerDone = false;

        Thread producer = new(() =>
        {
            int[] batch = new int[64];
            int next = 0;
            while (next < Total)
            {
                int n = Math.Min(Math.Min(batch.Length, Total - next), ring.FreeSpace);
                if (n == 0)
                {
                    Thread.SpinWait(20);
                    continue;
                }
                for (int i = 0; i < n; i++)
                {
                    batch[i] = next + i;
                }
                next += ring.Write(batch.AsSpan(0, n));
            }
            Volatile.Write(ref producerDone, true);
        });
        Thread consumer = new(() =>
        {
            int[] batch = new int[50];
            int iteration = 0;
            while (!Volatile.Read(ref producerDone) || ring.Available > 0)
            {
                if ((++iteration & 7) == 0)
                {
                    discarded += ring.DiscardAll();
                    continue;
                }
                int read = ring.Read(batch);
                for (int i = 0; i < read; i++)
                {
                    // After a discard the sequence jumps forward; it must never go backwards or repeat.
                    if (batch[i] <= lastSeen)
                    {
                        ordered = false;
                    }
                    lastSeen = batch[i];
                }
                consumed += read;
            }
        });
        producer.Start();
        consumer.Start();
        Assert.True(producer.Join(TimeSpan.FromSeconds(60)), "producer did not finish");
        Assert.True(consumer.Join(TimeSpan.FromSeconds(60)), "consumer did not finish");

        Assert.True(ordered, "a read after a discard returned an element older than one already seen");
        Assert.True(discarded > 0, "the discards never raced a write; the test did not exercise its case");
        Assert.Equal(0, ring.DroppedSamples);
        Assert.Equal(Total, consumed + discarded);
    }
}
