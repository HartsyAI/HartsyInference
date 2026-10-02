using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Pure-logic coverage for <see cref="RetainedSequenceStore"/> and <see cref="RetainedSequence"/>: bounds
/// (count and bytes, LRU eviction), the checkout/checkin busy-key contract, and dispose-on-evict/dispose-on-unload.
/// Uses a trivial <see cref="FakeSequenceState"/> instead of a real <see cref="FixedKvCache"/> so these run as fast
/// CPU unit tests with no backend/model involved — the store's bookkeeping does not care what it is storing.</summary>
public sealed class RetainedSequenceStoreTests
{
    private sealed class FakeSequenceState : ISequenceState
    {
        public int Length { get; private set; }
        public int Capacity { get; }
        public int MaxRollback => Length;
        public bool Disposed { get; private set; }

        public FakeSequenceState(int length, int capacity) { Length = length; Capacity = capacity; }

        public void Truncate(int newLength)
        {
            if (newLength < 0 || newLength > Length) throw new ArgumentOutOfRangeException(nameof(newLength));
            Length = newLength;
        }

        public void Reset() => Length = 0;
        public void Dispose() => Disposed = true;
    }

    /// <summary>Builds a <see cref="RetainedSequence"/> and returns it alongside the underlying
    /// <see cref="FakeSequenceState"/> directly — <c>seq.Cache</c> itself reads null once the sequence is disposed
    /// (by the store, on eviction or its own disposal), so a test that checks whether disposal actually happened
    /// must hold its own reference to the fake from before that point, not read it back through <c>seq</c>.</summary>
    private static (RetainedSequence Seq, FakeSequenceState Fake) Seq(int length, int capacity, long bytes)
    {
        FakeSequenceState fake = new(length, capacity);
        RetainedSequence seq = new();
        seq.Update(fake, [.. Enumerable.Range(0, length)], bytes);
        return (seq, fake);
    }

    [Fact]
    public void CheckIn_ThenCheckout_ReturnsTheSameEntry()
    {
        using RetainedSequenceStore store = new(maxEntries: 4, maxBytes: 1024);
        (RetainedSequence seq, _) = Seq(length: 3, capacity: 8, bytes: 100);

        store.CheckIn("a", seq);
        RetainedSequence? got = store.Checkout("a");

        Assert.Same(seq, got);
        Assert.Equal(0, store.Count);
    }

    [Fact]
    public void Checkout_OnABusyOrMissingKey_ReturnsNull()
    {
        using RetainedSequenceStore store = new(maxEntries: 4, maxBytes: 1024);
        (RetainedSequence seq, _) = Seq(3, 8, 100);
        store.CheckIn("a", seq);
        RetainedSequence? first = store.Checkout("a");   // now "checked out" -- removed from the store

        Assert.NotNull(first);
        Assert.Null(store.Checkout("a"));   // a second caller on the same (now busy) key sees a miss
        Assert.Null(store.Checkout("never-stored"));
    }

    [Fact]
    public void CheckIn_EvictsLeastRecentlyUsed_WhenEntryCountExceedsMax()
    {
        using RetainedSequenceStore store = new(maxEntries: 2, maxBytes: 1_000_000);
        (RetainedSequence a, FakeSequenceState fakeA) = Seq(1, 8, 10);
        (RetainedSequence b, _) = Seq(1, 8, 10);
        (RetainedSequence c, _) = Seq(1, 8, 10);

        store.CheckIn("a", a);
        store.CheckIn("b", b);
        store.CheckIn("c", c);   // over the 2-entry cap: "a" (least recently used) is evicted

        Assert.Equal(2, store.Count);
        Assert.True(fakeA.Disposed);
        Assert.Null(store.Checkout("a"));
        Assert.NotNull(store.Checkout("b"));
        Assert.NotNull(store.Checkout("c"));
    }

    [Fact]
    public void Checkout_RefreshesRecency_SoItIsNotTheNextEviction()
    {
        using RetainedSequenceStore store = new(maxEntries: 2, maxBytes: 1_000_000);
        store.CheckIn("a", Seq(1, 8, 10).Seq);
        store.CheckIn("b", Seq(1, 8, 10).Seq);

        // Touch "a": check it out and back in, making "b" the least recently used.
        RetainedSequence a = store.Checkout("a")!;
        store.CheckIn("a", a);

        (RetainedSequence c, _) = Seq(1, 8, 10);
        store.CheckIn("c", c);   // should evict "b", not "a"

        Assert.NotNull(store.Checkout("a"));
        Assert.Null(store.Checkout("b"));
    }

    [Fact]
    public void CheckIn_EvictsLeastRecentlyUsed_WhenBytesExceedBudget()
    {
        using RetainedSequenceStore store = new(maxEntries: 10, maxBytes: 150);
        store.CheckIn("a", Seq(1, 8, 100).Seq);
        store.CheckIn("b", Seq(1, 8, 100).Seq);   // 200 > 150: "a" is evicted to make room

        Assert.Equal(100, store.BytesUsed);
        Assert.Null(store.Checkout("a"));
        Assert.NotNull(store.Checkout("b"));
    }

    [Fact]
    public void CheckIn_ASingleEntryHeavierThanTheBudget_IsStillKept()
    {
        using RetainedSequenceStore store = new(maxEntries: 10, maxBytes: 100);
        (RetainedSequence huge, _) = Seq(1, 8, 10_000);

        store.CheckIn("huge", huge);   // nothing else to evict; kept anyway rather than refused

        Assert.Equal(1, store.Count);
        Assert.NotNull(store.Checkout("huge"));
    }

    [Fact]
    public void Dispose_DisposesEveryRetainedEntry()
    {
        RetainedSequenceStore store = new(maxEntries: 4, maxBytes: 1024);
        (RetainedSequence a, FakeSequenceState fakeA) = Seq(1, 8, 10);
        (RetainedSequence b, FakeSequenceState fakeB) = Seq(1, 8, 10);
        store.CheckIn("a", a);
        store.CheckIn("b", b);

        store.Dispose();

        Assert.True(fakeA.Disposed);
        Assert.True(fakeB.Disposed);
        Assert.Equal(0, store.Count);
    }

    [Fact]
    public void CheckIn_AfterDispose_DisposesTheArgumentInsteadOfStoringIt()
    {
        RetainedSequenceStore store = new(maxEntries: 4, maxBytes: 1024);
        store.Dispose();
        (RetainedSequence seq, FakeSequenceState fake) = Seq(1, 8, 10);

        store.CheckIn("a", seq);

        Assert.True(fake.Disposed);
    }

    [Fact]
    public void RetainedSequence_Update_DisposesThePreviousCache_UnlessItIsTheSameInstance()
    {
        RetainedSequence seq = new();
        FakeSequenceState first = new(0, 8);
        seq.Update(first, [], 10);

        FakeSequenceState second = new(1, 8);
        seq.Update(second, [1], 20);   // a DIFFERENT cache replaces it: the old one must be disposed

        Assert.True(first.Disposed);
        Assert.False(second.Disposed);

        seq.Update(second, [1, 2], 20);   // the SAME cache, just more tokens committed: no self-dispose

        Assert.False(second.Disposed);
        seq.Dispose();
        Assert.True(second.Disposed);
    }
}
