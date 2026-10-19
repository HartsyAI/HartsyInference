using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using HartsyInference.Core.Rope;
using Xunit;

namespace HartsyInference.Core.Tests.Rope;

/// <summary>Lifetime, identity, leasing and retirement rules of <see cref="RopeTableCache"/>, driven through a recording backend.</summary>
public sealed class RopeTableCacheTests
{
    private static readonly float[] SignatureA = [0.5f, -1f, 2f, 3f];
    private static readonly float[] SignatureB = [1.5f, 0.25f, -2f, 0.75f];
    private static readonly float[] SignatureC = [4f, 4f, 4f, 4f];

    [Fact]
    public void Acquire_HostTables_MatchComputePairsBitForBit()
    {
        RopeFrequencyPlan plan = Plan();
        using RopeTableCache cache = new(plan, Settings());
        RopeRecordingBackend backend = new();
        float[] expectedCos = new float[8];
        float[] expectedSin = new float[8];
        MultiAxisRopeTables.ComputePairs(plan, SignatureA, 2, RopeAngleOrder.MultiplyByInverse, expectedCos, expectedSin);

        using RopeTableLease lease = cache.Acquire(backend, 0, 2, SignatureA);

        Assert.True(lease.IsValid);
        Assert.Equal(2, lease.Tokens);
        Assert.Equal(expectedCos, lease.HostCos.ToArray());
        Assert.Equal(expectedSin, lease.HostSin.ToArray());
        Assert.Null(lease.DeviceCos);
        Assert.Empty(backend.Calls);
    }

    [Fact]
    public void Acquire_SameSignature_ReusesHostTables()
    {
        using RopeTableCache cache = new(Plan(), Settings());
        RopeRecordingBackend backend = new();

        using RopeTableLease first = cache.Acquire(backend, 0, 2, SignatureA);
        using RopeTableLease second = cache.Acquire(backend, 0, 2, SignatureA);

        Assert.True(SameStorage(first.HostCos, second.HostCos));
    }

    [Fact]
    public void Acquire_SignatureComparesBits_NegativeZeroIsADifferentSignature()
    {
        using RopeTableCache cache = new(Plan(), Settings());
        RopeRecordingBackend backend = new();
        float[] positive = [0f, 1f, 2f, 3f];
        float[] negative = [-0f, 1f, 2f, 3f];

        using RopeTableLease first = cache.Acquire(backend, 0, 2, positive);
        using RopeTableLease second = cache.Acquire(backend, 0, 2, negative);

        Assert.False(SameStorage(first.HostCos, second.HostCos));
    }

    [Fact]
    public void Acquire_Preload_BuildsDeviceRowsAndPreloadsEachTensor()
    {
        RopeFrequencyPlan plan = Plan();
        using RopeTableCache cache = new(plan, Settings(preload: true));
        RopeRecordingBackend backend = new();

        using RopeTableLease lease = cache.Acquire(backend, 0, 2, SignatureA);

        Assert.NotNull(lease.DeviceCos);
        Assert.NotNull(lease.DeviceSin);
        Assert.Equal(new[] { "preload:1", "preload:1" }, backend.Calls);
    }

    [Fact]
    public void Acquire_PartialPreloadFailure_RollsBackTheFirstTensor_AndRecovers()
    {
        using RopeTableCache cache = new(Plan(), Settings(preload: true));
        RopeRecordingBackend backend = new() { FailPreloadCall = 2 };

        Assert.Throws<InvalidOperationException>(() => cache.Acquire(backend, 0, 2, SignatureA));
        Assert.Equal(new[] { "preload:1", "preload:1", "free:1" }, backend.Calls);

        backend.FailPreloadCall = 0;
        using RopeTableLease lease = cache.Acquire(backend, 0, 2, SignatureA);

        Assert.NotNull(lease.DeviceCos);
        Assert.Equal(new[] { "preload:1", "preload:1", "free:1", "preload:1", "preload:1" }, backend.Calls);
    }

    [Fact]
    public void Acquire_FailedBuild_KeepsPublishedTablesValid()
    {
        using RopeTableCache cache = new(Plan(), Settings());
        RopeRecordingBackend backend = new();
        using RopeTableLease published = cache.Acquire(backend, 0, 2, SignatureA);
        float[] nonFinite = [float.NaN, 1f, 2f, 3f];

        Assert.ThrowsAny<ArgumentException>(() => cache.Acquire(backend, 0, 2, nonFinite));
        using RopeTableLease again = cache.Acquire(backend, 0, 2, SignatureA);

        Assert.True(SameStorage(published.HostCos, again.HostCos));
    }

    [Fact]
    public void Retention_EvictsTheLeastRecentlyUsedSignature()
    {
        using RopeTableCache cache = new(Plan(), Settings(retained: 2));
        RopeRecordingBackend backend = new();
        using RopeTableLease a = cache.Acquire(backend, 0, 2, SignatureA);
        using (cache.Acquire(backend, 0, 2, SignatureB)) { }
        using (cache.Acquire(backend, 0, 2, SignatureC)) { }

        using RopeTableLease aAgain = cache.Acquire(backend, 0, 2, SignatureA);

        Assert.False(SameStorage(a.HostCos, aAgain.HostCos));
    }

    [Fact]
    public void Retirement_CallsBeforeRetire_BeforeFreeing()
    {
        using RopeTableCache cache = new(Plan(), Settings(preload: true, retained: 1),
            backend => ((RopeRecordingBackend)backend).Calls.Add("beforeRetire"));
        RopeRecordingBackend backend = new();
        using (RopeTableLease a = cache.Acquire(backend, 0, 2, SignatureA)) { }

        using (RopeTableLease b = cache.Acquire(backend, 0, 2, SignatureB)) { }

        Assert.Equal(new[] { "preload:1", "preload:1", "beforeRetire", "free:2", "preload:1", "preload:1" }, backend.Calls);
    }

    [Fact]
    public void Retirement_IsDeferredWhileALeaseIsOutstanding()
    {
        using RopeTableCache cache = new(Plan(), Settings(preload: true, retained: 1));
        RopeRecordingBackend backend = new();
        RopeTableLease a = cache.Acquire(backend, 0, 2, SignatureA);
        using (cache.Acquire(backend, 0, 2, SignatureB)) { }
        Assert.DoesNotContain("free:2", backend.Calls);

        a.Dispose();

        Assert.Contains("free:2", backend.Calls);
    }

    [Fact]
    public void CopiedLease_ReleasesOnlyOnce()
    {
        using RopeTableCache cache = new(Plan(), Settings(preload: true, retained: 1));
        RopeRecordingBackend backend = new();
        RopeTableLease first = cache.Acquire(backend, 0, 2, SignatureA);
        RopeTableLease copy = first;
        RopeTableLease second = cache.Acquire(backend, 0, 2, SignatureA);

        first.Dispose();
        copy.Dispose();
        using (cache.Acquire(backend, 0, 2, SignatureB)) { }

        Assert.DoesNotContain("free:2", backend.Calls);
        second.Dispose();
        Assert.Contains("free:2", backend.Calls);
    }

    [Fact]
    public void StaleLease_CannotReleaseASlotThatWasReused()
    {
        using RopeTableCache cache = new(Plan(), Settings(preload: true, retained: 1));
        RopeRecordingBackend backend = new();
        RopeTableLease stale = cache.Acquire(backend, 0, 2, SignatureA);
        RopeTableLease stalecopy = stale;
        stale.Dispose();
        RopeTableLease current = cache.Acquire(backend, 0, 2, SignatureB);
        int freesBeforeStaleRelease = backend.Calls.Count(call => call == "free:2");

        stalecopy.Dispose();
        using (cache.Acquire(backend, 0, 2, SignatureC)) { }

        Assert.Equal(freesBeforeStaleRelease, backend.Calls.Count(call => call == "free:2"));
        current.Dispose();
        Assert.Equal(freesBeforeStaleRelease + 1, backend.Calls.Count(call => call == "free:2"));
    }

    [Fact]
    public void FailedFree_StaysTrackedAndRetriesOnTheNextAcquire()
    {
        using RopeTableCache cache = new(Plan(), Settings(preload: true, retained: 1));
        RopeRecordingBackend backend = new();
        using (RopeTableLease a = cache.Acquire(backend, 0, 2, SignatureA)) { }
        backend.FailFree = true;

        Assert.Throws<InvalidOperationException>(() => cache.Acquire(backend, 0, 2, SignatureB));
        Assert.Contains("free-failed:2", backend.Calls);

        backend.FailFree = false;
        using (cache.Acquire(backend, 0, 2, SignatureB)) { }

        Assert.Single(backend.Calls, call => call == "free:2");
    }

    [Fact]
    public void Acquire_PreloadWithZeroTokens_IsRejected()
    {
        using RopeTableCache cache = new(Plan(), Settings(preload: true));
        RopeRecordingBackend backend = new();

        Assert.ThrowsAny<ArgumentException>(() => cache.Acquire(backend, 0, 0, []));
        Assert.Empty(backend.Calls);
    }

    [Fact]
    public void DefaultLease_IsAnEmptyNoOp()
    {
        RopeTableLease lease = default;

        Assert.False(lease.IsValid);
        Assert.Equal(0, lease.Tokens);
        lease.Dispose();
    }

    [Fact]
    public void Dispose_RejectsNewAcquisitions_AndDefersFreeingToTheLastLease()
    {
        RopeTableCache cache = new(Plan(), Settings(preload: true));
        RopeRecordingBackend backend = new();
        RopeTableLease lease = cache.Acquire(backend, 0, 2, SignatureA);

        cache.Dispose();

        Assert.Throws<ObjectDisposedException>(() => cache.Acquire(backend, 0, 2, SignatureB));
        Assert.DoesNotContain("free:2", backend.Calls);
        lease.Dispose();
        Assert.Contains("free:2", backend.Calls);
        cache.Dispose();
    }

    [Fact]
    public void Acquire_RejectsBadArguments()
    {
        using RopeTableCache cache = new(Plan(), Settings());
        RopeRecordingBackend backend = new();

        Assert.ThrowsAny<ArgumentException>(() => cache.Acquire(backend, -1, 2, SignatureA));
        Assert.ThrowsAny<ArgumentException>(() => cache.Acquire(backend, 0, 1, SignatureA));
    }

    private static RopeFrequencyPlan Plan() => RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 4], Theta = 10000 });

    private static RopeTableSettings Settings(bool preload = false, int retained = 2) => new()
    {
        AngleOrder = RopeAngleOrder.MultiplyByInverse,
        Layout = RopeTableLayout.InterleavedPair,
        Preload = preload,
        RetainedSignatures = retained,
    };

    private static bool SameStorage(ReadOnlySpan<float> first, ReadOnlySpan<float> second) =>
        Unsafe.AreSame(
            ref Unsafe.AsRef(in MemoryMarshal.GetReference(first)),
            ref Unsafe.AsRef(in MemoryMarshal.GetReference(second)));
}
