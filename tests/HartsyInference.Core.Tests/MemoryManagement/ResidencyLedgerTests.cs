using HartsyInference.Core.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>The residency ledger refuses a reservation its pool cannot take and changes nothing when it refuses.</summary>
public sealed class ResidencyLedgerTests
{
    [Fact]
    public void Reserve_ChargesTheDevicePoolUntilItIsFull()
    {
        ResidencyLedger ledger = new(deviceBytes: 100, hostPinnedBytes: 0);

        Assert.True(ledger.TryReserve(ResidencyAccount.KvCache, 60));
        Assert.False(ledger.TryReserve(ResidencyAccount.Activations, 41));
        Assert.True(ledger.TryReserve(ResidencyAccount.Activations, 40));
        Assert.Equal(0L, ledger.Available(ResidencyAccount.Headroom));
    }

    [Fact]
    public void Refusal_LeavesEveryAccountUnchanged()
    {
        ResidencyLedger ledger = new(deviceBytes: 10, hostPinnedBytes: 0);

        Assert.False(ledger.TryReserve(ResidencyAccount.KvCache, 11));
        ResidencyLedgerSnapshot snapshot = ledger.Snapshot();

        Assert.Equal(0L, snapshot.DeviceReserved);
        Assert.Empty(snapshot.ReservedByAccount);
    }

    [Fact]
    public void PinnedStaging_IsChargedToHostPinnedMemoryNotTheDevice()
    {
        ResidencyLedger ledger = new(deviceBytes: 10, hostPinnedBytes: 50);

        Assert.True(ledger.TryReserve(ResidencyAccount.PinnedStaging, 50));
        Assert.True(ledger.TryReserve(ResidencyAccount.KvCache, 10));
        ResidencyLedgerSnapshot snapshot = ledger.Snapshot();

        Assert.Equal(10L, snapshot.DeviceReserved);
        Assert.Equal(50L, snapshot.HostPinnedReserved);
        Assert.Equal(50L, snapshot.ReservedByAccount[ResidencyAccount.PinnedStaging]);
    }

    [Fact]
    public void Release_ReturnsBytes_AndRejectsMoreThanIsHeld()
    {
        ResidencyLedger ledger = new(deviceBytes: 100, hostPinnedBytes: 0);
        Assert.True(ledger.TryReserve(ResidencyAccount.Engram, 30));

        ledger.Release(ResidencyAccount.Engram, 20);

        Assert.Equal(90L, ledger.Available(ResidencyAccount.Engram));
        Assert.Throws<InvalidOperationException>(() => ledger.Release(ResidencyAccount.Engram, 11));
    }

    [Fact]
    public void Constructor_RejectsNegativeBudgets()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ResidencyLedger(deviceBytes: -1, hostPinnedBytes: 0));
    }
}
