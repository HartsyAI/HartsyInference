namespace HartsyInference.Core.MemoryManagement;

/// <summary>A reading of a <see cref="ResidencyLedger"/>: both budgets, what each pool holds, and the non-zero reservations by account.</summary>
public sealed record ResidencyLedgerSnapshot(long DeviceBytes, long DeviceReserved, long HostPinnedBytes, long HostPinnedReserved,
    IReadOnlyDictionary<ResidencyAccount, long> ReservedByAccount);
