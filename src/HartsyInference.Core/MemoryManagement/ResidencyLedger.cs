namespace HartsyInference.Core.MemoryManagement;

/// <summary>Reservations against a device budget and a host-pinned budget, by account. Thread-safe; a refused reservation changes nothing.</summary>
public sealed class ResidencyLedger
{
    private static readonly int AccountCount = Enum.GetValues<ResidencyAccount>().Length;

    private readonly object _gate = new();
    private readonly long[] _reserved = new long[AccountCount];
    private long _deviceReserved;
    private long _hostPinnedReserved;

    /// <param name="deviceBytes">Budget of the device pool, in bytes.</param>
    /// <param name="hostPinnedBytes">Budget of the host-pinned pool, in bytes.</param>
    public ResidencyLedger(long deviceBytes, long hostPinnedBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(hostPinnedBytes);
        DeviceBytes = deviceBytes;
        HostPinnedBytes = hostPinnedBytes;
    }

    public long DeviceBytes { get; }

    public long HostPinnedBytes { get; }

    /// <summary>Bytes the account's pool can still take.</summary>
    public long Available(ResidencyAccount account)
    {
        ValidateAccount(account);
        lock (_gate)
        {
            return PoolFree(account);
        }
    }

    /// <summary>Reserves <paramref name="bytes"/> for <paramref name="account"/> when its pool has that much free; otherwise returns false and changes nothing.</summary>
    public bool TryReserve(ResidencyAccount account, long bytes)
    {
        ValidateAccount(account);
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        lock (_gate)
        {
            if (bytes > PoolFree(account))
                return false;
            Charge(account, bytes);
            return true;
        }
    }

    /// <summary>Returns <paramref name="bytes"/> the account holds.</summary>
    public void Release(ResidencyAccount account, long bytes)
    {
        ValidateAccount(account);
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        lock (_gate)
        {
            if (bytes > _reserved[(int)account])
                throw new InvalidOperationException($"{account} holds {_reserved[(int)account]} bytes, so {bytes} cannot be released.");
            Charge(account, -bytes);
        }
    }

    /// <summary>A consistent reading of both pools and the non-zero reservations.</summary>
    public ResidencyLedgerSnapshot Snapshot()
    {
        lock (_gate)
        {
            Dictionary<ResidencyAccount, long> byAccount = new();
            foreach (ResidencyAccount account in Enum.GetValues<ResidencyAccount>())
            {
                if (_reserved[(int)account] != 0)
                    byAccount[account] = _reserved[(int)account];
            }
            return new ResidencyLedgerSnapshot(DeviceBytes, _deviceReserved, HostPinnedBytes, _hostPinnedReserved, byAccount);
        }
    }

    private long PoolFree(ResidencyAccount account) =>
        account == ResidencyAccount.PinnedStaging ? HostPinnedBytes - _hostPinnedReserved : DeviceBytes - _deviceReserved;

    private void Charge(ResidencyAccount account, long delta)
    {
        _reserved[(int)account] += delta;
        if (account == ResidencyAccount.PinnedStaging)
            _hostPinnedReserved += delta;
        else
            _deviceReserved += delta;
    }

    private static void ValidateAccount(ResidencyAccount account)
    {
        if (!Enum.IsDefined(account))
            throw new ArgumentOutOfRangeException(nameof(account), account, "Not a residency account.");
    }
}
