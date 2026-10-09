using HartsyInference.Core.Exceptions;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Engine.Planning.Memory;

namespace HartsyInference.Engine.Placement;

/// <summary>Judges a load's memory demand against free memory and places its components. Pure: no file is opened and no weight is mapped.</summary>
public static class ResidencyPlanner
{
    /// <summary>The host device key: dense, embedding and head weights stay host-resident, routed experts run on the host.</summary>
    public const string HostDevice = "cpu";

    /// <summary>The storage key for components read by row, such as the Engram table.</summary>
    public const string StorageDevice = "storage";

    /// <summary>Judges <paramref name="demand"/> against <paramref name="availableBytes"/> of free memory. The working set and headroom are reserved on a ledger
    /// in account order; any account the ledger refuses makes the plan infeasible and is named. The stored weights are then judged against what remains.</summary>
    public static ResidencyPlan Plan(ResidencyDemand demand, long availableBytes)
    {
        ArgumentNullException.ThrowIfNull(demand);
        ArgumentOutOfRangeException.ThrowIfNegative(availableBytes);
        ResidencyLedger ledger = new(availableBytes, hostPinnedBytes: 0);
        Dictionary<ResidencyAccount, long> refused = [];
        Reserve(ledger, ResidencyAccount.Headroom, demand.HeadroomBytes, refused);
        foreach (KeyValuePair<ResidencyAccount, long> entry in demand.WorkingSet.OrderBy(static entry => entry.Key))
            Reserve(ledger, entry.Key, entry.Value, refused);

        ResidencyLedgerSnapshot snapshot = ledger.Snapshot();
        long workingSet = demand.WorkingSet.Values.Sum();
        long mapped = demand.DenseBytes + demand.ExpertBytes;
        long remaining = snapshot.DeviceBytes - snapshot.DeviceReserved;
        long needed = workingSet + demand.HeadroomBytes;
        MemoryFitVerdict verdict = refused.Count > 0 ? MemoryFitVerdict.Infeasible
            : mapped <= remaining ? MemoryFitVerdict.Resident
            : MemoryFitVerdict.Streamed;
        string reason = verdict switch
        {
            MemoryFitVerdict.Infeasible => $"Needs {ByteFormat.GbF1(needed)} of working memory and headroom against {ByteFormat.GbF1(availableBytes)} free, short by "
                + $"{ByteFormat.GbF1(needed - availableBytes)}. Only the accounts the ledger refused are listed: "
                + string.Join(", ", refused.OrderBy(static entry => entry.Key).Select(static entry => $"{entry.Key} {ByteFormat.GbF1(entry.Value)}")) + ".",
            MemoryFitVerdict.Streamed => $"Fits: {ByteFormat.GbF1(needed)} of working memory, but the {ByteFormat.GbF1(mapped)} of mapped weights exceed the "
                + $"{ByteFormat.GbF1(remaining)} left, so they are read from storage again.",
            _ => $"Fits resident: {ByteFormat.GbF1(needed)} of working memory and {ByteFormat.GbF1(mapped)} of mapped weights within {ByteFormat.GbF1(availableBytes)} free.",
        };
        return new ResidencyPlan
        {
            Verdict = verdict,
            Components =
            [
                new ComponentResidency(ResidencyComponent.DenseWeights, ResidencyMode.HostResident, HostDevice, demand.DenseBytes),
                new ComponentResidency(ResidencyComponent.RoutedExperts, ResidencyMode.CpuExecute, HostDevice, demand.ExpertBytes),
                new ComponentResidency(ResidencyComponent.Engram, ResidencyMode.Streamed, StorageDevice, demand.EngramBytes),
            ],
            AvailableBytes = availableBytes,
            WorkingSetBytes = workingSet,
            HeadroomBytes = demand.HeadroomBytes,
            MappedBytes = mapped,
            ReservedByAccount = snapshot.ReservedByAccount,
            RefusedByAccount = refused,
            Reason = reason,
        };
    }

    /// <summary>Refuses a plan that places a component where this build does not run it: the host takes host-resident and CPU-executed components,
    /// and storage takes streamed ones. GPU placement is a later change.</summary>
    /// <exception cref="HartsyInferenceException">A component is placed on a device this build does not run it on.</exception>
    public static void RequireAdmitted(ResidencyPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        foreach (ComponentResidency component in plan.Components)
        {
            bool admitted = component.Device switch
            {
                HostDevice => component.Mode is ResidencyMode.HostResident or ResidencyMode.CpuExecute,
                StorageDevice => component.Mode is ResidencyMode.Streamed,
                _ => false,
            };
            if (!admitted)
                throw new HartsyInferenceException($"{component.Component} cannot run as {component.Mode} on '{component.Device}': DeepSeek-V4.1 runs on the host in this build.");
        }
    }

    private static void Reserve(ResidencyLedger ledger, ResidencyAccount account, long bytes, Dictionary<ResidencyAccount, long> refused)
    {
        if (bytes == 0)
            return;
        if (!ledger.TryReserve(account, bytes))
            refused[account] = bytes;
    }
}
