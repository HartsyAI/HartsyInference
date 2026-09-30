using HartsyInference.BenchmarkRunner.Contracts;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>Device configuration and exclusivity, attested from the controller. The controller holds no CUDA
/// context, so every compute process it sees belongs to somebody else — no self-PID exclusion is needed, which
/// matters because a container reports host-namespace PIDs and a self-filter there would hide the real tenant.</summary>
public static class DeviceAttestation
{
    private static readonly string[] Queried =
    [
        "driver_version", "vbios_version", "compute_cap", "memory.total", "power.limit", "power.max_limit",
        "power.default_limit", "persistence_mode", "ecc.mode.current", "clocks.max.sm", "clocks.max.mem", "compute_mode"
    ];

    /// <summary>The configuration fields that separate cohorts. Driver and capacity are keyed separately.</summary>
    private static readonly string[] Profiled =
    [
        "power.limit", "power.max_limit", "power.default_limit", "persistence_mode", "ecc.mode.current",
        "clocks.max.sm", "clocks.max.mem", "compute_mode"
    ];

    private static readonly TimeSpan Deadline = TimeSpan.FromSeconds(30);

    /// <summary>Resolves the nvidia-smi UUID whose hashed form is this device's recorded identity. Matching on
    /// the hash keeps the raw UUID out of <see cref="DeviceRecord"/>; matching on an ordinal would be wrong,
    /// because CUDA enumerates fastest-first and nvidia-smi enumerates by PCI order.</summary>
    public static string? ResolveUuid(DeviceRecord device)
    {
        if (!device.Selector.StartsWith("cuda:", StringComparison.Ordinal))
            return null;
        foreach (string line in NvidiaSmi.Query(["--query-gpu=uuid", "--format=csv,noheader"], Deadline) ?? [])
            if (Hardware.IdentityHash(NvidiaSmi.NormalizeUuid(line)) == device.Identity)
                return line.Trim();
        return null;
    }

    public static Result Capture(DeviceRecord device, bool allowShared)
    {
        string? uuid = ResolveUuid(device);
        if (uuid is null)
            return new Result(new AttestationRecord { Source = AttestationRecord.Unavailable, SharedDeviceAllowed = allowShared },
                null, []);
        SortedDictionary<string, string> fields = new(StringComparer.Ordinal);
        string[]? values = NvidiaSmi.Query(["-i", uuid, "--query-gpu=" + string.Join(',', Queried), "--format=csv,noheader"], Deadline);
        string[] row = values is { Length: > 0 } ? NvidiaSmi.Fields(values[0]) : [];
        if (row.Length != Queried.Length)
            return new Result(new AttestationRecord { Source = AttestationRecord.Unavailable, SharedDeviceAllowed = allowShared },
                uuid, []);
        for (int i = 0; i < Queried.Length; i++)
            fields[Queried[i]] = row[i];
        (int count, long bytes, string[] tenants) = Tenants(uuid);
        return new Result(
            new AttestationRecord
            {
                Source = AttestationRecord.Smi,
                Fields = fields,
                SharedProcessCount = count,
                SharedProcessBytes = bytes,
                SharedDeviceAllowed = allowShared
            }, uuid, tenants);
    }

    /// <summary>Compute processes on this device. The returned lines carry PIDs and paths for the console only;
    /// the exported record keeps the count and the total.</summary>
    public static (int Count, long Bytes, string[] Lines) Tenants(string uuid)
    {
        List<string> lines = [];
        long bytes = 0;
        foreach (string line in NvidiaSmi.Query(
            ["--query-compute-apps=pid,process_name,used_memory,gpu_uuid", "--format=csv,noheader,nounits"], Deadline) ?? [])
        {
            // Anchored at both ends: a process path may contain a comma, and a positional parse would then
            // shift the uuid out of its column and silently drop that tenant.
            string[] row = NvidiaSmi.Fields(line);
            if (row.Length < 4 || NvidiaSmi.NormalizeUuid(row[^1]) != NvidiaSmi.NormalizeUuid(uuid))
                continue;
            bytes += long.TryParse(row[^2], out long mib) ? mib * 1024 * 1024 : 0;
            lines.Add($"pid {row[0]} ({string.Join(',', row[1..^2])}) holding {row[^2]} MiB");
        }

        return (lines.Count, bytes, lines.ToArray());
    }

    /// <summary>The cohort-key summary. <c>unattested</c> when nvidia-smi could not describe the device, so an
    /// unattested campaign is never pooled with an attested one.</summary>
    public static string Profile(AttestationRecord record) => record.Source == AttestationRecord.Smi
        ? string.Join(';', Profiled.Select(field => field + "=" + (record.Fields.GetValueOrDefault(field) ?? "?")))
        : "unattested";

    /// <summary>Attestation plus the detail that stays on this machine: the raw UUID the sampler binds to, and
    /// the tenant lines the operator is shown.</summary>
    public sealed record Result(AttestationRecord Record, string? Uuid, string[] Tenants);
}
