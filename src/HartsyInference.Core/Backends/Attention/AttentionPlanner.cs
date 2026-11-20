using System.Collections.Concurrent;
using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>Chooses among a backend's rows. Rows are tried in preference order; the first whose capability covers the call
/// and whose runtime admission passes is the plan, and a recoverable decline at run time falls to the next.</summary>
public static class AttentionPlanner
{
    private static readonly ConcurrentDictionary<string, byte> ReportedFailures = new(StringComparer.Ordinal);

    /// <summary>The row <see cref="Run"/> would try first, without launching anything.</summary>
    public static AttentionPlan Plan(IReadOnlyList<AttentionKernelRow> rows, in AttentionCall call)
    {
        StringBuilder? declined = null;
        foreach (AttentionKernelRow row in rows)
        {
            AttentionExclusion ex = row.Capability.Excludes(in call);
            string? why = ex != AttentionExclusion.None ? ex.ToString() : row.Admits(in call) ? null : "not admitted";
            if (why is null)
            {
                AttentionPreparationCost? cost = row.Preparation(in call);
                return new(row, row.Variant(in call), "first row that covers and admits the call", declined?.ToString(), cost,
                    cost is { } chosen ? Cheaper(rows, row, chosen.Total, in call) : null);
            }
            (declined ??= new()).Append(declined.Length > 0 ? ", " : "").Append(row.Capability.Id).Append(" (").Append(why).Append(')');
        }
        return new(null, null, declined?.ToString() ?? "no rows", null);
    }

    /// <summary>The row after <paramref name="chosen"/> that covers and admits the call and states the least preparation
    /// below <paramref name="chosenBytes"/>, as "id bytes"; null when none states less. A report, not a choice.</summary>
    private static string? Cheaper(IReadOnlyList<AttentionKernelRow> rows, AttentionKernelRow chosen, long chosenBytes, in AttentionCall call)
    {
        AttentionKernelRow? best = null;
        AttentionPreparationCost bestCost = default;
        long bestBytes = chosenBytes;
        bool after = false;
        foreach (AttentionKernelRow row in rows)
        {
            if (ReferenceEquals(row, chosen)) { after = true; continue; }
            if (!after || row.Capability.Excludes(in call) != AttentionExclusion.None || !row.Admits(in call)) continue;
            if (row.Preparation(in call) is not { } cost || cost.Total >= bestBytes) continue;
            best = row;
            bestCost = cost;
            bestBytes = cost.Total;
        }
        return best is null ? null : $"{best.Capability.Id} {bestCost}";
    }

    /// <summary>Runs the first row that covers, admits and completes the call; returns it (allocation-free).</summary>
    /// <exception cref="AttentionUnsupportedException">No row covers and admits the call.</exception>
    /// <exception cref="AttentionRecoverableException">Every admitted row declined without changing visible state.</exception>
    public static AttentionKernelRow Run(IReadOnlyList<AttentionKernelRow> rows, in AttentionCall call)
    {
        bool declined = false;
        for (int i = 0; i < rows.Count; i++)
        {
            AttentionKernelRow row = rows[i];
            if (row.Capability.Excludes(in call) != AttentionExclusion.None || !row.Admits(in call)) continue;
            try
            {
                if (row.Run(in call)) return row;
            }
            catch (Exception error) when (NameFailedRow(row, error))
            {
                throw;
            }
            declined = true;
        }
        string explanation = Explain(rows, in call);
        if (declined)
            throw new AttentionRecoverableException($"Every admitted attention row declined without serving this call: {explanation}");
        throw new AttentionUnsupportedException($"No attention row covers and admits this call: {explanation}");
    }

    /// <summary>Logs, once per row and exception type, which row a sticky failure came from; the exception propagates
    /// unchanged (always false, so the filter never catches). Out-of-memory is the caller's to retry and is not reported.</summary>
    private static bool NameFailedRow(AttentionKernelRow row, Exception error)
    {
        if (error is OutOfMemoryException or OutOfVramException) return false;
        string id = row.Capability.Id;
        if (ReportedFailures.TryAdd(id + " " + error.GetType().Name, 0))
            Logs.Warning($"Attention row {id} failed and nothing retries it ({error.GetType().Name}: {error.Message}).");
        return false;
    }

    /// <summary>Every row's verdict after a failed <see cref="Run"/>: what excluded it, or that it was not admitted or declined.</summary>
    private static string Explain(IReadOnlyList<AttentionKernelRow> rows, in AttentionCall call)
    {
        if (rows.Count == 0) return "no rows";
        StringBuilder sb = new();
        foreach (AttentionKernelRow row in rows)
        {
            AttentionExclusion ex = row.Capability.Excludes(in call);
            string why = ex != AttentionExclusion.None ? ex.ToString() : row.Admits(in call) ? "declined" : "not admitted";
            sb.Append(sb.Length > 0 ? ", " : "").Append(row.Capability.Id).Append(" (").Append(why).Append(')');
        }
        return sb.ToString();
    }
}
