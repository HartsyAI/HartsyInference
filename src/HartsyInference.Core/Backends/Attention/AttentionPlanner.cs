using System.Text;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>Chooses among a backend's rows. Rows are tried in preference order; the first whose capability covers the call
/// and whose runtime admission passes is the plan, and a recoverable decline at run time falls to the next.</summary>
public static class AttentionPlanner
{
    /// <summary>The row <see cref="Run"/> would try first, without launching anything.</summary>
    public static AttentionPlan Plan(IReadOnlyList<AttentionKernelRow> rows, in AttentionCall call)
    {
        StringBuilder? declined = null;
        foreach (AttentionKernelRow row in rows)
        {
            AttentionExclusion ex = row.Capability.Excludes(in call);
            string? why = ex != AttentionExclusion.None ? ex.ToString() : row.Admits(in call) ? null : "not admitted";
            if (why is null)
                return new(row, row.Variant(in call), "first row that covers and admits the call", declined?.ToString());
            (declined ??= new()).Append(declined.Length > 0 ? ", " : "").Append(row.Capability.Id).Append(" (").Append(why).Append(')');
        }
        return new(null, null, declined?.ToString() ?? "no rows", null);
    }

    /// <summary>Runs the first row that covers, admits and completes the call; returns it (allocation-free).</summary>
    /// <exception cref="AttentionUnsupportedException">No row served the call.</exception>
    public static AttentionKernelRow Run(IReadOnlyList<AttentionKernelRow> rows, in AttentionCall call)
    {
        for (int i = 0; i < rows.Count; i++)
        {
            AttentionKernelRow row = rows[i];
            if (row.Capability.Excludes(in call) != AttentionExclusion.None || !row.Admits(in call)) continue;
            if (row.Run(in call)) return row;
        }
        throw new AttentionUnsupportedException($"No attention row served this call: {Explain(rows, in call)}");
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
