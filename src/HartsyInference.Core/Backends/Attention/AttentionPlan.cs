using System.Text;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>A planner's choice, its reason, and the rows ahead of it that declined.</summary>
/// <param name="Preparation">What the chosen row expects to prepare for the call; null when it has not stated it.</param>
/// <param name="Cheaper">A later row that covers and admits the call and states less preparation than the chosen one, with
/// its statement. Reported only: the chosen row is the first that covers and admits, whatever this says.</param>
public readonly record struct AttentionPlan(AttentionKernelRow? Row, string? Variant, string Reason, string? Declined,
    AttentionPreparationCost? Preparation = null, string? Cheaper = null)
{
    /// <summary>Row id, or "none".</summary>
    public string RowId => Row?.Capability.Id ?? "none";

    /// <summary>A stable one-line description: row, variant, numerics consumed, lowering, capture safety, reason.</summary>
    public string Describe()
    {
        if (Row is null) return $"none: {Reason}";
        AttentionKernelCapability c = Row.Capability;
        StringBuilder sb = new(RowId);
        if (Variant is not null) sb.Append(' ').Append(Variant);
        if (c.UsesQueryKey != AttentionFormats.None || c.UsesProbValue != AttentionFormats.None)
            sb.Append(" qk=").Append(c.UsesQueryKey).Append(" pv=").Append(c.UsesProbValue);
        if (c.Saturates) sb.Append(" saturates");
        if (c.Lowering != AttentionLowering.None) sb.Append(" lowering=").Append(c.Lowering);
        sb.Append(c.CaptureSafe ? " capture-safe" : " not-capture-safe");
        sb.Append(" prepares ").Append(Preparation is { } cost ? cost.ToString() : "unstated");
        if (Cheaper is not null) sb.Append(" (less preparation: ").Append(Cheaper).Append(')');
        if (Declined is not null) sb.Append(" after ").Append(Declined);
        return sb.ToString();
    }
}
