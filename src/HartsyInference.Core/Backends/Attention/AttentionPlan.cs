using System.Text;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>A planner's choice, its reason, and the rows ahead of it that declined.</summary>
public readonly record struct AttentionPlan(AttentionKernelRow? Row, string? Variant, string Reason, string? Declined)
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
        if (Declined is not null) sb.Append(" after ").Append(Declined);
        return sb.ToString();
    }
}
