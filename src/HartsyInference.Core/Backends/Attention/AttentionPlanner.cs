using System.Collections.Concurrent;
using System.Numerics;
using System.Text;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Operand dtypes a kernel row serves.</summary>
[Flags]
public enum AttentionDTypes
{
    /// <summary>None.</summary>
    None = 0,

    /// <summary>F32.</summary>
    F32 = 1,

    /// <summary>F16.</summary>
    F16 = 2,

    /// <summary>BF16.</summary>
    BF16 = 4,

    /// <summary>FP8 (E4M3 or E5M2) storage the row widens itself.</summary>
    Fp8 = 8,

    /// <summary>Any other storage (block-quantized, integer) the row widens itself.</summary>
    Other = 16,

    /// <summary>Every storage dtype.</summary>
    Any = F32 | F16 | BF16 | Fp8 | Other,
}

/// <summary>Operand layouts a kernel row serves.</summary>
[Flags]
public enum AttentionLayouts
{
    /// <summary>None.</summary>
    None = 0,

    /// <summary><see cref="AttentionLayout.HeadMajor"/>.</summary>
    HeadMajor = 1,

    /// <summary><see cref="AttentionLayout.TokenMajor"/>.</summary>
    TokenMajor = 2,

    /// <summary><see cref="AttentionLayout.PackedRows"/>.</summary>
    PackedRows = 4,
}

/// <summary>Mask components a kernel row applies itself.</summary>
[Flags]
public enum AttentionMaskSupport
{
    /// <summary>No mask.</summary>
    None = 0,

    /// <summary>An additive bias shared by every head (<c>[1|B, 1, 1|Sq, Skv]</c>).</summary>
    HeadSharedBias = 1,

    /// <summary>An additive bias with a head axis.</summary>
    PerHeadBias = 2,

    /// <summary>Causal.</summary>
    Causal = 4,

    /// <summary>A sliding window.</summary>
    Window = 8,

    /// <summary>Block-causal.</summary>
    BlockCausal = 16,
}

/// <summary>Work a row adds around its kernel; each costs memory and time the planner counts.</summary>
[Flags]
public enum AttentionLowering
{
    /// <summary>None.</summary>
    None = 0,

    /// <summary>K/V heads repeated to the query count.</summary>
    WidenKv = 1,

    /// <summary>Operands permuted to another layout.</summary>
    Permute = 2,

    /// <summary>Operands cast to another dtype.</summary>
    Cast = 4,

    /// <summary>A mask rule built as a bias tensor.</summary>
    MaterializeMask = 8,

    /// <summary>A broadcast mask expanded to full rows.</summary>
    ExpandMask = 16,

    /// <summary>The full score matrix held in memory.</summary>
    MaterializeScores = 32,

    /// <summary>Computed on the host from device data.</summary>
    HostReference = 64,
}

/// <summary>What a kernel row can serve, declared up front so a planner and a test can reason about it without a GPU.
/// A row's runtime admission (<see cref="AttentionKernelRow.Admits"/>) may only narrow this, never widen it.</summary>
public sealed record AttentionKernelCapability
{
    /// <summary>Stable row id, reported in plans.</summary>
    public required string Id { get; init; }

    /// <summary>The legacy entry point this row belongs to.</summary>
    public required AttentionRoute Entry { get; init; }

    /// <summary>Query/output dtypes.</summary>
    public required AttentionDTypes QueryTypes { get; init; }

    /// <summary>Key/value dtypes.</summary>
    public required AttentionDTypes KvTypes { get; init; }

    /// <summary>Layouts of each of Q, K, V and the output; they may differ.</summary>
    public required AttentionLayouts Layouts { get; init; }

    /// <summary>Reads operands through their offsets and strides; otherwise only a whole tensor in its layout's natural strides.</summary>
    public bool StridedViews { get; init; }

    /// <summary>Largest batch.</summary>
    public int MaxBatch { get; init; } = int.MaxValue;

    /// <summary>Smallest head dim.</summary>
    public int MinHeadDim { get; init; }

    /// <summary>Largest head dim.</summary>
    public int MaxHeadDim { get; init; } = int.MaxValue;

    /// <summary>Mask components applied in-kernel (or by a declared lowering).</summary>
    public AttentionMaskSupport Masks { get; init; }

    /// <summary>Reads fewer K/V heads than Q heads in place.</summary>
    public bool Gqa { get; init; }

    /// <summary>Soft-cap, sinks and ALiBi.</summary>
    public bool LogitTerms { get; init; }

    /// <summary>Reads positions from a device buffer.</summary>
    public bool DevicePositions { get; init; }

    /// <summary>No allocation, sync, free-memory read or plan build on the call, so a step graph may capture it.</summary>
    public bool CaptureSafe { get; init; }

    /// <summary>Fewest query rows.</summary>
    public int MinQueryTokens { get; init; }

    /// <summary>Most query rows.</summary>
    public int MaxQueryTokens { get; init; } = int.MaxValue;

    /// <summary>Formats the QKᵀ multiply runs in, beyond F32.</summary>
    public AttentionFormats UsesQueryKey { get; init; }

    /// <summary>Formats the PV multiply runs in, beyond F32.</summary>
    public AttentionFormats UsesProbValue { get; init; }

    /// <summary>The row multiplies in each operand's own storage format, so an operand in a refused format excludes it whatever
    /// <see cref="UsesQueryKey"/> and <see cref="UsesProbValue"/> declare (an F16 operand runs F16 math).</summary>
    public bool ComputesInOperandFormat { get; init; }

    /// <summary>Clips out-of-range values when narrowing.</summary>
    public bool Saturates { get; init; }

    /// <summary>The format F32 operands are narrowed to, which the call must permit for both multiplies; None for a row that
    /// narrows nothing or still admits on a legacy flag.</summary>
    public AttentionFormats NarrowsF32To { get; init; }

    /// <summary>Work added around the kernel.</summary>
    public AttentionLowering Lowering { get; init; }

    /// <summary>The first declared limit <paramref name="call"/> exceeds, or <see cref="AttentionExclusion.None"/> when the
    /// row can serve it. Allocation-free: it runs on every call for every row ahead of the chosen one.</summary>
    public AttentionExclusion Excludes(in AttentionCall call)
    {
        if (call.Route != AttentionRoute.Auto && call.Route != Entry) return AttentionExclusion.Entry;
        if (!Has(QueryTypes, call.Query.DType) || !Has(QueryTypes, call.Output.DType)) return AttentionExclusion.QueryType;
        if (!Has(KvTypes, call.Key.DType) || !Has(KvTypes, call.Value.DType)) return AttentionExclusion.KvType;
        if (NarrowsF32To != AttentionFormats.None && !call.Numerics.Permits(NarrowsF32To, NarrowsF32To)
            && (call.Query.DType == DType.F32 || call.Key.DType == DType.F32 || call.Value.DType == DType.F32))
            return AttentionExclusion.Numerics;
        if ((Layouts & Of(call.Query.Layout)) == 0 || (Layouts & Of(call.Key.Layout)) == 0
            || (Layouts & Of(call.Value.Layout)) == 0 || (Layouts & Of(call.Output.Layout)) == 0) return AttentionExclusion.Layout;
        if (!StridedViews && !(call.Query.IsPlain && call.Key.IsPlain && call.Value.IsPlain && call.Output.IsPlain)) return AttentionExclusion.Strides;
        if (call.Batch > MaxBatch) return AttentionExclusion.Batch;
        if (call.QkHeadDim < MinHeadDim || call.QkHeadDim > MaxHeadDim) return AttentionExclusion.HeadDim;
        if (call.KvHeads != call.Heads && !Gqa) return AttentionExclusion.GroupedKv;
        if (call.QueryTokens < MinQueryTokens || call.QueryTokens > MaxQueryTokens) return AttentionExclusion.QueryRows;
        if (call.Logits.Any && !LogitTerms) return AttentionExclusion.LogitTerms;
        if (call.Positions.OnDevice && !DevicePositions) return AttentionExclusion.DevicePositions;
        if (((UsesQueryKey | UsesProbValue) & call.Numerics.Refused) != 0) return AttentionExclusion.RefusedFormat;
        if (ComputesInOperandFormat && (AttentionNumerics.OperandFormats(in call) & call.Numerics.Refused) != 0)
            return AttentionExclusion.RefusedFormat;
        AttentionMask m = call.Mask;
        if (m.Causal && (Masks & AttentionMaskSupport.Causal) == 0) return AttentionExclusion.Causal;
        if ((m.WindowLeft is not null || m.WindowRight is not null) && (Masks & AttentionMaskSupport.Window) == 0) return AttentionExclusion.Window;
        if (m.BlockSize > 0 && (Masks & AttentionMaskSupport.BlockCausal) == 0) return AttentionExclusion.BlockCausal;
        if (m.Bias is { } bias)
        {
            bool perHead = bias.Shape.Rank >= 3 && bias.Shape[bias.Shape.Rank - 3] > 1;
            if ((Masks & (perHead ? AttentionMaskSupport.PerHeadBias : AttentionMaskSupport.HeadSharedBias)) == 0)
                return perHead ? AttentionExclusion.PerHeadBias : AttentionExclusion.Bias;
        }
        return AttentionExclusion.None;
    }

    private static bool Has(AttentionDTypes set, DType t) => (set & Of(t)) != 0;

    private static AttentionDTypes Of(DType t) =>
        t == DType.F32 ? AttentionDTypes.F32 : t == DType.F16 ? AttentionDTypes.F16 : t == DType.BF16 ? AttentionDTypes.BF16
        : t.IsFp8 ? AttentionDTypes.Fp8 : AttentionDTypes.Other;

    private static AttentionLayouts Of(AttentionLayout l) => l switch
    {
        AttentionLayout.HeadMajor => AttentionLayouts.HeadMajor,
        AttentionLayout.TokenMajor => AttentionLayouts.TokenMajor,
        _ => AttentionLayouts.PackedRows,
    };
}

/// <summary>The declared limit a call exceeded.</summary>
public enum AttentionExclusion
{
    /// <summary>Covered.</summary>
    None,

    /// <summary>The call is pinned to another entry point.</summary>
    Entry,

    /// <summary>Query or output dtype.</summary>
    QueryType,

    /// <summary>Key or value dtype.</summary>
    KvType,

    /// <summary>Operand layout.</summary>
    Layout,

    /// <summary>An offset or strided operand on a row that reads whole tensors.</summary>
    Strides,

    /// <summary>Batch.</summary>
    Batch,

    /// <summary>Head dim.</summary>
    HeadDim,

    /// <summary>Fewer K/V heads than Q heads.</summary>
    GroupedKv,

    /// <summary>Query row count.</summary>
    QueryRows,

    /// <summary>Soft-cap, sinks or ALiBi.</summary>
    LogitTerms,

    /// <summary>Device positions.</summary>
    DevicePositions,

    /// <summary>Causal.</summary>
    Causal,

    /// <summary>A window.</summary>
    Window,

    /// <summary>Block-causal.</summary>
    BlockCausal,

    /// <summary>A head-shared bias.</summary>
    Bias,

    /// <summary>A per-head bias.</summary>
    PerHeadBias,

    /// <summary>A multiply format the call refuses.</summary>
    RefusedFormat,

    /// <summary>A narrowing the call does not permit.</summary>
    Numerics,
}

/// <summary>One kernel a backend can run for attention: its declared capability, the runtime conditions that admit a call
/// (knobs, kernel presence, failure state, budget) and the launch.</summary>
public abstract class AttentionKernelRow
{
    /// <summary>What the row can serve.</summary>
    public abstract AttentionKernelCapability Capability { get; }

    /// <summary>Runtime admission; the planner asks only after <see cref="AttentionKernelCapability.Excludes"/> passed.</summary>
    public abstract bool Admits(in AttentionCall call);

    /// <summary>Runs the call. False is a recoverable decline (no state changed) and the next admitted row runs.</summary>
    public abstract bool Run(in AttentionCall call);

    /// <summary>The variant a run took, when one row covers several (a materialized row's dtype path); null otherwise.</summary>
    public virtual string? Variant(in AttentionCall call) => null;
}

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

/// <summary>The row each attention shape class ran on, logged once per class at debug level.</summary>
public static class AttentionPlanLog
{
    private static readonly ConcurrentDictionary<string, byte> Seen = new(StringComparer.Ordinal);

    /// <summary>Logs <paramref name="row"/> for <paramref name="call"/>'s shape class the first time that class runs on
    /// <paramref name="backend"/>. Sequence lengths are bucketed to powers of two so decode does not log every token.</summary>
    public static void Ran(string backend, AttentionKernelRow row, in AttentionCall call)
    {
        if (Logs.MinLevel > LogLevel.Debug) return;
        static uint Bucket(int n) => BitOperations.RoundUpToPowerOf2((uint)Math.Max(1, n));
        string mask = (call.Mask.Causal ? " causal" : "") + (call.Mask.WindowLeft is { } w ? $" window{w}" : "")
            + (call.Mask.Bias is { } b ? $" bias{b.Shape}" : "");
        string key = $"{backend} {call.Route} {call.Query.DType}/{call.Key.DType} h{call.Heads}/{call.KvHeads} d{call.QkHeadDim} "
            + $"sq<={Bucket(call.QueryTokens)} skv<={Bucket(call.KeyTokens)}{mask}";
        if (Seen.TryAdd(key, 0))
            Logs.Debug($"Attention {key}: {new AttentionPlan(row, row.Variant(in call), "ran", null).Describe()}");
    }

    /// <summary>Warns once per backend type that its attention runs on the host because it declares no kernel rows.</summary>
    public static void HostReference(string backend)
    {
        if (Seen.TryAdd("host-reference " + backend, 0))
            Logs.Warning($"{backend} declares no attention rows: attention runs on the host reference.");
    }
}

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
