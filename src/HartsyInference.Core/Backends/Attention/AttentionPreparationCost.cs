using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>The preparation a row expects to run for one call, in bytes produced, by the evidence kind that records the
/// step when it runs. It is an estimate made from the call's shapes and types before anything launches;
/// <see cref="AttentionEvidence"/> is what ran. A row states it so that planning can compare candidates by the copying
/// they need, and a test compares each row's statement with its evidence.</summary>
/// <param name="Cast">Operands and outputs converted to another type.</param>
/// <param name="Repeat">Key/value heads repeated up to the query heads.</param>
/// <param name="Permute">Operands and outputs moved between token-major and head-major.</param>
/// <param name="MaterializeMask">A causal or window mask built as a tensor.</param>
/// <param name="ExpandMask">A stored mask expanded to the layout the kernel reads.</param>
/// <param name="MaterializeScores">The score matrix built in full.</param>
/// <param name="HostReference">Tensors that cross to the host for the host reference.</param>
public readonly record struct AttentionPreparationCost(long Cast = 0, long Repeat = 0, long Permute = 0, long MaterializeMask = 0,
    long ExpandMask = 0, long MaterializeScores = 0, long HostReference = 0)
{
    /// <summary>No preparation.</summary>
    public static AttentionPreparationCost None => default;

    /// <summary>Every step's bytes together: the one figure candidates are compared by.</summary>
    public long Total => Cast + Repeat + Permute + MaterializeMask + ExpandMask + MaterializeScores + HostReference;

    /// <summary>The estimate for evidence kind <paramref name="kind"/>; 0 for a kind a row does not estimate.</summary>
    public long Of(AttentionEvidenceKind kind) => kind switch
    {
        AttentionEvidenceKind.Cast => Cast,
        AttentionEvidenceKind.Repeat => Repeat,
        AttentionEvidenceKind.Permute => Permute,
        AttentionEvidenceKind.MaterializeMask => MaterializeMask,
        AttentionEvidenceKind.ExpandMask => ExpandMask,
        AttentionEvidenceKind.MaterializeScores => MaterializeScores,
        AttentionEvidenceKind.HostReference => HostReference,
        _ => 0,
    };

    /// <summary>Bytes of the whole tensor behind <paramref name="operand"/>.</summary>
    public static long Bytes(in AttentionOperand operand) => operand.Storage.DType.ComputeByteCount(operand.Storage.ElementCount);

    /// <summary>Bytes of the operand's elements once stored as <paramref name="type"/>.</summary>
    public static long BytesAs(in AttentionOperand operand, DType type) => type.ComputeByteCount(operand.Storage.ElementCount);

    /// <summary>Bytes of casting each of Q, K and V that is not already <paramref name="compute"/> to it, and of casting the
    /// output back when it is not <paramref name="compute"/> either.</summary>
    public static long CastsTo(in AttentionCall call, DType compute)
    {
        long bytes = 0;
        if (call.Query.DType != compute) bytes += BytesAs(call.Query, compute);
        if (call.Key.DType != compute) bytes += BytesAs(call.Key, compute);
        if (call.Value.DType != compute) bytes += BytesAs(call.Value, compute);
        if (call.Output.DType != compute) bytes += Bytes(call.Output);
        return bytes;
    }

    /// <summary>One line for a plan description: the total and the steps that make it up.</summary>
    public override string ToString()
    {
        if (Total == 0) return "0 B";
        System.Text.StringBuilder sb = new();
        sb.Append(Total).Append(" B (");
        int parts = 0;
        void Part(string name, long bytes)
        {
            if (bytes == 0) return;
            sb.Append(parts++ > 0 ? ", " : "").Append(name).Append(' ').Append(bytes);
        }
        Part("cast", Cast);
        Part("repeat", Repeat);
        Part("permute", Permute);
        Part("mask", MaterializeMask);
        Part("expand-mask", ExpandMask);
        Part("scores", MaterializeScores);
        Part("host", HostReference);
        return sb.Append(')').ToString();
    }
}
