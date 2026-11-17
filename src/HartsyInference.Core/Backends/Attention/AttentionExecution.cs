namespace HartsyInference.Core.Backends.Attention;

/// <summary>Shared preflight and position normalization between the public attention entries and backend-owned rows.</summary>
internal static class AttentionExecution
{
    /// <summary>Validates and plans one call without launching it.</summary>
    public static AttentionPlan Plan(IReadOnlyList<AttentionKernelRow> rows, in AttentionCall call) =>
        AttentionPlanner.Plan(rows, Prepare(in call));

    /// <summary>Normalizes and plans a call whose boundary was already validated.</summary>
    public static AttentionPlan PlanValidated(IReadOnlyList<AttentionKernelRow> rows, in AttentionCall call) =>
        AttentionPlanner.Plan(rows, Normalize(in call));

    /// <summary>Validates, normalizes and runs one call through the supplied rows.</summary>
    public static AttentionKernelRow Run(IReadOnlyList<AttentionKernelRow> rows, in AttentionCall call) =>
        AttentionPlanner.Run(rows, Prepare(in call));

    /// <summary>Normalizes and runs a call whose boundary was already validated.</summary>
    public static AttentionKernelRow RunValidated(IReadOnlyList<AttentionKernelRow> rows, in AttentionCall call) =>
        AttentionPlanner.Run(rows, Normalize(in call));

    /// <summary>Validates a call and gives unbounded flash rows the full key prefix their legacy entries receive.</summary>
    public static AttentionCall Prepare(in AttentionCall call)
    {
        AttentionCallValidation.Validate(in call);
        return Normalize(in call);
    }

    private static AttentionCall Normalize(in AttentionCall call) =>
        call.Route is AttentionRoute.Flash or AttentionRoute.FlashDevice && !call.Positions.Bounded
            ? call with { Positions = call.Positions with { KvLength = call.KeyTokens, Bounded = true } }
            : call;
}
