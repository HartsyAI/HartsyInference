using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe;

/// <summary>
/// Runs one layer's planned experts on the side each assignment names: GPU assignments through an
/// <see cref="IExpertDeviceRunner"/>, CPU assignments through the F32 reference. Rows are expert-major, as
/// <c>MoeBuildDispatch</c> lays them out: the rows of each planned expert are contiguous, in plan order, and the output uses the
/// same layout. Placement changes where an expert runs, never which rows it receives or where its output lands.
/// </summary>
public static class HeterogeneousExpertExecutor
{
    /// <summary>Executes the plan.</summary>
    /// <param name="program">Activation and clamp bounds, shared by both sides.</param>
    /// <param name="plan">The planner's assignments, in the order their rows are laid out.</param>
    /// <param name="gathered"><c>(sum of rows) × H</c> inputs, expert-major.</param>
    /// <param name="hidden">Model width H.</param>
    /// <param name="output"><c>(sum of rows) × H</c> outputs, same layout as <paramref name="gathered"/>.</param>
    /// <param name="hostWeights">F32 weights of an expert, for CPU assignments.</param>
    /// <param name="device">Runs GPU assignments; required when the plan has any.</param>
    /// <remarks>If a device run throws, <paramref name="output"/> is left partially written and must be discarded.</remarks>
    /// <exception cref="ArgumentException">The buffers do not match the plan's rows, or an assignment serves no rows.</exception>
    /// <exception cref="InvalidOperationException">The plan places experts on the GPU but no device runner was supplied.</exception>
    public static void Execute(ExpertProgram program, ReadOnlySpan<ExpertAssignment> plan, ReadOnlySpan<float> gathered, int hidden,
        Span<float> output, Func<ExpertKey, F32ExpertWeights> hostWeights, IExpertDeviceRunner? device)
    {
        ArgumentNullException.ThrowIfNull(program);
        ArgumentNullException.ThrowIfNull(hostWeights);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hidden);
        program.Validated();

        long pairs = 0;
        bool anyGpu = false;
        foreach (ExpertAssignment assignment in plan)
        {
            if (assignment.Rows <= 0) throw new ArgumentException("Every planned expert serves at least one row.", nameof(plan));
            pairs += assignment.Rows;
            anyGpu |= assignment.Placement == ExpertPlacement.Gpu;
        }
        if (gathered.Length != pairs * hidden) throw new ArgumentException($"Gathered rows must hold {pairs} rows of {hidden}.", nameof(gathered));
        if (output.Length != gathered.Length) throw new ArgumentException("The output must match the gathered rows.", nameof(output));
        if (anyGpu && device is null) throw new InvalidOperationException("The plan places experts on the GPU, but no device runner was supplied.");

        int offset = 0;
        foreach (ExpertAssignment assignment in plan)
        {
            int length = assignment.Rows * hidden;
            ReadOnlySpan<float> x = gathered.Slice(offset, length);
            Span<float> y = output.Slice(offset, length);
            if (assignment.Placement == ExpertPlacement.Gpu) device!.Run(assignment.Key, x, assignment.Rows, y);
            else ExpertProgramReference.Apply(program, hostWeights(assignment.Key), x, assignment.Rows, y);
            offset += length;
        }
    }
}
