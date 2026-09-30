namespace HartsyInference.Core.Backends;

/// <summary>Counts one step-graph owner's failed captures, so a failure disables capture only once it repeats.</summary>
/// <remarks>A capture can fail for a reason outside its owner: on a blocking compute stream, any use of the legacy stream
/// in the same context, such as another engine's synchronous copy on the same GPU, invalidates it. That failure is
/// transient; a capture-illegal op in the owner's own step fails every time and still ends in eager for good.</remarks>
public sealed class StepGraphFailureBudget
{
    /// <summary>Failed captures after which the owner stops capturing.</summary>
    public const int MaxFailures = 3;

    private int _failures;

    /// <summary>Records one failed capture; true once the owner should stop capturing for good.</summary>
    public bool RecordFailure() => ++_failures >= MaxFailures;
}
