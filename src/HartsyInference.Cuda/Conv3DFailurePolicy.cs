using System;

namespace HartsyInference.Cuda;

/// <summary>Decides whether a failed native causal 3-D convolution disables that path for the whole session.</summary>
internal static class Conv3DFailurePolicy
{
    /// <summary>True for a deterministic failure, which disables the native 3-D path for the session. False when an
    /// allocator OOM-retry began during the failing stage (<paramref name="oomRetriesInFailingStage"/> &gt; 0): the
    /// shortfall belongs to that moment, so only the call falls back to per-tap.</summary>
    public static bool DisablesSession(long oomRetriesInFailingStage) => oomRetriesInFailingStage == 0;
}

/// <summary>Names the stage of one native 3-D conv call and counts allocator OOM-retries seen inside that stage only,
/// so a retry in an earlier stage does not excuse a later deterministic failure.</summary>
internal sealed class Conv3DStageTracker(Func<long> oomRetries)
{
    private long _stageStart = oomRetries();

    public string Name { get; private set; } = "setup";

    public void Enter(string name)
    {
        Name = name;
        _stageStart = oomRetries();
    }

    /// <summary>True when no OOM-retry began during the stage that is failing now.</summary>
    public bool DisablesSession => Conv3DFailurePolicy.DisablesSession(oomRetries() - _stageStart);
}
