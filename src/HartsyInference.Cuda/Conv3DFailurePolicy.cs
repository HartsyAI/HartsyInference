namespace HartsyInference.Cuda;

/// <summary>Decides whether a failed native causal 3-D convolution disables that path for the whole session.</summary>
internal static class Conv3DFailurePolicy
{
    /// <summary>True for a deterministic failure, which disables the native 3-D path for the session. False when an
    /// allocator OOM-retry began during the call (<paramref name="oomRetriesDuringCall"/> &gt; 0): the shortfall belongs
    /// to that moment, so only the call falls back to per-tap.</summary>
    public static bool DisablesSession(long oomRetriesDuringCall) => oomRetriesDuringCall == 0;
}
