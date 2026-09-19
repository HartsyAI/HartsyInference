namespace HartsyInference.Diffusion.Prompting;

/// <summary>Whether cached conditioning may be reused for the request in hand.</summary>
/// <remarks><para>Pipelines cache text conditioning across generations so a repeat prompt skips the whole encoder
/// phase, and they key it on the token ids. Prompt weighting breaks that key: once the recipe has taken the
/// emphasis off the text, <c>(cat:1.5)</c> and <c>cat</c> tokenize to the SAME ids. A pipeline that stores
/// already-blended conditioning would then serve the weighted tensor to the next plain request, and the plain one
/// back to a weighted repeat.</para>
/// <para>The failure only shows on the SECOND generation of a prompt inside one process, so it survives any gate
/// that runs one image per invocation. This exists to make the rule testable rather than remembered.</para>
/// <para>A pipeline that caches the PLAIN conditioning and blends a per-request copy does not need this — its
/// cached tensor is weight-independent by construction. Which of the two a family can do is decided by whether it
/// trims the conditioning before caching it.</para></remarks>
public static class ConditioningCacheKey
{
    /// <summary>True when <paramref name="cachedIds"/>/<paramref name="cachedWeights"/> describe the same prompt
    /// AND the same emphasis as the request's.</summary>
    public static bool Matches(int[]? cachedIds, float[]? cachedWeights, int[] requestIds, float[]? requestWeights)
    {
        ArgumentNullException.ThrowIfNull(requestIds);
        return cachedIds is not null
            && cachedIds.AsSpan().SequenceEqual(requestIds)
            && WeightsMatch(cachedWeights, requestWeights);
    }

    /// <summary>Whether two weight arrays describe the same emphasis. Null means "unweighted", which is distinct
    /// from an all-ones array only in that the unweighted path never built one.</summary>
    public static bool WeightsMatch(float[]? cached, float[]? request) =>
        cached is null ? request is null : request is not null && cached.AsSpan().SequenceEqual(request);
}
