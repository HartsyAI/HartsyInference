namespace HartsyInference.Engine.Planning;

/// <summary>Raised before model construction when a video plan contains one or more blocking issues.</summary>
public sealed class VideoPlanningException : Exception
{
    /// <summary>The rejected plan, including stable issue codes for transport mapping.</summary>
    public VideoPlan Plan { get; }

    /// <summary>Creates an exception from a rejected preflight result.</summary>
    public VideoPlanningException(VideoPlan plan)
        : base(BuildMessage(plan))
    {
        Plan = plan ?? throw new ArgumentNullException(nameof(plan));
    }

    private static string BuildMessage(VideoPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        IEnumerable<string> errors = plan.Issues
            .Where(issue => issue.Severity == VideoPlanIssueSeverity.Error)
            .Select(issue => $"{issue.Code}: {issue.Message}");
        return $"Video request failed preflight: {string.Join("; ", errors)}";
    }
}
