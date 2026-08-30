using HartsyInference.Engine.Planning;

namespace HartsyInference.API.Endpoints;

/// <summary>Client-safe video preflight result with no server filesystem or artifact inventory details.</summary>
public sealed class NativeVideoPlanResponse
{
    /// <summary>Detected checkpoint and adapter profile.</summary>
    public required NativeVideoModelProfile Profile { get; init; }

    /// <summary>Resolved values generation will execute.</summary>
    public required VideoEffectiveSettings EffectiveSettings { get; init; }

    /// <summary>Blocking errors and non-blocking warnings.</summary>
    public required IReadOnlyList<VideoPlanIssue> Issues { get; init; }

    /// <summary>Resolved component formats keyed by role, without local paths or hashes.</summary>
    public required IReadOnlyDictionary<string, string> ComponentFormats { get; init; }

    /// <summary>Whether preflight found no blocking issue.</summary>
    public required bool IsValid { get; init; }

    /// <summary>Projects the Engine plan onto the remote HTTP contract.</summary>
    internal static NativeVideoPlanResponse Create(VideoPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        return new NativeVideoPlanResponse
        {
            Profile = NativeVideoModelProfile.Create(plan.Profile),
            EffectiveSettings = plan.EffectiveSettings,
            Issues = plan.Issues,
            ComponentFormats = plan.ComponentFormats,
            IsValid = plan.IsValid,
        };
    }
}
