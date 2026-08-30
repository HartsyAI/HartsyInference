using System.Text.RegularExpressions;
using HartsyInference.Engine.Planning;

namespace HartsyInference.API.Endpoints;

/// <summary>Client-safe video preflight result with no server filesystem or artifact inventory details.</summary>
public sealed class NativeVideoPlanResponse
{
    private static readonly Regex _sha256 = new(
        @"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])",
        RegexOptions.CultureInvariant);

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
            Issues = SanitizeIssues(plan.Issues),
            ComponentFormats = plan.ComponentFormats,
            IsValid = plan.IsValid,
        };
    }

    /// <summary>Projects diagnostics onto the remote contract without disclosing host paths or exact artifact
    /// identities that may have been included by a low-level loader or hash-binding check.</summary>
    internal static IReadOnlyList<VideoPlanIssue> SanitizeIssues(IReadOnlyList<VideoPlanIssue> issues)
    {
        ArgumentNullException.ThrowIfNull(issues);
        return Array.AsReadOnly(issues.Select(SanitizeIssue).ToArray());
    }

    /// <summary>Creates one client-safe diagnostic while preserving its stable code, severity, and request field.</summary>
    internal static VideoPlanIssue SanitizeIssue(VideoPlanIssue issue)
    {
        ArgumentNullException.ThrowIfNull(issue);
        string message = ContainsRootedPath(issue.Message)
            ? $"Local artifact details were omitted; use issue code '{issue.Code}' and server logs for diagnosis."
            : _sha256.Replace(issue.Message, "<redacted-sha256>");
        return issue with { Message = message };
    }

    private static bool ContainsRootedPath(string message)
    {
        char[] separators = [' ', '\t', '\r', '\n', '\'', '"', '`', '(', ')', '[', ']', '{', '}', ',', ';', '='];
        foreach (string token in message.Split(separators, StringSplitOptions.RemoveEmptyEntries))
        {
            string candidate = token.TrimEnd('.', ':', '!', '?');
            if ((candidate.Length > 0 && candidate[0] == '/')
                || candidate.StartsWith(@"\\", StringComparison.Ordinal)
                || candidate.StartsWith("file:", StringComparison.OrdinalIgnoreCase)
                || (candidate.Length >= 3 && char.IsAsciiLetter(candidate[0]) && candidate[1] == ':'
                    && candidate[2] is '/' or '\\'))
            {
                return true;
            }
        }
        return false;
    }
}
