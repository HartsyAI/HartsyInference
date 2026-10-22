using System.Globalization;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Planning;

/// <summary>Builds the per-parameter provenance of a resolved generation by comparing what the caller requested with
/// what will run: unset values carry the default's source, set values that run unchanged are user overrides, and set
/// values planning had to change are normalizations naming the requested value.</summary>
public static class ParameterProvenance
{
    /// <summary>The provenance of one numeric parameter.</summary>
    public static ResolvedParameter Of<T>(T? requested, T effective, ParameterSource unsetSource)
        where T : struct, IFormattable, IEquatable<T>
    {
        string value = Format(effective);
        if (requested is not T asked)
        {
            return new ResolvedParameter(value, unsetSource);
        }
        return asked.Equals(effective)
            ? new ResolvedParameter(value, ParameterSource.UserOverride)
            : new ResolvedParameter(value, ParameterSource.Normalized, $"requested {Format(asked)}");
    }

    /// <summary>The provenance of one optional numeric parameter a family may leave to its pipeline.</summary>
    public static ResolvedParameter Of<T>(T? requested, T? effective, ParameterSource unsetSource)
        where T : struct, IFormattable, IEquatable<T> =>
        effective is T value ? Of(requested, value, unsetSource) : new ResolvedParameter(null, unsetSource);

    /// <summary>The provenance of one named choice, such as a sampler, compared case-insensitively.</summary>
    public static ResolvedParameter Of(string? requested, string? effective, ParameterSource unsetSource)
    {
        if (requested is null)
        {
            return new ResolvedParameter(effective, unsetSource);
        }
        return string.Equals(requested, effective, StringComparison.OrdinalIgnoreCase)
            ? new ResolvedParameter(effective, ParameterSource.UserOverride)
            : new ResolvedParameter(effective, ParameterSource.Normalized, $"requested {requested}");
    }

    /// <summary>A seed: the caller's when it named one, otherwise drawn at random.</summary>
    /// <remarks>A caller's seed outside the 31-bit range the samplers take is wrapped into it, so it reports as normalized.</remarks>
    public static ResolvedParameter Seed(long requested, long effective) =>
        RecipeRequestMapper.MapSeed(requested) is null
            ? new ResolvedParameter(Format(effective), ParameterSource.Derived, "random")
            : Of((long?)requested, effective, ParameterSource.UserOverride);

    /// <summary>The provenance of every tunable a video plan resolves.</summary>
    public static IReadOnlyDictionary<string, ResolvedParameter> ForVideo(VideoRequest request,
        VideoEffectiveSettings effective, ParameterSource unsetSource)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(effective);
        return new Dictionary<string, ResolvedParameter>(StringComparer.Ordinal)
        {
            [nameof(VideoRequest.Width)] = Of(request.Width, effective.Width, unsetSource),
            [nameof(VideoRequest.Height)] = Of(request.Height, effective.Height, unsetSource),
            [nameof(VideoRequest.Frames)] = Of(request.Frames, effective.Frames, unsetSource),
            [nameof(VideoRequest.Fps)] = Of(request.Fps, effective.Fps, unsetSource),
            [nameof(VideoRequest.Steps)] = Of(request.Steps, effective.Steps, unsetSource),
            [nameof(VideoRequest.CfgScale)] = Of(request.CfgScale, effective.CfgScale, unsetSource),
            [nameof(VideoRequest.FlowShift)] = Of(request.FlowShift, effective.FlowShift, unsetSource),
            [nameof(VideoRequest.AudioFlowShift)] = Of(request.AudioFlowShift, effective.AudioFlowShift, unsetSource),
            [nameof(VideoRequest.Sampler)] = Of(request.Sampler, effective.Sampler, unsetSource),
            [nameof(VideoRequest.Scheduler)] = Of(request.Scheduler, effective.Scheduler, unsetSource),
            [nameof(VideoRequest.Seed)] = Seed(request.Seed, effective.Seed),
        };
    }

    /// <summary>The provenance of every tunable an image request resolves, with the seed the result reports.</summary>
    public static IReadOnlyDictionary<string, ResolvedParameter> ForImage(ImageRequest request, ImageRequest resolved,
        long seed)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(resolved);
        const ParameterSource unset = ParameterSource.RecipeDefault;
        return new Dictionary<string, ResolvedParameter>(StringComparer.Ordinal)
        {
            [nameof(ImageRequest.Width)] = Of(request.Width, resolved.Width, unset),
            [nameof(ImageRequest.Height)] = Of(request.Height, resolved.Height, unset),
            [nameof(ImageRequest.Steps)] = Of(request.Steps, resolved.Steps, unset),
            [nameof(ImageRequest.CfgScale)] = Of(request.CfgScale, resolved.CfgScale, unset),
            [nameof(ImageRequest.Sampler)] = Of(request.Sampler, resolved.Sampler, unset),
            [nameof(ImageRequest.Scheduler)] = Of(request.Scheduler, resolved.Scheduler, unset),
            [nameof(ImageRequest.SigmaShift)] = Of(request.SigmaShift, resolved.SigmaShift, unset),
            [nameof(ImageRequest.Seed)] = Seed(request.Seed, seed),
        };
    }

    /// <summary>A copy of <paramref name="provenance"/> with <paramref name="name"/> replaced.</summary>
    public static IReadOnlyDictionary<string, ResolvedParameter> With(
        IReadOnlyDictionary<string, ResolvedParameter> provenance, string name, ResolvedParameter parameter)
    {
        ArgumentNullException.ThrowIfNull(provenance);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        return new Dictionary<string, ResolvedParameter>(provenance, StringComparer.Ordinal) { [name] = parameter };
    }

    /// <summary>One log line: every parameter with its value and source, in a stable order.</summary>
    public static string Describe(RecipeIdentity recipe, IReadOnlyDictionary<string, ResolvedParameter> provenance)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        ArgumentNullException.ThrowIfNull(provenance);
        IEnumerable<string> parts = provenance.OrderBy(entry => entry.Key, StringComparer.Ordinal)
            .Select(entry => $"{entry.Key}={entry.Value}");
        return $"{recipe}: {string.Join(", ", parts)}";
    }

    private static string Format<T>(T value) where T : IFormattable =>
        value.ToString(value is float or double ? "R" : null, CultureInfo.InvariantCulture);
}
