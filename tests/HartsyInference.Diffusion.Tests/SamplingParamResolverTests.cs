using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the engine-level sampler resolution, whose headline behaviour changed on 2026-08-20: an unavailable
/// sampler is now REFUSED by name instead of silently becoming Euler.
///
/// <para>The old behaviour was the single most user-hostile thing in the sampling surface. Someone migrating a ComfyUI
/// workflow asked for <c>dpmpp_2m_sde_karras</c>, got a Euler image plus a <c>Logs.Verbose</c> line on a server they
/// were not reading, and concluded the engine was broken. These tests exist because that failure is invisible in any
/// output-based check — the image is perfectly good, just not the one that was asked for.</para></summary>
public sealed class SamplingParamResolverTests
{
    private static ImageRequest Request(string? sampler = null, string? scheduler = null) =>
        new ImageRequest { Prompt = "test", Sampler = sampler, Scheduler = scheduler };

    /// <summary>No selection means no opinion — the family's own default, unchanged.</summary>
    [Fact]
    public void NoSelection_ResolvesToNull()
    {
        Assert.Null(SamplingParamResolver.ResolveSchedulerName(Request()));
    }

    /// <summary>The legacy SD-family names keep resolving exactly as before, so existing SD1.5/SDXL requests are
    /// untouched by the refusal.</summary>
    [Theory]
    [InlineData("ddim", "ddim")]
    [InlineData("dpm++2m", "dpm++2m")]
    [InlineData("dpmpp_2m", "dpmpp_2m")]
    [InlineData("lcm", "lcm")]
    [InlineData("tcd", "tcd")]
    public void LegacyNames_StillResolve(string requested, string expected)
    {
        Assert.Equal(expected, SamplingParamResolver.ResolveSchedulerName(Request(sampler: requested)));
    }

    /// <summary>A sampler from the new registry, and a compound selection, both pass through for the pipeline to split
    /// rather than being flattened here — the resolver does not know which family will run.</summary>
    [Theory]
    [InlineData("euler_ancestral")]
    [InlineData("dpmpp_2m_sde_karras")]
    [InlineData("heun_exponential")]
    public void RegistrySamplersAndCompounds_PassThrough(string requested)
    {
        Assert.Equal(requested, SamplingParamResolver.ResolveSchedulerName(Request(sampler: requested)));
    }

    /// <summary>A genuinely unknown sampler throws, and the message names both the value and what IS available — the
    /// whole point of the change.</summary>
    [Fact]
    public void UnknownSampler_ThrowsAndListsAlternatives()
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(
            () => SamplingParamResolver.ResolveSchedulerName(Request(sampler: "totally_made_up")));
        Assert.Contains("totally_made_up", ex.Message, StringComparison.Ordinal);
        Assert.Contains("dpmpp_2m", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>An unknown sigma schedule is refused separately from the sampler, so the message points at the half
    /// that is actually wrong.</summary>
    [Fact]
    public void UnknownSchedule_ThrowsNamingTheSchedule()
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(
            () => SamplingParamResolver.ResolveSchedulerName(Request(sampler: "euler_notaschedule")));
        Assert.Contains("euler_notaschedule", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>The legacy scheduler path refuses a sampler it cannot run, rather than returning Euler.
    ///
    /// <para>This is the other half of the fix, and it matters because <see cref="SamplingParamResolver"/> deliberately
    /// passes registry names through without knowing the target family. A family without the sampler seam
    /// (<c>StableDiffusion15Pipeline</c>, <c>SdxlRefinerPipeline</c>) hands the name straight to
    /// <see cref="SchedulerFactory"/> — which must refuse it, or the silent fallback simply moves one layer down.</para></summary>
    [Fact]
    public void SchedulerFactory_RefusesASamplerItCannotRun()
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => SchedulerFactory.Create("euler_ancestral"));
        Assert.Contains("euler_ancestral", ex.Message, StringComparison.Ordinal);
        Assert.Contains("sampler seam", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>The resolver is reached from the recipe pipelines, not left as an unreferenced helper.
    ///
    /// <para>It WAS unreferenced: <c>Sd15RecipePipeline</c> and <c>SdxlRecipePipeline</c> each read
    /// <c>request.Sampler ?? request.Scheduler</c> directly, so every validation added here would have been dead code
    /// and an unavailable sampler would only have failed later, after the checkpoint had loaded. This pins the wiring
    /// rather than the resolution — the behaviour is covered above; what is fragile is that somebody re-inlines the
    /// raw read and quietly disconnects the refusal again.</para></summary>
    [Fact]
    public void RecipePipelines_RouteThroughTheResolver()
    {
        string sd15 = File.ReadAllText(RepoFile("Recipes/Image/Sd15RecipePipeline.cs"));
        string sdxl = File.ReadAllText(RepoFile("Recipes/Image/SdxlRecipePipeline.cs"));

        Assert.Contains("SamplingParamResolver.ResolveSchedulerName(request)", sd15, StringComparison.Ordinal);
        Assert.Contains("SamplingParamResolver.ResolveSchedulerName(request)", sdxl, StringComparison.Ordinal);
        Assert.DoesNotContain("Scheduler = request.Sampler ?? request.Scheduler", sd15, StringComparison.Ordinal);
        Assert.DoesNotContain("Scheduler = request.Sampler ?? request.Scheduler", sdxl, StringComparison.Ordinal);
    }

    /// <summary>Locates an Engine source file from the test binary, walking up to the repo root.</summary>
    private static string RepoFile(string relative)
    {
        DirectoryInfo? dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "HartsyInference.sln")))
        {
            dir = dir.Parent;
        }
        Assert.NotNull(dir);
        return Path.Combine(dir!.FullName, "src", "HartsyInference.Engine", relative);
    }

    /// <summary>Null and the explicit default still give the Euler base, which is what a seam-carrying pipeline passes
    /// when its sampler supplies the integrator.</summary>
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("euler")]
    public void SchedulerFactory_DefaultsToEuler(string? name)
    {
        Assert.Equal("euler", SchedulerFactory.Create(name).Name);
    }
}
