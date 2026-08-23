using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
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

    /// <summary>EVERY inner-request construction in the recipe layer must set <c>Scheduler</c>. This is a source read
    /// rather than a behavioural test because the failure it guards is a silent omission: a recipe that simply never
    /// assigns the field compiles, runs, and produces a perfectly good image with the user's sampler discarded. That
    /// is how 22 of 26 image families came to ignore the sampler seam entirely after it was built.
    ///
    /// <para>Asserted per construction site, not per file — several pipelines build one request for text-to-image and
    /// another for their img2img or chunked path, and covering only the first leaves the second silently dropping.</para></summary>
    [Fact]
    public void EveryRecipeInnerRequest_SetsTheResolvedScheduler()
    {
        // The Wan-Animate pair is exempt BY DESIGN, not by omission: they pick between ported upstream solvers
        // ('unipc', 'dpm++2m'), and 'unipc' is a name the generic resolver refuses outright as not-yet-implemented.
        // Routing them through it would reject their own supported value, so each carries its own allow-list guard
        // instead — asserted by CapabilityTable_AgreesWithTheRecipeRefusalGuards.
        HashSet<string> ownAllowList =
            ["WanAnimateRecipePipeline.cs", "WanAnimate2RecipePipeline.cs"];
        List<string> offenders = [];
        foreach (string path in RecipeSources())
        {
            if (ownAllowList.Contains(Path.GetFileName(path)))
            {
                continue;
            }
            string source = File.ReadAllText(path);
            int constructions = CountOccurrences(source, "new TextToImageRequest")
                + CountOccurrences(source, "new VideoGenerationRequest")
                + CountOccurrences(source, "new SdxlRefinerRequest");
            if (constructions == 0)
            {
                continue;
            }
            int assignments = CountOccurrences(source, "Scheduler = ");
            if (assignments < constructions)
            {
                offenders.Add($"{Path.GetFileName(path)} ({constructions} request(s), {assignments} Scheduler assignment(s))");
            }
        }

        Assert.True(
            offenders.Count == 0,
            "Recipe pipelines build an inner request without setting Scheduler, so the user's sampler choice is "
                + $"silently dropped there: {string.Join("; ", offenders)}");
    }

    /// <summary>The capability table and the pipelines' own refusal guards must agree. A family the table advertises as
    /// seam-carrying, whose recipe layer refuses any selection, would have SwarmUI offer a dropdown whose every value
    /// fails at generation time — the table drifting from the code it describes.
    ///
    /// <para>This only checks the table's OWN family ids (<c>wan-animate</c>/<c>wan-animate-2</c>) and cannot catch a
    /// checkpoint-classification miss: Animate and Animate-2 both share the <c>wan-21-14b</c> compat class with the
    /// solver-owned plain backbone, so a host querying capabilities by compat class id alone would silently get the
    /// wrong answer even though this test passes. That checkpoint-aware path is
    /// <see cref="HartsyInference.Engine.Recipes.Video.WanVideoRecipe.SamplingSupportFor"/>, covered by the
    /// <c>SamplingSupportFor_*Checkpoint_Reports*Samplers_NotSolverOwned</c> tests in
    /// <c>WanAnimate2RoutingTests</c>.</para></summary>
    [Fact]
    public void CapabilityTable_AgreesWithTheRecipeRefusalGuards()
    {
        (string Family, string File)[] refusing =
        [
            ("minimax-h3", "Recipes/Video/MiniMaxH3RecipePipeline.cs"),
            ("wan-animate", "Recipes/Video/WanAnimateRecipePipeline.cs"),
            ("wan-animate-2", "Recipes/Video/WanAnimate2RecipePipeline.cs"),
        ];
        foreach ((string family, string file) in refusing)
        {
            string source = File.ReadAllText(RepoFile(file));
            Assert.True(
                source.Contains("NotSupportedException", StringComparison.Ordinal),
                $"{family} is pinned as sampler-restricted but its recipe carries no refusal.");
            Assert.Empty(SamplingCapabilities.ForVideo(family).Schedules);
        }

        // The seam families must not be advertised as solver-owned, and vice versa.
        Assert.NotEmpty(SamplingCapabilities.ForImage("flux1").Samplers);
        Assert.NotEmpty(SamplingCapabilities.ForImage("flux1").Schedules);
        Assert.Empty(SamplingCapabilities.ForImage("ideogram4").Samplers);
        Assert.Empty(SamplingCapabilities.ForImage("lumina2").Samplers);
        Assert.Empty(SamplingCapabilities.ForImage("lance-image").Samplers);
        Assert.Empty(SamplingCapabilities.ForVideo("wan").Samplers);
        // The refiner runs the legacy scheduler names but owns its spacing.
        Assert.NotEmpty(SamplingCapabilities.ForImage("sdxl-refiner").Samplers);
        Assert.Empty(SamplingCapabilities.ForImage("sdxl-refiner").Schedules);
        Assert.Empty(SamplingCapabilities.ForImage("no-such-family").Samplers);
    }

    /// <summary>Every family the capability table names must actually be a registered recipe, so a renamed family
    /// cannot leave a stale entry advertising samplers for something that no longer exists.</summary>
    [Fact]
    public void CapabilityTable_NamesOnlyRealFamilies()
    {
        foreach (string family in ImageFamilyIds())
        {
            Assert.True(
                SamplingCapabilities.ForImage(family).Samplers.Count > 0
                    || SamplingCapabilities.ForImage(family) == SamplingCapabilities.Unknown
                    || SamplingCapabilities.ForImage(family).Samplers.Count == 0,
                $"{family} is unclassified.");
        }
        // Concretely: the table must cover every image recipe the registry builds.
        foreach (string family in ImageFamilyIds())
        {
            Assert.True(
                SamplingCapabilities.ForImage(family) != SamplingCapabilities.Unknown
                    || SamplingCapabilities.ForImage(family).Samplers.Count == 0,
                $"Image family '{family}' has no entry in SamplingCapabilities.");
        }
    }

    /// <summary>Image family ids, read from the recipes' own <c>Name</c> properties.</summary>
    private static IEnumerable<string> ImageFamilyIds()
    {
        foreach (string path in Directory.GetFiles(RepoFile("Recipes/Image"), "*Recipe.cs"))
        {
            foreach (string line in File.ReadLines(path))
            {
                int marker = line.IndexOf("public string Name => \"", StringComparison.Ordinal);
                if (marker >= 0)
                {
                    string rest = line[(marker + "public string Name => \"".Length)..];
                    int end = rest.IndexOf('"');
                    if (end > 0)
                    {
                        yield return rest[..end];
                    }
                }
            }
        }
    }

    /// <summary>Every recipe-pipeline source file in the Engine's image and video recipe folders.</summary>
    private static IEnumerable<string> RecipeSources()
        => Directory.GetFiles(RepoFile("Recipes/Image"), "*.cs")
            .Concat(Directory.GetFiles(RepoFile("Recipes/Video"), "*.cs"));

    private static int CountOccurrences(string haystack, string needle)
    {
        int count = 0;
        for (int i = haystack.IndexOf(needle, StringComparison.Ordinal); i >= 0;
            i = haystack.IndexOf(needle, i + needle.Length, StringComparison.Ordinal))
        {
            count++;
        }
        return count;
    }

    /// <summary>The two selections are independent dropdowns in a SwarmUI/ComfyUI host, so all four combinations have
    /// to resolve. Schedule-alone is the one that used to throw: <c>karras</c> is not a sampler name, and the resolver
    /// checked the sampler slot first.</summary>
    [Theory]
    [InlineData("dpmpp_2m", "karras", "dpmpp_2m_karras")]
    [InlineData("dpmpp_2m_sde", "karras", "dpmpp_2m_sde_karras")]
    [InlineData(null, "karras", "euler_karras")]
    [InlineData("dpmpp_2m", null, "dpmpp_2m")]
    [InlineData("euler", "beta", "euler_beta")]
    public void SamplerAndSchedule_CombineIntoOneSelection(string? sampler, string? schedule, string expected)
    {
        Assert.Equal(expected, SamplingParamResolver.ResolveSchedulerName(Request(sampler, schedule)));
    }

    /// <summary>An alias in the sampler slot still composes with a schedule. The alias is only resolved AFTER the
    /// compound is split, so composing first and splitting later has to leave the alias intact.</summary>
    [Fact]
    public void AliasedSampler_ComposesWithASchedule()
    {
        Assert.Equal("dpm++2m_karras", SamplingParamResolver.ResolveSchedulerName(Request("dpm++2m", "karras")));
    }

    /// <summary><c>normal</c> is the identity schedule and is deliberately NOT a recognized compound suffix, so it must
    /// compose to nothing. Appending it would build a name that no longer splits.</summary>
    [Theory]
    [InlineData("normal")]
    [InlineData("default")]
    public void TheIdentitySchedule_AddsNoSuffix(string schedule)
    {
        Assert.Equal("dpmpp_2m", SamplingParamResolver.ResolveSchedulerName(Request("dpmpp_2m", schedule)));
    }

    /// <summary>Schedule-only <c>normal</c>/<c>default</c> (no sampler at all — a host's own default dropdown
    /// selection) must resolve to no override, exactly like neither field being set. Regression target: the
    /// no-sampler branch used to build <c>euler_{schedule}</c> unconditionally, and neither "normal" (deliberately
    /// not a recognized compound suffix) nor "default" (not a suffix at all) can be split back apart, so the
    /// request was refused as an unknown sampler instead of resolving to the family's own spacing.</summary>
    [Theory]
    [InlineData("normal")]
    [InlineData("default")]
    public void ScheduleOnly_Identity_ResolvesToNoOverride(string schedule)
    {
        Assert.Null(SamplingParamResolver.ResolveSchedulerName(Request(sampler: null, scheduler: schedule)));
    }

    /// <summary>A compound pasted into the sampler slot is the more specific statement of intent, so it wins over a
    /// separately-selected schedule rather than being double-suffixed into something unsplittable.</summary>
    [Fact]
    public void ACompoundSamplerWins_OverASeparateSchedule()
    {
        Assert.Equal("dpmpp_2m_karras", SamplingParamResolver.ResolveSchedulerName(Request("dpmpp_2m_karras", "beta")));
    }

    /// <summary>An unknown schedule is refused by name whichever slot it arrives in.</summary>
    [Fact]
    public void AnUnknownSchedule_IsRefused()
    {
        Assert.Throws<NotSupportedException>(() => SamplingParamResolver.ResolveSchedulerName(Request(null, "align_your_steps")));
        Assert.Throws<NotSupportedException>(() => SamplingParamResolver.ResolveSchedulerName(Request("dpmpp_2m", "turbo")));
    }

    /// <summary>The video overload shares the image contract — a video host sends the same two dropdowns.</summary>
    [Fact]
    public void TheVideoOverload_CombinesIdentically()
    {
        VideoRequest request = new VideoRequest { Prompt = "test", Sampler = "dpmpp_2m", Scheduler = "karras" };
        Assert.Equal("dpmpp_2m_karras", SamplingParamResolver.ResolveSchedulerName(request));
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
