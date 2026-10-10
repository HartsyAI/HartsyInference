using HartsyInference.Engine.Recipes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins which video families declare init-image / end-frame conditioning.
///
/// <para>The video side previously had <b>no</b> feature gate while transports already populated
/// <c>VideoRequest.InitImage</c>, so a family that never read it produced a text-to-video clip and gave no sign the
/// image had been dropped. That is strictly worse than a refusal, and it is the failure this pin exists to prevent
/// recurring: a family may only appear below once its recipe pipeline genuinely consumes the conditioning.</para></summary>
public sealed class VideoFeatureDeclarationTests
{
    /// <summary>Families whose recipe pipeline reads <c>VideoRequest.InitImage</c>.</summary>
    private static readonly string[] ExpectedInitImage =
    [
        // WanVideoRecipe registers once per checkpoint variant, so it appears under several family ids.
        "wan", "wan-22-5b", "wan-21-14b", "wan-21-1_3b",
        "wan-vace", "wan-animate", "wan-animate-2", "wan-s2v",   // already wired before this phase
        "kandinsky5-video",                     // Phase 5b — EncodeFirstFrame now reached
        "minimax-h3",                           // keyframe conditioning
    ];

    /// <summary>End-frame conditioning is rarer: most i2v families generate forward from a start frame only.
    /// <para><c>wan-21-1_3b</c> is absent deliberately. It shares the identical non-concat code path with
    /// <c>wan-22-5b</c>, so the symmetric per-frame-timestep-pin mechanism should cover it — but no local 1.3B
    /// checkpoint exists to run and look at, and this backlog's rule is real-checkpoint verification, not "works by
    /// symmetry". <c>WanVideoRecipe.Supports</c> narrows it explicitly; see the remarks there.</para>
    /// <para><c>wan-21-14b</c> is absent because neither its T2V nor its concat-I2V checkpoints have been run with an
    /// end frame, and the generic <c>wan</c> slug because it carries no size: <c>WanVideoRecipe.SupportsFor</c> adds the
    /// bit back for a TI2V-5B file under it.</para></summary>
    private static readonly string[] ExpectedEndFrame =
    [
        "wan-22-5b",
        "minimax-h3",
    ];

    /// <summary>Every video family whose recipe calls <c>RecipeLoraMerge.Apply</c> before its transformer's
    /// <c>LoadWeights</c> — as of 2026-08-20 that is all of them.
    /// <para>Same reasoning as the image-side pin: declaring the bit without wiring the merge passes the feature gate,
    /// never merges, and yields a normal clip the user reads as a weak LoRA. No runtime check can catch it, because
    /// nothing ever asks.</para></summary>
    private static readonly string[] ExpectedLora =
    [
        "wan", "wan-22-5b", "wan-21-14b", "wan-21-1_3b",
        "wan-vace", "wan-animate", "wan-animate-2", "wan-s2v",
        "minimax-h3",
        // 2026-08-20 sweep. hunyuan-video, ltx-video-2 and lance-video declared NO conditioning at all before this
        // (they inherited IVideoRecipe's None); LoRA is the first bit each of them carries. LtxVideo2Recipe
        // registers under both its dev and distilled family ids, so it appears twice.
        "hunyuan-video", "ltx-video", "ltx-video-2", "ltx-2.5-distilled", "lance-video", "kandinsky5-video",
    ];

    private readonly ITestOutputHelper _output;

    public VideoFeatureDeclarationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void InitImageIsDeclaredByExactlyTheWiredVideoFamilies()
    {
        string[] actual = DeclaringFamilies(VideoFeatures.InitImage);
        _output.WriteLine($"init-image: {string.Join(", ", actual)}");
        Assert.Equal([.. ExpectedInitImage.Order(StringComparer.Ordinal)], actual);
    }

    [Fact]
    public void LoraIsDeclaredByExactlyTheVideoFamiliesThatMergeIt()
    {
        string[] actual = DeclaringFamilies(VideoFeatures.Lora);
        _output.WriteLine($"lora: {string.Join(", ", actual)}");
        Assert.Equal([.. ExpectedLora.Order(StringComparer.Ordinal)], actual);
    }

    [Fact]
    public void EndFrameIsDeclaredByExactlyTheFamiliesThatConsumeIt()
    {
        string[] actual = DeclaringFamilies(VideoFeatures.EndFrame);
        _output.WriteLine($"end-frame: {string.Join(", ", actual)}");
        Assert.Equal([.. ExpectedEndFrame.Order(StringComparer.Ordinal)], actual);
    }

    /// <summary>An end frame is generated *toward* from a start frame, so declaring it alone is incoherent.</summary>
    [Fact]
    public void NoFamilyDeclaresEndFrameWithoutInitImage()
    {
        foreach (string family in DeclaringFamilies(VideoFeatures.EndFrame))
        {
            Assert.True((Supports(family) & VideoFeatures.InitImage) != VideoFeatures.None,
                $"{family} declares EndFrame without InitImage; an end frame needs a start frame to generate toward.");
        }
    }

    private static string[] DeclaringFamilies(VideoFeatures feature) =>
        [.. VideoRecipeRegistry.RegisteredNames
            .Distinct(StringComparer.Ordinal)
            .Where(name => (Supports(name) & feature) != VideoFeatures.None)
            .Order(StringComparer.Ordinal)];

    private static VideoFeatures Supports(string family) =>
        VideoRecipeRegistry.Resolve(family)?.Supports ?? VideoFeatures.None;
}
