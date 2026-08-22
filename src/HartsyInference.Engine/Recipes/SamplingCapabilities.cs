using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Engine.Recipes;

/// <summary>Which sampler and sigma-schedule names each model family actually accepts, so a host can offer the right
/// choices (and refuse the wrong ones) BEFORE a generation starts.
///
/// <para>This exists because sampler support is not a composition feature: it cannot ride
/// <see cref="ImageFeatures"/>/<see cref="VideoFeatures"/>, whose bits describe what a request may CONTAIN, whereas
/// this describes which VALUES one field may take. Without it a host's only way to discover support is to run a
/// generation and catch the pipeline's <see cref="NotSupportedException"/> — after the checkpoint has loaded and the
/// text encoder has run. SwarmUI in particular needs the answer up front, because a selection this engine cannot serve
/// should route the request to a backend that can, rather than fail it.</para>
///
/// <para><b>Empty <see cref="SamplingSupport.Samplers"/> means the family takes no selection at all</b> — it samples
/// with its own solver and any named sampler is refused. That is not the same as "unknown family", which
/// <see cref="ForImage"/> reports by returning <see cref="Unknown"/>.</para>
///
/// <para><b>Known limit:</b> three families accept a sampler on their text-to-image path but not on a secondary one
/// (Boogu's reference-edit loop, OmniGen 2's edit loop, Flux.1's non-drain-free path — masked inpaint, ControlNet,
/// Kontext, regional). Those refuse from inside the pipeline, so a host must still translate a
/// <see cref="NotSupportedException"/> into a user-facing error. Separately, no flow-match family can combine a
/// re-spaced sigma schedule with img2img or inpaint: the init latent is noised at the family's own sigma, so the
/// schedule would start the sampler from a noise level the latent does not carry.</para></summary>
public static class SamplingCapabilities
{
    /// <summary>The sampler and sigma-schedule names one family accepts. Both lists empty means no selection is
    /// honored; the family runs its own solver over its own spacing.</summary>
    /// <param name="Samplers">Accepted sampler names, or empty when the family takes no sampler selection.</param>
    /// <param name="Schedules">Accepted sigma-schedule names, or empty when the family owns its own spacing.</param>
    public sealed record SamplingSupport(IReadOnlyList<string> Samplers, IReadOnlyList<string> Schedules);

    /// <summary>The answer for a family this engine does not recognize: nothing is claimed.</summary>
    public static SamplingSupport Unknown { get; } = new([], []);

    /// <summary>Families carrying the sampler seam: every implemented sampler over every implemented schedule.</summary>
    private static readonly SamplingSupport FullSeam = new(SamplerRegistry.Names, SigmaSchedule.Names);

    /// <summary>The SD family: the seam, plus the legacy scheduler names whose own update rules it still runs.</summary>
    private static readonly SamplingSupport SeamPlusLegacy = new(
        [.. SamplerRegistry.Names.Concat(SchedulerFactory.Names).Distinct(StringComparer.Ordinal)],
        SigmaSchedule.Names);

    /// <summary>A family that samples with its own solver and refuses every named sampler.</summary>
    private static readonly SamplingSupport SolverOwned = new([], []);

    private static readonly Dictionary<string, SamplingSupport> Image = new(StringComparer.OrdinalIgnoreCase)
    {
        ["anima"] = FullSeam,
        ["auraflow"] = FullSeam,
        ["boogu"] = FullSeam,
        ["chroma"] = FullSeam,
        ["chroma-radiance"] = FullSeam,
        ["ernie-image"] = FullSeam,
        ["f-lite"] = FullSeam,
        ["flux1"] = FullSeam,
        ["flux2"] = FullSeam,
        ["hidream"] = FullSeam,
        ["hunyuan-image"] = FullSeam,
        ["kandinsky5"] = FullSeam,
        ["krea2"] = FullSeam,
        ["lens"] = FullSeam,
        ["mage-flow"] = FullSeam,
        ["omnigen2"] = FullSeam,
        ["qwen-image"] = FullSeam,
        ["sd3"] = FullSeam,
        ["zeta-chroma"] = FullSeam,
        ["zimage"] = FullSeam,
        ["sd15"] = SeamPlusLegacy,
        ["sdxl"] = SeamPlusLegacy,
        // The refiner never gained the seam: it runs the legacy scheduler path, which owns its own spacing, so the
        // sampler names resolve but a sigma schedule has nothing to re-space.
        ["sdxl-refiner"] = new(SchedulerFactory.Names, []),
        // Fused or bespoke update rules the pair contract cannot express — see each pipeline's refusal for why.
        ["ideogram4"] = SolverOwned,
        ["lumina2"] = SolverOwned,
        ["lance-image"] = SolverOwned,
    };

    private static readonly Dictionary<string, SamplingSupport> Video = new(StringComparer.OrdinalIgnoreCase)
    {
        ["hunyuan-video"] = FullSeam,
        ["kandinsky5-video"] = FullSeam,
        ["lance-video"] = FullSeam,
        // Upstream's own solver names, not the seam's: these two pick between ported solvers rather than driving one.
        ["wan-animate"] = new(["unipc"], []),
        ["wan-animate-2"] = new(["dpm++2m", "unipc"], []),
        ["wan"] = SolverOwned,
        ["wan-22-5b"] = SolverOwned,
        ["wan-21-1_3b"] = SolverOwned,
        ["wan-21-14b"] = SolverOwned,
        ["wan-vace"] = SolverOwned,
        ["wan-s2v"] = SolverOwned,
        ["ltx-video"] = SolverOwned,
        ["ltx-video-2"] = SolverOwned,
        ["ltx-2.5-distilled"] = SolverOwned,
        ["minimax-h3"] = SolverOwned,
    };

    /// <summary>What the image family <paramref name="familyId"/> accepts, or <see cref="Unknown"/> if unrecognized.</summary>
    public static SamplingSupport ForImage(string familyId)
        => familyId is not null && Image.TryGetValue(familyId, out SamplingSupport? support) ? support : Unknown;

    /// <summary>What the video family <paramref name="familyId"/> accepts, or <see cref="Unknown"/> if unrecognized.</summary>
    public static SamplingSupport ForVideo(string familyId)
        => familyId is not null && Video.TryGetValue(familyId, out SamplingSupport? support) ? support : Unknown;

    /// <summary>Whether <paramref name="familyId"/> (image or video) accepts any sampler selection at all.</summary>
    public static bool AcceptsSelection(string familyId)
        => ForImage(familyId).Samplers.Count > 0 || ForVideo(familyId).Samplers.Count > 0;
}
