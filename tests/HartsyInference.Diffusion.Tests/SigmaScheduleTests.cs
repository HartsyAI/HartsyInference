using HartsyInference.Diffusion.Sampling;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Structural gates for the sigma-schedule half of sampling.
///
/// <para>These check the invariants a sampler relies on rather than reproducing reference arrays: a schedule that
/// silently returns a non-descending array, loses the terminal zero, or drifts off the family's own sigma range does
/// not throw — it produces a subtly wrong image, which is the failure mode the whole named-refusal design exists to
/// avoid. Reference-value parity against ComfyUI's <c>comfy/samplers.py</c> is a separate, later pass.</para></summary>
public sealed class SigmaScheduleTests
{
    /// <summary>A plausible SDXL-shaped base schedule: descending, terminal zero.</summary>
    private static float[] BaseSigmas(int steps)
    {
        float[] sigmas = new float[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            sigmas[i] = 14.6f * (1f - ((float)i / steps)) + 0.03f;
        }
        sigmas[steps] = 0f;
        return sigmas;
    }

    /// <summary>Every schedule must return <c>steps + 1</c> strictly-descending values ending at exactly zero. The
    /// terminal zero is what the Euler update's last <c>dt</c> is measured against; losing it leaves the final step
    /// short and the image visibly noisy — the same defect the LTX-2.5 sigma-stretch bug was.</summary>
    [Theory]
    [InlineData("normal")]
    [InlineData("karras")]
    [InlineData("exponential")]
    [InlineData("sgm_uniform")]
    [InlineData("simple")]
    [InlineData("ddim_uniform")]
    [InlineData("beta")]
    [InlineData("kl_optimal")]
    [InlineData("linear_quadratic")]
    public void Schedule_IsDescendingAndTerminatesAtZero(string name)
    {
        const int Steps = 20;
        float[] sigmas = SigmaSchedule.Apply(name, BaseSigmas(Steps));

        Assert.Equal(Steps + 1, sigmas.Length);
        Assert.Equal(0f, sigmas[Steps]);
        for (int i = 0; i < Steps; i++)
        {
            Assert.True(float.IsFinite(sigmas[i]), $"{name}[{i}] = {sigmas[i]} is not finite.");
            Assert.True(sigmas[i] > 0f, $"{name}[{i}] = {sigmas[i]} must be positive before the terminal zero.");
            Assert.True(sigmas[i] > sigmas[i + 1],
                $"{name} is not strictly descending at {i}: {sigmas[i]} then {sigmas[i + 1]}.");
        }
    }

    /// <summary>A schedule re-spaces the family's own range; it must not invent a wider one. The first sigma is the
    /// family's sigma_max, and nothing may exceed it — a schedule starting ABOVE the range would ask the model to
    /// denoise a noise level it was never trained on.</summary>
    [Theory]
    [InlineData("karras")]
    [InlineData("exponential")]
    [InlineData("sgm_uniform")]
    [InlineData("simple")]
    [InlineData("beta")]
    [InlineData("kl_optimal")]
    public void Schedule_StaysWithinTheFamilysOwnRange(string name)
    {
        float[] baseSigmas = BaseSigmas(20);
        float[] sigmas = SigmaSchedule.Apply(name, baseSigmas);
        float max = baseSigmas[0];

        foreach (float sigma in sigmas)
        {
            Assert.True(sigma <= max + 1e-4f, $"{name} produced {sigma}, above the family's sigma_max {max}.");
        }
    }

    /// <summary><c>normal</c> is the identity — it must hand back the family's own array untouched, so a request that
    /// names no schedule is provably the pre-existing behaviour.</summary>
    [Fact]
    public void NormalSchedule_IsTheIdentity()
    {
        float[] baseSigmas = BaseSigmas(15);
        Assert.Same(baseSigmas, SigmaSchedule.Apply("normal", baseSigmas));
        Assert.Same(baseSigmas, SigmaSchedule.Apply(null, baseSigmas));
        Assert.Same(baseSigmas, SigmaSchedule.Apply("", baseSigmas));
    }

    /// <summary>Karras spacing must front-load: with rho = 7 the steps cluster toward low sigma, so the schedule
    /// spends more of its budget near the end. Checking the direction of the skew catches an inverted rho.</summary>
    [Fact]
    public void KarrasSpacing_ClustersTowardLowSigma()
    {
        float[] karras = SigmaSchedule.Apply("karras", BaseSigmas(20));
        float firstHalfDrop = karras[0] - karras[10];
        float secondHalfDrop = karras[10] - karras[20];
        Assert.True(firstHalfDrop > secondHalfDrop,
            $"Karras should drop faster early (got {firstHalfDrop} then {secondHalfDrop}); rho may be inverted.");
    }

    /// <summary>An unknown schedule throws and names the alternatives, rather than silently becoming `normal`.</summary>
    [Fact]
    public void UnknownSchedule_ThrowsAndListsWhatIsAvailable()
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(
            () => SigmaSchedule.Apply("not_a_schedule", BaseSigmas(10)));
        Assert.Contains("karras", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>Compound ComfyUI names split on the LONGEST schedule suffix. <c>ddim_uniform</c> and <c>sgm_uniform</c>
    /// overlap, so a shortest-match rule would cut <c>dpmpp_2m_ddim_uniform</c> in the wrong place and then reject the
    /// leftover as an unknown sampler.</summary>
    [Theory]
    [InlineData("dpmpp_2m_sde_karras", "dpmpp_2m_sde", "karras")]
    [InlineData("euler_ancestral", "euler_ancestral", null)]
    [InlineData("dpmpp_2m", "dpmpp_2m", null)]
    [InlineData("heun_exponential", "heun", "exponential")]
    [InlineData("dpmpp_2m_ddim_uniform", "dpmpp_2m", "ddim_uniform")]
    [InlineData("lms_beta", "lms", "beta")]
    public void CompoundName_SplitsIntoSamplerAndSchedule(string input, string sampler, string? schedule)
    {
        (string actualSampler, string? actualSchedule) = SamplerRegistry.SplitCompound(input);
        Assert.Equal(sampler, actualSampler);
        Assert.Equal(schedule, actualSchedule);
    }

    /// <summary>The refusal distinguishes "recognized ComfyUI sampler, not built yet" from "typo" — different answers
    /// for whoever reads it, so the message must not flatten them together.</summary>
    [Fact]
    public void NotYetImplementedSampler_RefusesDifferentlyFromATypo()
    {
        float[] sigmas = SigmaSchedule.Apply("normal", BaseSigmas(10));

        NotSupportedException known = Assert.Throws<NotSupportedException>(
            () => SamplerRegistry.Create("uni_pc", sigmas, 0));
        Assert.Contains("not implemented yet", known.Message, StringComparison.Ordinal);

        NotSupportedException typo = Assert.Throws<NotSupportedException>(
            () => SamplerRegistry.Create("eulr", sigmas, 0));
        Assert.Contains("Unknown sampler", typo.Message, StringComparison.Ordinal);
    }
}
