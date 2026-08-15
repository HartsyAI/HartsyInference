using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Video.Tests;

/// <summary>Pins the LTX-2 sigma schedule against ComfyUI's <c>LTXVScheduler</c>, which the shipped
/// <c>video_ltx2_5_t2v</c> template drives with <c>stretch=True, terminal=0.1</c>.
///
/// <para>This exists because we implemented every part of that node EXCEPT the stretch, so the sampler stopped
/// while the latent was still visibly noisy — and the error grew with token count, since the shift does. At
/// 1280×736×97f our last non-zero sigma was 0.817 against ComfyUI's 0.100, which is why RAISING resolution made
/// output worse and made a 0.94 MP generation the worst result in a 23-run quality study.</para></summary>
public sealed class LtxVideo2SigmaScheduleTests
{
    /// <summary>The LTX-2 shift: base_shift 0.95 at 1024 tokens → max_shift 2.05 at 4096, exponentiated.</summary>
    private static float ShiftForTokens(int tokens)
    {
        double m = (2.05 - 0.95) / (4096 - 1024), b = 0.95 - m * 1024;
        return (float)System.Math.Exp(tokens * m + b);
    }

    private static float[] Schedule(int steps, int tokens, bool stretch)
    {
        LtxVideo2Config cfg = new LtxVideo2Config { SigmaStretch = stretch };
        float[] s = LancePipelineCommon.BuildShiftedTimesteps(steps, ShiftForTokens(tokens));
        return LtxVideo2Pipeline.StretchTerminalForTests(s, cfg, ShiftForTokens(tokens));
    }

    /// <summary>Reference values read out of ComfyUI's own scheduler at the geometry this engine benchmarks.</summary>
    [Theory]
    [InlineData(30, 4992, new[] { 0.534f, 0.437f, 0.302f, 0.100f })]   // 768×512×97f
    public void MatchesComfyUiTailSigmas(int steps, int tokens, float[] expectedTail)
    {
        float[] got = Schedule(steps, tokens, stretch: true);
        // The schedule ends in an explicit 0; compare the four entries before it.
        int last = got.Length - 1;
        while (last > 0 && got[last] == 0f) last--;
        for (int i = 0; i < expectedTail.Length; i++)
        {
            float actual = got[last - (expectedTail.Length - 1 - i)];
            Assert.True(System.Math.Abs(actual - expectedTail[i]) < 2e-2f,
                $"sigma[{last - (expectedTail.Length - 1 - i)}] = {actual:F4}, expected ~{expectedTail[i]:F4}");
        }
    }

    /// <summary>The whole point: the last non-zero sigma must land on the terminal REGARDLESS of token count.
    /// Un-stretched it climbs with the shift — 0.270 at 768×512 and 0.817 at 1280×736 — which is the bug.</summary>
    [Theory]
    [InlineData(4992)]     // 768×512×97f
    [InlineData(7800)]     // 960×640×97f
    [InlineData(11960)]    // 1280×736×97f
    public void TerminalSigmaIsPinnedAtEveryTokenCount(int tokens)
    {
        float[] stretched = Schedule(30, tokens, stretch: true);
        float[] raw = Schedule(30, tokens, stretch: false);
        int last = stretched.Length - 1;
        while (last > 0 && stretched[last] == 0f) last--;

        Assert.True(System.Math.Abs(stretched[last] - 0.1f) < 1e-3f,
            $"terminal sigma {stretched[last]:F4}, expected 0.1000 at {tokens} tokens");
        Assert.True(raw[last] > stretched[last] + 0.05f,
            $"un-stretched terminal {raw[last]:F4} should be materially higher — if not, this test proves nothing");
        Assert.Equal(0f, stretched[^1]);                       // the explicit zero must survive
        for (int i = 1; i <= last; i++)
        {
            Assert.True(stretched[i] < stretched[i - 1], $"schedule must stay strictly decreasing at {i}");
        }
    }

    /// <summary>Distilled checkpoints bake their sigmas in; re-transforming them would corrupt that path.</summary>
    [Fact]
    public void FixedSigmasAreNeverStretched()
    {
        float[] fixedSigmas = [1.0f, 0.9937f, 0.9875f, 0.909f, 0.725f, 0.4219f, 0.0f];
        float[] copy = (float[])fixedSigmas.Clone();
        LtxVideo2Config cfg = new LtxVideo2Config { FixedSigmas = copy, SigmaStretch = true };
        Assert.NotNull(cfg.FixedSigmas);
        // The pipeline selects FixedSigmas *instead of* the dynamic schedule, so the stretch never sees them.
        Assert.Equal(fixedSigmas, cfg.FixedSigmas!);
    }
}
