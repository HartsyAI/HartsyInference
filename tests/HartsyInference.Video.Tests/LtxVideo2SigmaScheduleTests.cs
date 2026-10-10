using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Video.Tests;

/// <summary>Pins the LTX-2 sigma schedule against ComfyUI's <c>LTXVScheduler</c>, which the shipped
/// <c>video_ltx2_5_t2v</c> template drives with <c>stretch=True, terminal=0.1</c>.
///
/// <para>This exists because we implemented every part of that node EXCEPT the stretch, so the sampler stopped
/// while the latent was still visibly noisy — and the error grew with token count, since the shift does.</para></summary>
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
    /// Un-stretched it climbs with the shift — 0.817 at 1280×736 — which is the bug.</summary>
    [Theory]
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

    /// <summary>Default-OFF is the whole gating contract: unset, and at any cap the geometry does not reach, the
    /// engine must produce the exact float it produced before this knob existed.</summary>
    [Theory]
    [InlineData(4992, 0)]
    [InlineData(4992, 8000)]      // cap above tokens
    [InlineData(27280, -1)]       // a negative cap is not a cap
    public void CapIsInertUnlessItBites(int tokens, int cap)
    {
        Assert.Equal(ShiftForTokens(tokens), LtxVideo2Pipeline.ComputeShift(tokens, new LtxVideo2Config { ShiftMaxTokens = cap }));
    }

    /// <summary>The direct override replaces the fit outright, at any geometry, and outranks a token cap. It exists
    /// because the token cap bottoms out at exp(0.583) = 1.79 and cannot reach shift 1, which is what SwarmUI's
    /// ComfyUI backend runs for LTX-2.</summary>
    [Theory]
    [InlineData(27280, 1.0f)]
    public void DirectOverrideReplacesTheFit(int tokens, float shift)
    {
        Assert.Equal(shift, LtxVideo2Pipeline.ComputeShift(tokens, new LtxVideo2Config { ShiftOverride = shift }));
        Assert.Equal(shift, LtxVideo2Pipeline.ComputeShift(tokens,
            new LtxVideo2Config { ShiftOverride = shift, ShiftMaxTokens = 4096 }));
    }

    /// <summary>What the cap is actually for: at the user's geometry the post-stretch schedule spends 33 of its 40
    /// steps above σ 0.9 — an effectively 4-step denoise — and the cap restores a spread.</summary>
    [Fact]
    public void CapRestoresSigmaSpreadAtTheFailingGeometry()
    {
        LtxVideo2Config capped = new LtxVideo2Config { ShiftMaxTokens = 4096 };
        float[] raw = LtxVideo2Pipeline.StretchTerminalForTests(
            LancePipelineCommon.BuildShiftedTimesteps(40, ShiftForTokens(27280)), capped, 1f);
        float[] fixedUp = LtxVideo2Pipeline.StretchTerminalForTests(
            LancePipelineCommon.BuildShiftedTimesteps(40, LtxVideo2Pipeline.ComputeShift(27280, capped)), capped, 1f);
        int rawStuck = System.Linq.Enumerable.Count(raw[..^1], s => s > 0.9f);
        int cappedStuck = System.Linq.Enumerable.Count(fixedUp[..^1], s => s > 0.9f);
        Assert.Equal(33, rawStuck);
        Assert.True(cappedStuck <= 20, $"capped schedule still stalls for {cappedStuck} of 40 steps");
    }
}
