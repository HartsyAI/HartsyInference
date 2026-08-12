using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Calibrates <see cref="MiniMaxH3ActivationEstimate.EstimateFloorBytes"/> against real-CUDA runs of the
/// actual fp8 checkpoint on this box (RTX 4090, 24 GiB), captured directly from CLI output rather than derived by
/// hand. Two earlier versions of this estimator each passed a similar arithmetic check and were still wrong in
/// opposite directions: a 4x q/k/v/attn-output form false-refused a geometry already proven to complete, and its
/// replacement UNDER-counted <see cref="MiniMaxH3Transformer.AttentionChunked"/>'s pass-1 peak badly enough that a
/// 141-frame sharded request cleared pre-flight and then OOMed mid-forward on the 12 GiB card. Both are why these
/// numbers are measured rather than reasoned about; see the class remarks on
/// <see cref="MiniMaxH3ActivationEstimate"/> for the shape the floor actually takes.
/// <para><b>The three runs below predate the pass-1 projection split</b> (k+v projected in pass 1, q re-projected
/// per chunk in pass 2), which removed a full <c>seq*inner*F32</c> from the pass-1 peak. Their measured
/// available-VRAM window is still the calibration anchor — that is a property of the box, not of the estimator —
/// but the floors quoted in this list are the OLD 3x ones and will not reproduce. The post-split boundary measured
/// on 2026-08-12 is <see cref="SplitEnabledSeq"/>.</para>
/// Three CLI runs, same session,
/// resident DiT ~20,959,250,528 bytes, free VRAM 24.0-24.03 GB (a small natural drift between runs):
/// <list type="bullet">
/// <item>39f@960x960 (seq 11,442 by this class's own conservative formula) — <c>floor=2,926,612,480</c>,
/// <c>available=3,064,084,384</c> — did NOT refuse, and the generation completed both denoise steps for real.</item>
/// <item>90f@960x960 (seq 25,112) — <c>floor=4,298,424,320</c>, <c>available=3,072,604,064</c> — refused.</item>
/// <item>481f@1280x704, the incident that started this overhaul (seq 127,076 by the engine's own internal
/// layout — this class's conservative formula independently estimates 126,864) — <c>floor=14,530,715,648</c>,
/// <c>available=3,072,604,064</c> — refused by more than 11 GB of margin.</item>
/// </list></summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class MiniMaxH3ActivationEstimateTests
{
    /// <summary>39 frames @ 960x960: <c>VideoLatentFrames(39)=12</c>, latentH=latentW=60, frameRows=900 -&gt;
    /// 10,800 video rows + 130 audio rows + this class's 512-row conservative text/reference allowance.</summary>
    private const int KnownGoodSeq = 11_442;

    /// <summary>90 frames @ 960x960: <c>VideoLatentFrames(90)=27</c>, frameRows=900 -&gt; 24,300 video rows + 300
    /// audio rows + the same 512-row allowance.</summary>
    private const int KnownOomSeq = 25_112;

    /// <summary>481 frames @ 1280x704: <c>VideoLatentFrames(481)=142</c>, latentH=44, latentW=80, frameRows=880
    /// -&gt; 124,960 video rows + 1,604 audio rows + the allowance.</summary>
    private const int IncidentSeq = 126_864;

    /// <summary>Bytes actually free before the DiT loads, measured by the 39f/90f CLI runs above.</summary>
    private const long MeasuredFreeBytes = 3_072_604_064L + 20_959_250_528L;

    /// <summary>The fp8 DiT's real resident footprint, measured by <c>EstimateResidentWeightBytes()</c> in the
    /// same runs — not the ~19.5 GiB checkpoint-file-size heuristic used elsewhere for the resident/streamed
    /// decision, which undercounts the F32-promoted norm/rope tensors.</summary>
    private const long MeasuredResidentWeightBytes = 20_959_250_528L;

    private const long MeasuredAvailableForActivations = MeasuredFreeBytes - MeasuredResidentWeightBytes;

    [Fact]
    public void EstimateFloorBytes_GrowsMonotonicallyWithSequenceLength()
    {
        MiniMaxH3Config config = new MiniMaxH3Config();
        long floorGood = MiniMaxH3ActivationEstimate.EstimateFloorBytes(KnownGoodSeq, config, DType.F32);
        long floorOom = MiniMaxH3ActivationEstimate.EstimateFloorBytes(KnownOomSeq, config, DType.F32);
        long floorIncident = MiniMaxH3ActivationEstimate.EstimateFloorBytes(IncidentSeq, config, DType.F32);

        Assert.True(floorGood < floorOom, $"{floorGood} should be < {floorOom}");
        Assert.True(floorOom < floorIncident, $"{floorOom} should be < {floorIncident}");
    }

    [Fact]
    public void EstimateFloorBytes_MatchesTheMeasuredRealCudaFeasibilityBoundary()
    {
        MiniMaxH3Config config = new MiniMaxH3Config();
        long floorGood = MiniMaxH3ActivationEstimate.EstimateFloorBytes(KnownGoodSeq, config, DType.F32);
        long floorOom = MiniMaxH3ActivationEstimate.EstimateFloorBytes(KnownOomSeq, config, DType.F32);

        Assert.True(floorGood < MeasuredAvailableForActivations,
            $"the known-good geometry's floor ({floorGood} bytes) should fit the measured "
            + $"{MeasuredAvailableForActivations}-byte window it actually completed in");
        Assert.True(floorOom > MeasuredAvailableForActivations,
            $"the known-OOM geometry's floor ({floorOom} bytes) should exceed the measured "
            + $"{MeasuredAvailableForActivations}-byte window it actually failed in");
    }

    /// <summary>56 frames @ 768x768 by this class's own conservative formula. Measured on the 4090 on 2026-08-12
    /// with 19,988 MB of resident DiT and 22,634 MB free, leaving 2,646 MB for activations: BEFORE the projection
    /// split the engine refused it at a 2,771 MB floor (125 MB short); AFTER the split the same geometry ran two
    /// denoise steps and wrote all 57 outputs. So the post-split floor for this length must fit that window, and
    /// the pre-split 3x form must not.</summary>
    private const int SplitEnabledSeq = 10_490;

    private const long MeasuredAvailableBytes20260812 = 2_646L * 1024 * 1024;

    [Fact]
    public void EstimateFloorBytes_PostSplit_FitsTheGeometryThatUsedToRefuse()
    {
        MiniMaxH3Config config = new MiniMaxH3Config();
        long floor = MiniMaxH3ActivationEstimate.EstimateFloorBytes(SplitEnabledSeq, config, DType.F32);

        Assert.True(floor <= MeasuredAvailableBytes20260812,
            $"56f@768x768's floor ({floor} bytes) must fit the {MeasuredAvailableBytes20260812}-byte window it "
            + "actually completed a real generation in after the projection split");

        // The pass-1 term the split removed is one full seq*inner*F32; without it this geometry refused for real.
        long preSplitFloor = floor + (long)SplitEnabledSeq * config.NumAttentionHeads * config.AttentionHeadDim * DType.F32.SizeInBytes;
        Assert.True(preSplitFloor > MeasuredAvailableBytes20260812,
            $"the pre-split floor ({preSplitFloor} bytes) must still exceed that window — it is what refused by 125 MB");
    }

    [Fact]
    public void EstimateFloorBytes_IncidentGeometry_FarExceedsTheMeasuredWindow()
    {
        MiniMaxH3Config config = new MiniMaxH3Config();
        long floorIncident = MiniMaxH3ActivationEstimate.EstimateFloorBytes(IncidentSeq, config, DType.F32);

        Assert.True(floorIncident > MeasuredAvailableForActivations,
            "the geometry that actually OOM'd yesterday must still refuse under the corrected floor estimate");
    }
}
