using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight check of the ControlFoley generator. The released <c>controlfoley.pth</c> is 11.3 GB of float32,
/// so <c>tools/controlfoley/network_reference.py real</c> keeps the real large_44k dimensions and weights but only joint
/// blocks 0, 1 and 17 (the pre-only one) and fused blocks 0 and 1. <c>HARTSY_CONTROLFOLEY_DIR</c> names the folder with
/// its <c>controlfoley_trunc.safetensors</c> and <c>controlfoley_real_ref.safetensors</c>; without it the test does nothing.</summary>
[Trait("Category", "RealWeights")]
public sealed unsafe class ControlFoleyNetworkRealWeightTests(ITestOutputHelper output)
{
    private static string? Dir => Environment.GetEnvironmentVariable("HARTSY_CONTROLFOLEY_DIR");

    private static (float MaxAbs, float MaxRef) Diff(Tensor expected, Tensor actual)
    {
        Assert.Equal(expected.ElementCount, actual.ElementCount);
        float* e = (float*)expected.DataPointer, a = (float*)actual.DataPointer;
        float worst = 0f, scale = 0f;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(e[i] - a[i]));
            scale = MathF.Max(scale, MathF.Abs(e[i]));
        }

        return (worst, scale);
    }

    [Fact]
    public void TruncatedRealNetwork_MatchesOfficialVelocityAndTrajectory()
    {
        if (Dir is not { Length: > 0 } dir)
        {
            return;
        }

        using SafeTensorsLoader weights = new(), reference = new();
        weights.Load(Path.Combine(dir, "controlfoley_trunc.safetensors"));
        reference.Load(Path.Combine(dir, "controlfoley_real_ref.safetensors"));
        Dictionary<string, Tensor> w = weights.Descriptors.Keys.ToDictionary(k => k, weights.GetTensor);
        Dictionary<string, Tensor> r = reference.Descriptors.Keys.ToDictionary(k => k, reference.GetTensor);

        ControlFoleyNetworkConfig cfg = ControlFoleyNetworkConfig.Large44k with { Depth = 5, FusedDepth = 2 };
        using IBackend backend = new CpuBackend();
        using ControlFoleyNetwork net = new(cfg);
        net.LoadWeights(w);
        using ControlFoleyConditions cond = net.PreprocessConditions(backend, r["cond.clip_f"], r["cond.visual_f"], r["cond.sync_f"],
            r["cond.text_f"], r["cond.audio_f"], r["cond.timbre_f"]);
        using ControlFoleyConditions empty = net.GetEmptyConditions(backend, 1);

        (float d, float m) = Diff(r["ref.pre.clip_f"], cond.ClipF);
        output.WriteLine($"preprocess clip_f max|Δ|={d:E2} max|ref|={m:F3}");
        Assert.True(d <= 1e-3f * Math.Max(1f, m));

        using Tensor flow = net.PredictFlow(backend, r["cond.latent"], [0.5f], cond, out Tensor mm);
        mm.Dispose();
        (d, m) = Diff(r["ref.flow.t1"], flow);
        output.WriteLine($"velocity t=0.5 max|Δ|={d:E2} max|ref|={m:F3}");
        Assert.True(d <= 1e-3f * Math.Max(1f, m));

        using Tensor final = ControlFoleySampler.Sample(backend, net, cond, empty, r["cond.noise"], 5, 4.5f);
        (d, m) = Diff(r["ref.traj.final"], final);
        output.WriteLine($"5-step trajectory max|Δ|={d:E2} max|ref|={m:F3}");
        Assert.True(d <= 1e-3f * Math.Max(1f, m));
    }
}
