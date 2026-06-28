using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.F5Tts;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>F5-TTS smoke tests. The component-level tests exercise config + scheduler
/// math. The integration smoke test loads the actual model and runs a single forward
/// pass on synthetic input — verifies the architecture assembles, the weights load with
/// the expected key names, the depthwise / grouped Conv1D paths run, the RoPE applies
/// without going off the edge of the precomputed table, and that the output has the
/// expected shape.
///
/// <para>The DiT velocity and the full CFM sample loop are numerically validated against upstream F5-TTS
/// (both corr 1.0) — see PARITY_VERIFICATION. The gated real-weight test below guards the loaded forward.</para></summary>
public sealed class F5TtsSmokeTests
{
    [Fact]
    public void V1Base_ConfigMatchesPublishedShapes()
    {
        F5TtsConfig c = F5TtsConfig.V1Base;
        Assert.Equal(1024, c.Dim);
        Assert.Equal(22, c.Depth);
        Assert.Equal(16, c.Heads);
        Assert.Equal(64, c.HeadDim);
        Assert.Equal(2, c.FfMult);
        Assert.Equal(2048, c.FfInner);
        Assert.Equal(100, c.MelDim);
        Assert.Equal(512, c.TextDim);
        Assert.Equal(4, c.TextConvLayers);
        Assert.Equal(2_545, c.TextNumEmbeds);
        Assert.Equal(256, c.TimeFreqEmbedDim);
        Assert.Equal(31, c.ConvPosKernel);
        Assert.Equal(16, c.ConvPosGroups);
    }

    [Fact]
    public void SwaySampling_DefaultRecreatesUpstreamMath()
    {
        // Reference values from a quick Python check:
        //   u = linspace(0, 1, 33)
        //   u_swayed = u + (-1.0) * (cos(pi/2 * u) - 1 + u)
        //   = u - (cos(pi/2 * u) - 1 + u)
        //   = 1 - cos(pi/2 * u)
        // So at u=0, swayed=0; at u=0.5, swayed=1-cos(pi/4)≈0.293; at u=1, swayed=1.
        F5SwaySamplingScheduler sched = new(steps: 32, swayCoef: -1.0f);
        Assert.Equal(32, sched.Steps);
        Assert.Equal(33, sched.Timesteps.Length);
        Assert.Equal(0f, sched.Timesteps[0], precision: 5);
        Assert.Equal(1f, sched.Timesteps[^1], precision: 5);

        // Each delta is positive (timesteps strictly increasing).
        for (int i = 0; i < sched.Steps; i++)
            Assert.True(sched.Deltas[i] > 0f, $"delta {i} should be positive, got {sched.Deltas[i]}");

        // Sway shifts mass toward t=0: the middle timestep at u=0.5 should be less than 0.5.
        Assert.True(sched.Timesteps[16] < 0.5f, $"swayed midpoint should be < 0.5, got {sched.Timesteps[16]}");
    }

    [Fact]
    public void SwaySampling_ZeroCoef_IsIdentity()
    {
        F5SwaySamplingScheduler sched = new(steps: 8, swayCoef: 0f);
        for (int i = 0; i < sched.Timesteps.Length; i++)
            Assert.Equal((float)i / 8, sched.Timesteps[i], precision: 5);
    }

    /// <summary>Real-weight DiT forward on the F5TTS_v1_Base checkpoint. Gated on <c>F5_DIT</c> (path to
    /// <c>model_1250000.safetensors</c>; falls back to the model cache). The full numerical parity (DiT velocity
    /// corr 1.0 and the CFM sample loop corr 1.0 vs upstream) is recorded in PARITY_VERIFICATION; this guards the
    /// loaded forward against regressions — asserting finite output of the right shape and a non-degenerate
    /// velocity magnitude (the reference velocity has std ≈ 1.5).</summary>
    [Trait("Category", "Integration")]
    [Fact]
    public unsafe void F5Dit_RealWeights_VelocityIsFiniteAndSane()
    {
        string? path = Environment.GetEnvironmentVariable("F5_DIT");
        if (string.IsNullOrEmpty(path))
        {
            string repoDir = AudioModelCache.GetRepoDirectory("SWivid/F5-TTS");
            path = Path.Combine(repoDir, "F5TTS_v1_Base", "model_1250000.safetensors");
        }
        if (!File.Exists(path)) return;

        F5TtsConfig cfg = F5TtsConfig.V1Base;
        using ModelHandler.SafeTensors.SafeTensorsLoader ld = new();
        ld.Load(path);
        using F5Dit dit = new(cfg);
        dit.LoadWeights(ld.GetAllTensors());
        using CpuBackend backend = new();

        int t = 80;
        using Tensor noisy = new(new TensorShape(1, cfg.MelDim, t), DType.F32);
        using Tensor cond = new(new TensorShape(1, cfg.MelDim, t), DType.F32);
        float* np_ = (float*)noisy.DataPointer; float* cp = (float*)cond.DataPointer;
        for (int i = 0; i < cfg.MelDim * t; i++) { np_[i] = MathF.Sin(i * 0.013f); cp[i] = MathF.Cos(i * 0.017f) * 0.5f; }
        int[] text = new int[30];
        for (int i = 0; i < text.Length; i++) text[i] = (i * 37 + 11) % 2545;

        using Tensor v = dit.Forward(backend, noisy, cond, text, timestep: 0.3f);
        Assert.Equal(new TensorShape(1, t, cfg.MelDim), v.Shape);
        float* vp = (float*)v.DataPointer;
        double sumSq = 0;
        for (long i = 0; i < v.ElementCount; i++) { Assert.True(float.IsFinite(vp[i])); sumSq += (double)vp[i] * vp[i]; }
        double std = Math.Sqrt(sumSq / v.ElementCount);
        Assert.InRange(std, 0.3, 5.0);   // reference velocity std ≈ 1.5; guards against a silent regression
    }
}
