using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Regression guard for the ACE-Step v1.5 DiT GPU-residency rewrite. Runs <see cref="AceStep15Dit.Forward"/>
/// on the tiny synthetic checkpoint with fixed inputs (CPU, no real weights) and dumps the velocity to
/// <c>HARTSY_ACE15_DUMP</c>. Capture a golden from the pre-rewrite host path, then diff the post-rewrite device-op
/// path against it — the F32 math is identical, so maxAbs must stay ~0. Also runs a self-consistency check
/// (two forwards on the same inputs are bit-identical) so the file is a normal green test when no dump path is set.</summary>
public unsafe class AceStep15ResidencyParityTests
{
    private readonly ITestOutputHelper _output;
    public AceStep15ResidencyParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Dit_Forward_Synthetic_IsDeterministic_AndDumpsVelocity()
    {
        const int T = 12;   // latent frames (even; s = 6 > sliding window+1 = 4 so sliding layers get a mask)
        const int L = 5;    // condition tokens
        const float Sigma = 0.7f;

        AceStep15Config cfg = AceStep15SyntheticWeights.TinyConfig;
        int dim = cfg.HiddenSize;
        int ctxCh = cfg.InChannels - cfg.LatentChannels;

        System.Collections.Generic.Dictionary<string, Tensor> w = AceStep15SyntheticWeights.BuildModel(cfg);
        int encDim = (int)w["decoder.condition_embedder.weight"].Shape[1];

        Tensor noisy = Rand(new TensorShape(1, T, cfg.LatentChannels), 11);
        Tensor context = Rand(new TensorShape(1, T, ctxCh), 22);
        Tensor conditions = Rand(new TensorShape(1, L, encDim), 33);

        using AceStep15Dit dit = new(cfg);
        dit.LoadWeights(w);
        using CpuBackend backend = new();

        Tensor v1 = dit.Forward(backend, noisy, context, conditions, Sigma, Sigma);
        Tensor v2 = dit.Forward(backend, noisy, context, conditions, Sigma, Sigma);

        Assert.Equal(1, (int)v1.Shape[0]);
        Assert.Equal(T, (int)v1.Shape[1]);
        Assert.Equal(cfg.LatentChannels, (int)v1.Shape[2]);

        // Same inputs → identical outputs (no hidden nondeterminism in the rewrite).
        long n = v1.Shape.ElementCount;
        float* p1 = (float*)v1.DataPointer;
        float* p2 = (float*)v2.DataPointer;
        float selfMax = 0f;
        for (long i = 0; i < n; i++) selfMax = MathF.Max(selfMax, MathF.Abs(p1[i] - p2[i]));
        _output.WriteLine($"self-consistency maxAbs = {selfMax:E3} over {n} elems");
        Assert.True(selfMax == 0f, $"non-deterministic forward: maxAbs {selfMax}");

        string? dump = Environment.GetEnvironmentVariable("HARTSY_ACE15_DUMP");
        if (!string.IsNullOrEmpty(dump))
        {
            byte[] buf = new byte[n * sizeof(float)];
            fixed (byte* dst = buf) Buffer.MemoryCopy(p1, dst, buf.Length, buf.Length);
            File.WriteAllBytes(dump, buf);
            _output.WriteLine($"velocity dumped to {dump} ({n} floats)");

            string? golden = Environment.GetEnvironmentVariable("HARTSY_ACE15_GOLDEN");
            if (!string.IsNullOrEmpty(golden) && File.Exists(golden))
            {
                byte[] gbuf = File.ReadAllBytes(golden);
                Assert.Equal(gbuf.Length, buf.Length);
                float maxAbs = 0f, gMax = 0f;
                fixed (byte* gp = gbuf)
                {
                    float* gf = (float*)gp;
                    for (long i = 0; i < n; i++)
                    {
                        maxAbs = MathF.Max(maxAbs, MathF.Abs(gf[i] - p1[i]));
                        gMax = MathF.Max(gMax, MathF.Abs(gf[i]));
                    }
                }
                _output.WriteLine($"vs golden: maxAbs={maxAbs:E4}, rel={(gMax > 0 ? maxAbs / gMax : 0):E4}");
                Assert.True(maxAbs < 1e-4f, $"residency rewrite diverged from golden: maxAbs {maxAbs}");
            }
        }

        v1.Dispose(); v2.Dispose();
        noisy.Dispose(); context.Dispose(); conditions.Dispose();
    }

    private static Tensor Rand(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        long n = shape.ElementCount;
        uint s = (uint)(seed * 2654435761u + 1u);
        for (long i = 0; i < n; i++)
        {
            s = s * 1664525u + 1013904223u;
            p[i] = ((s >> 8) / (float)(1 << 24) - 0.5f) * 2.0f;
        }
        return t;
    }
}
