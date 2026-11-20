using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Idol;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Idol;

public sealed class IdolSamplerTests
{
    private static readonly IBackend Backend = new CpuBackend();

    [Fact]
    public void Sample_IsIdolSdksEulerLoopWithGuidance_OnATinyModel()
    {
        IdolConfig tiny = IdolModelTests.TinyConfig();
        Dictionary<string, Tensor> tensors = IdolModelTests.TinyTensors(tiny, seed: 11);
        try
        {
            using IdolWeights weights = IdolWeights.FromTensors(tiny, tensors);
            using IdolDenoiser model = new(weights);
            using Tensor positive = IdolModelTests.Random(new TensorShape(1, tiny.CondTokens, tiny.ModelChannels), 21, 1f);
            using Tensor negative = IdolModelTests.Random(new TensorShape(1, tiny.CondTokens, tiny.ModelChannels), 22, 1f);
            using Tensor z = IdolModelTests.Random(new TensorShape(1, tiny.Tokens, tiny.LatentChannels), 23, 1f);
            using Tensor cam = IdolModelTests.Random(new TensorShape(1, 1, tiny.CameraChannels), 24, 1f);
            const int steps = 3;
            const float cfg = 3f;
            const double shift = 3.0;

            (Tensor latent, Tensor camera) = IdolSampler.Sample(Backend, model, positive, negative, z, cam, steps, cfg, shift);
            using (latent)
            using (camera)
            {
                // idol-sdk's loop written out: t = linspace(1, 0, steps + 1) shifted, v = (1 + w) v(cond) - w v(zeros), x -= v (t - t_prev).
                float[] x = z.AsReadOnlySpan<float>().ToArray(), c = cam.AsReadOnlySpan<float>().ToArray();
                for (int i = 0; i < steps; i++)
                {
                    double t = Warp(1.0 - (double)i / steps, shift), tPrev = Warp(1.0 - (double)(i + 1) / steps, shift);
                    using Tensor xt = Tensor(x, z.Shape), ct = Tensor(c, cam.Shape);
                    (Tensor pl, Tensor pc) = model.Forward(Backend, xt, ct, (float)(1000.0 * t), positive);
                    (Tensor nl, Tensor nc) = model.Forward(Backend, xt, ct, (float)(1000.0 * t), negative);
                    using (pl) using (pc) using (nl) using (nc)
                    {
                        ReadOnlySpan<float> vl = pl.AsReadOnlySpan<float>(), vn = nl.AsReadOnlySpan<float>(), cl = pc.AsReadOnlySpan<float>(), cn = nc.AsReadOnlySpan<float>();
                        for (int k = 0; k < x.Length; k++) x[k] -= (float)(((1 + cfg) * vl[k] - cfg * vn[k]) * (t - tPrev));
                        for (int k = 0; k < c.Length; k++) c[k] -= (float)(((1 + cfg) * cl[k] - cfg * cn[k]) * (t - tPrev));
                    }
                }

                (double relLatent, _) = RelativeL2(latent.AsReadOnlySpan<float>(), x);
                (double relCamera, _) = RelativeL2(camera.AsReadOnlySpan<float>(), c);
                Assert.True(relLatent < 1e-5, $"latent relRMS {relLatent}");
                Assert.True(relCamera < 1e-5, $"camera relRMS {relCamera}");
                Assert.NotEqual(z.AsReadOnlySpan<float>().ToArray(), latent.AsReadOnlySpan<float>().ToArray());
            }
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    [Fact]
    public void Sample_WithoutGuidanceRunsTheConditionalPassOnly_AndLeavesTheNoiseAlone()
    {
        IdolConfig tiny = IdolModelTests.TinyConfig();
        Dictionary<string, Tensor> tensors = IdolModelTests.TinyTensors(tiny, seed: 12);
        try
        {
            using IdolWeights weights = IdolWeights.FromTensors(tiny, tensors);
            using IdolDenoiser model = new(weights);
            using Tensor positive = IdolModelTests.Random(new TensorShape(1, tiny.CondTokens, tiny.ModelChannels), 31, 1f);
            using Tensor negative = IdolModelTests.Random(new TensorShape(1, tiny.CondTokens, tiny.ModelChannels), 32, 1f);
            using Tensor z = IdolModelTests.Random(new TensorShape(1, tiny.Tokens, tiny.LatentChannels), 33, 1f);
            using Tensor cam = IdolModelTests.Random(new TensorShape(1, 1, tiny.CameraChannels), 34, 1f);
            float[] noise = z.AsReadOnlySpan<float>().ToArray();

            (Tensor a, Tensor ca) = IdolSampler.Sample(Backend, model, positive, negative, z, cam, steps: 2, guidance: 0f);
            (Tensor b, Tensor cb) = IdolSampler.Sample(Backend, model, positive, positive, z, cam, steps: 2, guidance: 0f);
            using (a) using (ca) using (b) using (cb)
            {
                // The negative stream is never read at guidance 0.
                Assert.Equal(a.AsReadOnlySpan<float>().ToArray(), b.AsReadOnlySpan<float>().ToArray());
                Assert.Equal(noise, z.AsReadOnlySpan<float>().ToArray());
            }

            Assert.Throws<ArgumentOutOfRangeException>(() => IdolSampler.Sample(Backend, model, positive, negative, z, cam, 2, -1f));
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    private static double Warp(double linear, double shift) => shift * linear / (1.0 + (shift - 1.0) * linear);

    private static Tensor Tensor(float[] values, TensorShape shape)
    {
        Tensor t = new(shape, DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static (double Relative, double MaxAbs) RelativeL2(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        double error = 0, reference = 0, max = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = (double)actual[i] - expected[i];
            error += d * d;
            reference += (double)expected[i] * expected[i];
            max = Math.Max(max, Math.Abs(d));
        }

        return (Math.Sqrt(error / reference), max);
    }
}
