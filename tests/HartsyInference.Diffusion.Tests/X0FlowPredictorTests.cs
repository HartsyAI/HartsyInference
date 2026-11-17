using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The clean-sample (x0) flow predictor and the SNR calibration, on the CPU backend. Expected values are worked
/// from the formulas the code states, by hand or by an independent loop in the test; nothing is compared with a reference
/// implementation.</summary>
public sealed class X0FlowPredictorTests
{
    private const float Floor = 0.05f;

    [Fact]
    public void GuidanceFreeVelocityIsTheFlooredExpression()
    {
        using CpuBackend backend = new();
        using Tensor x = Tensor1([0.5f, -1f, 2f]);
        float[] x0 = [0.1f, 0.2f, 0.3f];

        using DenoisePrediction prediction = new X0FlowPredictor(backend, (_, _, _) => (Tensor1(x0), null), 1f, Floor)
            .Predict(x, 0.4f, 0);

        Assert.Same(prediction.Cond, prediction.Uncond);
        Assert.Equal(1f, prediction.Guidance);
        AssertEqual([(0.5f - 0.1f) / 0.4f, (-1f - 0.2f) / 0.4f, (2f - 0.3f) / 0.4f], prediction.Cond);
    }

    [Fact]
    public void SigmaBelowTheFloorUsesTheFloor()
    {
        using CpuBackend backend = new();
        using Tensor x = Tensor1([1f]);

        using DenoisePrediction prediction = new X0FlowPredictor(backend, (_, _, _) => (Tensor1([0.5f]), null), 1f, Floor)
            .Predict(x, 0.01f, 0);

        AssertEqual([(1f - 0.5f) / Floor], prediction.Cond);
    }

    [Fact]
    public void WithoutATransformThePairGoesBackRawAtTheGuidanceTheAnchorImplies()
    {
        using CpuBackend backend = new();
        using Tensor x = Tensor1([0.5f, 1f]);
        float[] cond = [0.2f, 0.4f], uncond = [0.1f, 0.9f];
        const float sigma = 0.5f;

        using DenoisePrediction uncondAnchored = new X0FlowPredictor(backend,
            (_, _, _) => (Tensor1(cond), Tensor1(uncond)), 4f, Floor, GuidanceAnchor.Uncond).Predict(x, sigma, 0);
        using DenoisePrediction condAnchored = new X0FlowPredictor(backend,
            (_, _, _) => (Tensor1(cond), Tensor1(uncond)), 4f, Floor, GuidanceAnchor.Cond).Predict(x, sigma, 0);

        Assert.Equal(4f, uncondAnchored.Guidance);
        Assert.Equal(5f, condAnchored.Guidance);
        Assert.NotSame(uncondAnchored.Cond, uncondAnchored.Uncond);
        AssertEqual([(0.5f - 0.2f) / sigma, (1f - 0.4f) / sigma], uncondAnchored.Cond);
        AssertEqual([(0.5f - 0.1f) / sigma, (1f - 0.9f) / sigma], uncondAnchored.Uncond);
    }

    /// <summary>The raw-pair path must equal the existing hand-written conversion, bit for bit: it is what Chroma Radiance
    /// does today.</summary>
    [Fact]
    public void RawPairIsBitIdenticalToTheExistingConversion()
    {
        using CpuBackend backend = new();
        using Tensor x = Noise(9, 64);
        using Tensor condX0 = Noise(10, 64);
        float[] condValues = condX0.AsReadOnlySpan<float>().ToArray();

        using DenoisePrediction prediction = new X0FlowPredictor(backend, (_, _, _) => (Tensor1(condValues), null), 1f, 1e-6f)
            .Predict(x, 0.37f, 0);
        using Tensor expected = X0Prediction.ToVelocityDevice(backend, condX0, x, 0.37f);

        Assert.True(expected.AsReadOnlySpan<float>().SequenceEqual(prediction.Cond.AsReadOnlySpan<float>()));
    }

    [Fact]
    public void ATransformSeesTheGuidedX0AndTheVelocityIsTakenFromItsResult()
    {
        using CpuBackend backend = new();
        using Tensor x = Tensor1([0.5f, 1f]);
        float[] cond = [0.2f, 0.4f], uncond = [0.1f, 0.9f];
        const float sigma = 0.5f, scale = 4f;
        float[] seen = [];

        // Clamp to [0, 0.5] after the combine: u + 4(c - u) is [0.5, -1.1], so the clamp must act on it.
        X0FlowPredictor.GuidedX0Transform clamp = (guided, s) =>
        {
            seen = guided.AsReadOnlySpan<float>().ToArray();
            Assert.Equal(sigma, s);
            Tensor clamped = new(guided.Shape, DType.F32);
            for (int i = 0; i < seen.Length; i++) clamped.AsSpan<float>()[i] = Math.Clamp(seen[i], 0f, 0.5f);
            return clamped;
        };
        using DenoisePrediction prediction = new X0FlowPredictor(backend,
            (_, _, _) => (Tensor1(cond), Tensor1(uncond)), scale, Floor, GuidanceAnchor.Uncond, clamp).Predict(x, sigma, 0);

        Assert.Equal(1f, prediction.Guidance);
        Assert.Same(prediction.Cond, prediction.Uncond);
        Assert.Equal(0.1f + scale * (0.2f - 0.1f), seen[0], 5);
        Assert.Equal(0.9f + scale * (0.4f - 0.9f), seen[1], 5);
        AssertEqual([(0.5f - 0.5f) / sigma, (1f - 0f) / sigma], prediction.Cond);
    }

    [Fact]
    public void ACondAnchoredCombineIsCondPlusScaleTimesTheDifference()
    {
        using CpuBackend backend = new();
        using Tensor x = Tensor1([0f]);
        float[] seen = [];

        using DenoisePrediction prediction = new X0FlowPredictor(backend,
            (_, _, _) => (Tensor1([0.3f]), Tensor1([0.1f])), 2f, Floor, GuidanceAnchor.Cond,
            (guided, _) => { seen = guided.AsReadOnlySpan<float>().ToArray(); return guided; }).Predict(x, 0.5f, 0);

        Assert.Equal(0.3f + 2f * (0.3f - 0.1f), seen[0], 5);
    }

    /// <summary>A model that returns the true clean sample, driven by the real Euler sampler on the real schedule at the
    /// shift and floor PixVid states (50 steps, shift 11, floor 0.05), must land on that sample. The floor makes the
    /// last steps under-correct, so it lands near it, not on it; the bound is what the floor allows.</summary>
    [Fact]
    public void AnOracleModelDrivenByTheEulerSamplerRecoversTheCleanSample()
    {
        using CpuBackend backend = new();
        const int steps = 50;
        FlowMatchEulerDiscreteScheduler scheduler = new(shift: 11f);
        scheduler.SetTimesteps(steps);
        ISampler sampler = FlowMatchSampling.Resolve(null, scheduler, seed: 1, "x0 oracle test");
        float[] clean = [0.8f, -0.4f, 0.1f, 0.6f], noise = [-1.2f, 0.7f, 1.5f, -0.3f];
        using Tensor z = Tensor1(noise);
        X0FlowPredictor predictor = new(backend, (_, _, _) => (Tensor1(clean), null), 1f, Floor);

        sampler.Reset(z.Shape);
        for (int i = 0; i < steps; i++) sampler.Step(backend, z, predictor, i);

        float worst = 0f, span = 0f;
        for (int i = 0; i < clean.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(z.AsReadOnlySpan<float>()[i] - clean[i]));
            span = Math.Max(span, Math.Abs(noise[i] - clean[i]));
        }
        Assert.True(worst <= 0.05f * span, $"the sample ended {worst} from the clean one; the floor allows {0.05f * span}");
    }

    [Fact]
    public void CalibrationMapsTheEndpointsAndComposesByMultiplying()
    {
        SnrCalibration calibration = new(2.85f);

        Assert.Equal(0f, calibration.Sigma(0f));
        Assert.Equal(1f, calibration.Sigma(1f), 6);
        Assert.Equal(1f / 2.85f, calibration.InputScale(0f), 6);
        Assert.Equal(1f, calibration.InputScale(1f), 6);
        // sigma' < sigma for c > 1, and (1 - s')/s' = c (1 - s)/s.
        float sigma = 0.3f, calibrated = calibration.Sigma(sigma);
        Assert.True(calibrated < sigma);
        Assert.Equal(2.85f * (1f - sigma) / sigma, (1f - calibrated) / calibrated, 4);
        // Input scale is sigma' / sigma.
        Assert.Equal(calibrated / sigma, calibration.InputScale(sigma), 5);
        Assert.Equal(2.85f * 2f, calibration.Compose(new SnrCalibration(2f)).Scale, 5);
        // The 2x2-pooled case in the paper: timestep s / (2 - s), input scale 2 / (2 - s).
        SnrCalibration pooled = new(2f);
        Assert.Equal(0.3f / 1.7f, pooled.Sigma(0.3f), 6);
        // The paper's input factor 2 / (2 - s) is this 1 / (2 - s) times the pooling's own factor 2, which is not part of the calibration.
        Assert.Equal(1f / 1.7f, pooled.InputScale(0.3f), 5);
        Assert.Equal(1f, default(SnrCalibration).Compose(default).Scale);
    }

    [Fact]
    public void CalibrationRejectsANonPositiveScale()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SnrCalibration(0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SnrCalibration(-1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SnrCalibration(float.PositiveInfinity));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SnrCalibration(float.NaN));
        Assert.Equal(0.4f, default(SnrCalibration).Sigma(0.4f), 6);   // the unset struct is the identity
        Assert.Equal(1f, default(SnrCalibration).Scale);
        Assert.Equal(new SnrCalibration(1f), default(SnrCalibration));
        Assert.Equal(new SnrCalibration(1f).GetHashCode(), default(SnrCalibration).GetHashCode());
        Assert.True(new SnrCalibration(2f) != default);
    }

    /// <summary>Returning one tensor for both branches is how the engine signals a guidance-free evaluation, so it must not
    /// be read as a guided pair (which would use or free the tensor twice), on either path.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheSameTensorForBothBranchesIsGuidanceFree(bool withTransform)
    {
        using CpuBackend backend = new();
        using Tensor x = Tensor1([0.5f, 1f]);
        const float sigma = 0.5f;
        X0FlowPredictor.GuidedX0Transform? transform = withTransform ? (guided, _) => guided : null;

        using DenoisePrediction prediction = new X0FlowPredictor(backend,
            (_, _, _) => { Tensor both = Tensor1([0.2f, 0.4f]); return (both, both); }, 4f, Floor, GuidanceAnchor.Uncond, transform)
            .Predict(x, sigma, 0);

        Assert.Same(prediction.Cond, prediction.Uncond);
        Assert.Equal(1f, prediction.Guidance);
        AssertEqual([(0.5f - 0.2f) / sigma, (1f - 0.4f) / sigma], prediction.Cond);
    }

    [Fact]
    public void ATransformThatReturnsItsInputWorksOnAGuidanceFreeEvaluation()
    {
        using CpuBackend backend = new();
        using Tensor x = Tensor1([0.5f]);

        using DenoisePrediction prediction = new X0FlowPredictor(backend, (_, _, _) => (Tensor1([0.2f]), null), 4f, Floor,
            GuidanceAnchor.Uncond, (guided, _) => guided).Predict(x, 0.5f, 0);

        AssertEqual([(0.5f - 0.2f) / 0.5f], prediction.Cond);
    }

    [Fact]
    public void TheFloorIsForwardedAsTheEpsilonOfTheConversion()
    {
        using CpuBackend backend = new();
        using Tensor x = Noise(3, 16);
        using Tensor condX0 = Noise(4, 16);
        float[] condValues = condX0.AsReadOnlySpan<float>().ToArray();

        using DenoisePrediction prediction = new X0FlowPredictor(backend, (_, _, _) => (Tensor1(condValues), null), 1f, Floor)
            .Predict(x, 0.01f, 0);
        using Tensor expected = X0Prediction.ToVelocityDevice(backend, condX0, x, 0.01f, Floor);

        Assert.True(expected.AsReadOnlySpan<float>().SequenceEqual(prediction.Cond.AsReadOnlySpan<float>()));
    }

    [Fact]
    public void ATransformThatThrowsPropagatesItsException()
    {
        using CpuBackend backend = new();
        using Tensor x = Tensor1([0.5f]);
        X0FlowPredictor predictor = new(backend, (_, _, _) => (Tensor1([0.2f]), Tensor1([0.1f])), 4f, Floor,
            GuidanceAnchor.Uncond, (_, _) => throw new InvalidOperationException("transform failed"));

        Assert.Throws<InvalidOperationException>(() => predictor.Predict(x, 0.5f, 0));
    }

    private static Tensor Tensor1(float[] values)
    {
        Tensor tensor = new(new TensorShape(values.Length), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static Tensor Noise(int seed, int count)
    {
        Random rng = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
        return Tensor1(values);
    }

    private static void AssertEqual(float[] expected, Tensor actual)
    {
        ReadOnlySpan<float> values = actual.AsReadOnlySpan<float>();
        Assert.Equal(expected.Length, values.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], values[i], 4);
    }
}
