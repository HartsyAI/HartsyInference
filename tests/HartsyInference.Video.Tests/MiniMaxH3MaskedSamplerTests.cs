using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Video.Tests;

/// <summary>Locks the MiniMax-H3 mask algebra merged in ComfyUI PR #15375. Token mask <c>M</c> controls the DiT
/// input/timestep, while raw feature mask <c>m</c> controls the denoised-source blend; conflating the two creates
/// visible 2x2 latent-patch boundaries.</summary>
public sealed class MiniMaxH3MaskedSamplerTests
{
    [Fact]
    public void BlackTokenUsesFixedVisualConditionStrengthAtSigmaPointEight()
    {
        using CpuBackend backend = new CpuBackend();
        using Tensor source = TensorFrom([10f, 20f, 30f, 40f], new TensorShape(1, 4));
        using Tensor noise = TensorFrom([-10f, -20f, -30f, -40f], source.Shape);
        using Tensor state = TensorFrom([100f, 200f, 300f, 400f], source.Shape);
        using Tensor injection = new Tensor(source.Shape, DType.F32);
        using Tensor modelInput = new Tensor(source.Shape, DType.F32);
        using Tensor tokenMask = TensorFrom([0f], new TensorShape(1));

        MiniMaxH3Pipeline.BuildVideoMaskInjection(backend, injection, source, noise);
        MiniMaxH3Pipeline.BuildMaskedModelInput(backend, modelInput, state, injection, tokenMask);

        float pin = MiniMaxH3Schedule.VisualCondTimestep;
        float[] expected = Snapshot(source).Zip(Snapshot(noise), (s, n) => pin * s + (1f - pin) * n).ToArray();
        AssertClose(expected, Snapshot(modelInput), 1e-6f, "fixed condition injection");
        float currentSigmaValue = 0.2f * 10f + 0.8f * -10f;
        Assert.NotEqual(currentSigmaValue, Snapshot(modelInput)[0]);
    }

    [Fact]
    public void MixedRawFeaturesRemainDistinctWhenTokenMaskPoolsToOne()
    {
        using CpuBackend backend = new CpuBackend();
        using Tensor state = TensorFrom([100f, 200f, 300f, 400f], new TensorShape(1, 4));
        using Tensor source = TensorFrom([10f, 20f, 30f, 40f], state.Shape);
        using Tensor injection = TensorFrom([-1f, -2f, -3f, -4f], state.Shape);
        using Tensor modelInput = new Tensor(state.Shape, DType.F32);
        using Tensor denoised = new Tensor(state.Shape, DType.F32);
        using Tensor velocity = TensorFrom([0f, 0f, 0f, 0f], state.Shape);
        using Tensor tokenMask = TensorFrom([1f], new TensorShape(1));
        using Tensor featureMask = TensorFrom([0f, 0.25f, 0.75f, 1f], new TensorShape(1, 4));

        (IReadOnlyList<float>? tokens, IReadOnlyList<float>? features) =
            MiniMaxH3Pipeline.NormalizeDenoiseMask(
                [1f], [0f, 0.25f, 0.75f, 1f], source,
                expectedRows: 1, expectedFeatures: 4, patchArea: 4, modality: "video");
        Assert.NotNull(tokens);
        Assert.NotNull(features);

        MiniMaxH3Pipeline.BuildMaskedModelInput(backend, modelInput, state, injection, tokenMask);
        Assert.Equal(Snapshot(state), Snapshot(modelInput));
        MiniMaxH3Pipeline.AdvanceMaskedState(
            backend, modelInput, denoised, state, modelInput, velocity, source, featureMask,
            MaskBroadcastLayout.PackedChannelOuter, nativeSigma: 0.8f, nextNativeSigma: 0.4f);

        AssertClose([55f, 132.5f, 266.25f, 400f], Snapshot(modelInput), 1e-6f,
            "raw feature blend after M=1 pooling");
    }

    [Fact]
    public void OneStepTensorPathMatchesIndependentScalarOracle()
    {
        const int rows = 2, featuresPerRow = 8, patchArea = 4;
        float[] stateValues = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f];
        float[] injectionValues = [-4f, -3f, -2f, -1f, 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f];
        float[] velocityValues = [0.5f, -1f, 1.5f, -2f, 2.5f, -3f, 3.5f, -4f, 4.5f, -5f, 5.5f, -6f, 6.5f, -7f, 7.5f, -8f];
        float[] sourceValues = [20f, 19f, 18f, 17f, 16f, 15f, 14f, 13f, 12f, 11f, 10f, 9f, 8f, 7f, 6f, 5f];
        float[] tokenValues = [0.5f, 1f];
        float[] rawValues = [0f, 0.2f, 0.4f, 0.5f, 0.1f, 0.4f, 0.8f, 1f];
        const float sigma = 0.8f, nextSigma = 0.3f;

        using CpuBackend backend = new CpuBackend();
        TensorShape shape = new TensorShape(rows, featuresPerRow);
        using Tensor state = TensorFrom(stateValues, shape);
        using Tensor injection = TensorFrom(injectionValues, shape);
        using Tensor velocity = TensorFrom(velocityValues, shape);
        using Tensor source = TensorFrom(sourceValues, shape);
        using Tensor tokenMask = TensorFrom(tokenValues, new TensorShape(rows));
        using Tensor featureMask = TensorFrom(rawValues, new TensorShape(rows, patchArea));
        using Tensor modelInput = new Tensor(shape, DType.F32);
        using Tensor denoised = new Tensor(shape, DType.F32);

        MiniMaxH3Pipeline.BuildMaskedModelInput(backend, modelInput, state, injection, tokenMask);
        float[] expectedModelInput = new float[stateValues.Length];
        float[] expectedNext = new float[stateValues.Length];
        float stateStrength = nextSigma / sigma;
        for (int i = 0; i < stateValues.Length; i++)
        {
            int row = i / featuresPerRow;
            float token = tokenValues[row];
            float q = token * stateValues[i] + (1f - token) * injectionValues[i];
            expectedModelInput[i] = q;
            float dModel = q + sigma * velocityValues[i];
            float raw = rawValues[row * patchArea + i % patchArea];
            float d = raw * dModel + (1f - raw) * sourceValues[i];
            expectedNext[i] = stateStrength * stateValues[i] + (1f - stateStrength) * d;
        }
        AssertClose(expectedModelInput, Snapshot(modelInput), 1e-6f, "pooled token input");

        MiniMaxH3Pipeline.AdvanceMaskedState(
            backend, modelInput, denoised, state, modelInput, velocity, source, featureMask,
            MaskBroadcastLayout.PackedChannelOuter, sigma, nextSigma);

        AssertClose(expectedNext, Snapshot(modelInput), 2e-6f, "masked Euler state");
    }

    [Fact]
    public void AllWhiteRawMaskCollapsesToExactUnmaskedPath()
    {
        (IReadOnlyList<float>? tokens, IReadOnlyList<float>? features) =
            MiniMaxH3Pipeline.NormalizeDenoiseMask(
                [1f], [1f, 1f, 1f, 1f], source: null,
                expectedRows: 1, expectedFeatures: 4, patchArea: 4, modality: "video");

        Assert.Null(tokens);
        Assert.Null(features);
    }

    [Fact]
    public void InconsistentTokenAndRawMasksFailBeforeSampling()
    {
        using Tensor source = new(new TensorShape(1, 4), DType.F32);

        ArgumentException error = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3Pipeline.NormalizeDenoiseMask(
                [0.5f], [0f, 0.25f, 0.75f, 1f], source,
                expectedRows: 1, expectedFeatures: 4, patchArea: 4, modality: "video"));

        Assert.Contains("raw feature maximum requires 1", error.Message, StringComparison.Ordinal);
    }

    private static Tensor TensorFrom(float[] values, TensorShape shape)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static float[] Snapshot(Tensor tensor) => tensor.AsReadOnlySpan<float>().ToArray();

    private static void AssertClose(float[] expected, float[] actual, float tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"{label}: index {i}, expected {expected[i]:R}, got {actual[i]:R}");
        }
    }
}
