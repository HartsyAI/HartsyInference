using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Focused parity and residency gates for SD3's final AdaLN/projection and token-grid unpatchify.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Sd3FinalOutputResidencyTests
{
    private readonly ITestOutputHelper _output;

    public Sd3FinalOutputResidencyTests(ITestOutputHelper output) => _output = output;

    private static Sd3Config TinyConfig => new()
    {
        Depth = 1,
        HiddenSize = 32,
        NumHeads = 4,
        PatchSize = 2,
        InChannels = 3,
        JointAttentionDim = 16,
        PooledProjectionDim = 12,
        UseQkNorm = false,
    };

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Theory]
    [InlineData(1, 15)]
    [InlineData(2, 7)]
    public void ApplyFinalLayer_CpuMatchesLegacyHostMath(int batch, int seqLen)
    {
        Sd3Config config = TinyConfig;
        Dictionary<string, Tensor> weights = Sd3WeightBuilder.Build(config);
        try
        {
            using Sd3Transformer transformer = new(config);
            transformer.LoadWeights(weights);
            using CpuBackend cpu = new();
            using Tensor hidden = Sd3WeightBuilder.Rand(
                new TensorShape(batch, seqLen, config.HiddenSize), seed: 501 + batch, scale: 0.8f);
            using Tensor temb = Sd3WeightBuilder.Rand(
                new TensorShape(batch, config.HiddenSize), seed: 601 + seqLen, scale: 0.5f);

            using Tensor expected = LegacyFinalProjection(cpu, hidden, temb, weights, config, batch, seqLen);
            using Tensor actual = transformer.ApplyFinalLayer(cpu, hidden, temb, batch, seqLen);

            AssertClose(Snapshot(expected), Snapshot(actual), 2e-5f, "CPU final projection");
        }
        finally
        {
            Sd3WeightBuilder.DisposeAll(weights);
        }
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void ApplyFinalLayerAndUnpatchify_CudaMatchCpu_WithoutD2h()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int gridH = 3;
        const int gridW = 5;
        const int batch = 2;
        int seqLen = gridH * gridW;
        Sd3Config config = TinyConfig;
        Dictionary<string, Tensor> weights = Sd3WeightBuilder.Build(config);
        try
        {
            using Sd3Transformer transformer = new(config);
            transformer.LoadWeights(weights);
            using Tensor hiddenHost = Sd3WeightBuilder.Rand(
                new TensorShape(batch, seqLen, config.HiddenSize), seed: 701, scale: 0.8f);
            using Tensor tembHost = Sd3WeightBuilder.Rand(
                new TensorShape(batch, config.HiddenSize), seed: 702, scale: 0.5f);
            Unpatchify unpatchify = new(config.PatchSize, config.InChannels);

            using CpuBackend cpu = new();
            using Tensor projectedReference = LegacyFinalProjection(
                cpu, hiddenHost, tembHost, weights, config, batch, seqLen);
            using Tensor spatialReference = unpatchify.Forward(projectedReference, batch, gridH, gridW);
            float[] expected = Snapshot(spatialReference);

            using CudaBackend cuda = new(0, PtxDir());
            cuda.PreloadWeights(transformer.EnumerateSharedWeights());
            using Tensor hiddenCuda = new(hiddenHost.Shape, DType.F32);
            using Tensor tembCuda = new(tembHost.Shape, DType.F32);
            cuda.Scale(hiddenCuda, hiddenHost, 1.0f);
            cuda.Scale(tembCuda, tembHost, 1.0f);
            cuda.Sync();
            cuda.ResetD2hSyncCount();

            using Tensor projectedCuda = transformer.ApplyFinalLayer(cuda, hiddenCuda, tembCuda, batch, seqLen);
            using Tensor spatialCuda = unpatchify.Forward(cuda, projectedCuda, batch, gridH, gridW);
            cuda.Sync();

            Assert.Equal(0, cuda.GetD2hSyncCount());
            Assert.Equal(new TensorShape(batch, config.InChannels, gridH * config.PatchSize, gridW * config.PatchSize),
                spatialCuda.Shape);
            AssertClose(expected, Snapshot(spatialCuda), 4e-4f, "CUDA final output");
            cuda.FreeWeights(transformer.EnumerateSharedWeights());
        }
        finally
        {
            Sd3WeightBuilder.DisposeAll(weights);
        }
    }

    [Fact]
    public void BackendUnpatchify_BatchTwoPreservesLegacyLayout()
    {
        const int batch = 2;
        const int channels = 3;
        const int patch = 2;
        const int gridH = 2;
        const int gridW = 3;
        int patchDim = channels * patch * patch;
        using Tensor tokens = Sd3WeightBuilder.Rand(
            new TensorShape(batch, gridH * gridW, patchDim), seed: 801, scale: 1.0f);
        Unpatchify unpatchify = new(patch, channels);
        using CpuBackend cpu = new();

        using Tensor expected = unpatchify.Forward(tokens, batch, gridH, gridW);
        using Tensor actual = unpatchify.Forward(cpu, tokens, batch, gridH, gridW);

        Assert.Equal(Snapshot(expected), Snapshot(actual));
    }

    [Fact]
    public void UnpatchifyRejectsInvalidGridAndTokenShapeBeforeDispatch()
    {
        Unpatchify unpatchify = new(patchSize: 2, outChannels: 3);
        using CpuBackend cpu = new();
        using Tensor valid = new(new TensorShape(2, 6, 12), DType.F32);
        using Tensor wrongBatch = new(new TensorShape(1, 6, 12), DType.F32);
        using Tensor wrongSequence = new(new TensorShape(2, 5, 12), DType.F32);
        using Tensor wrongPatchVolume = new(new TensorShape(2, 6, 11), DType.F32);

        Assert.Throws<ArgumentOutOfRangeException>(() => unpatchify.Forward(cpu, valid, 2, 0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => unpatchify.Forward(cpu, valid, 2, 2, -1));
        Assert.Throws<ArgumentException>(() => unpatchify.Forward(cpu, wrongBatch, 2, 2, 3));
        Assert.Throws<ArgumentException>(() => unpatchify.Forward(cpu, wrongSequence, 2, 2, 3));
        Assert.Throws<ArgumentException>(() => unpatchify.Forward(cpu, wrongPatchVolume, 2, 2, 3));
    }

    private static Tensor LegacyFinalProjection(
        IBackend backend,
        Tensor hidden,
        Tensor temb,
        IReadOnlyDictionary<string, Tensor> weights,
        Sd3Config config,
        int batch,
        int seqLen)
    {
        int dim = config.HiddenSize;
        using Tensor activated = new(new TensorShape(batch, dim), DType.F32);
        backend.Silu(activated, temb);
        using Tensor modParams = new(new TensorShape(batch, 2 * dim), DType.F32);
        backend.Linear(modParams, activated, weights["norm_out.linear.weight"], weights["norm_out.linear.bias"]);

        using Tensor modulated = new(new TensorShape(batch, seqLen, dim), DType.F32);
        float* hiddenPtr = (float*)hidden.DataPointer;
        float* modPtr = (float*)modParams.DataPointer;
        float* outputPtr = (float*)modulated.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int modBase = b * 2 * dim;
            for (int s = 0; s < seqLen; s++)
            {
                int row = (b * seqLen + s) * dim;
                float mean = 0.0f;
                for (int d = 0; d < dim; d++) mean += hiddenPtr[row + d];
                mean /= dim;
                float variance = 0.0f;
                for (int d = 0; d < dim; d++)
                {
                    float diff = hiddenPtr[row + d] - mean;
                    variance += diff * diff;
                }
                variance /= dim;
                float invStd = 1.0f / MathF.Sqrt(variance + 1e-6f);
                for (int d = 0; d < dim; d++)
                {
                    float normed = (hiddenPtr[row + d] - mean) * invStd;
                    outputPtr[row + d] = normed * (1.0f + modPtr[modBase + dim + d])
                        + modPtr[modBase + d];
                }
            }
        }

        Tensor projected = new(
            new TensorShape(batch, seqLen, config.PatchSize * config.PatchSize * config.InChannels), DType.F32);
        try
        {
            backend.Linear(projected, modulated, weights["proj_out.weight"], weights["proj_out.bias"]);
            return projected;
        }
        catch
        {
            projected.Dispose();
            throw;
        }
    }

    private static float[] Snapshot(Tensor tensor)
    {
        float[] result = new float[tensor.ElementCount];
        new ReadOnlySpan<float>((float*)tensor.DataPointer, result.Length).CopyTo(result);
        return result;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxError = 0.0f;
        int maxIndex = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(expected[i]),
                $"{label}: expected value at {i} is non-finite ({expected[i]}).");
            Assert.True(float.IsFinite(actual[i]),
                $"{label}: actual value at {i} is non-finite ({actual[i]}).");
            float error = MathF.Abs(expected[i] - actual[i]);
            Assert.True(float.IsFinite(error),
                $"{label}: absolute error at {i} is non-finite (expected={expected[i]}, actual={actual[i]}).");
            if (error > maxError)
            {
                maxError = error;
                maxIndex = i;
            }
        }
        Assert.True(maxError <= tolerance,
            $"{label}: max error {maxError:E6} at {maxIndex} exceeds {tolerance:E2}.");
    }
}
