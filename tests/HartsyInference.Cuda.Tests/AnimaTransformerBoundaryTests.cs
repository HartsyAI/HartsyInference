using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// End-to-end boundary coverage for Anima's patch-input and spatial-output transforms. A zero-block
/// transformer isolates the two boundaries plus the timestep/final projection without hiding a host
/// round-trip inside attention or MLP work.
/// </summary>
[Collection("CudaSerial")]
public sealed unsafe class AnimaTransformerBoundaryTests
{
    private const int Batch = 2;
    private const int LatentHeight = 6;
    private const int LatentWidth = 10;
    private const int TextSequence = 7;
    private const int TextWidth = 8;

    private readonly ITestOutputHelper _output;

    public AnimaTransformerBoundaryTests(ITestOutputHelper output) => _output = output;

    private static AnimaConfig TinyConfig => new()
    {
        HiddenSize = 8,
        NumHeads = 2,
        HeadDim = 4,
        NumLayers = 0,
        MlpRatio = 2.0f,
        InChannels = 2,
        OutChannels = 3,
        PatchSize = (1, 2, 2),
        MaxSize = (1, 8, 8),
        RopeScale = (1.0f, 1.0f, 1.0f),
        RopeTheta = 10_000.0f,
        ConcatPaddingMask = true,
        ConditionDim = 24,
        EmbeddedTimestepDim = 8,
        AdaLnLoraDim = 4,
        RmsNormEps = 1e-6f,
        QkNormEps = 1e-6f,
        LlmAdapter = new AnimaLlmAdapterConfig
        {
            HiddenSize = TextWidth,
            NumHeads = 2,
            HeadDim = 4,
            FfnHiddenSize = 16,
            NumLayers = 0,
            CodebookVocab = 32,
            RmsNormEps = 1e-6f,
            QkNormEps = 1e-6f,
        },
    };

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Forward_CpuAndCudaMatch_OnBatchedRectangularGrid_WithoutIntermediateD2h()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        AnimaConfig config = TinyConfig;
        Dictionary<string, Tensor> weights = BuildWeights(config);
        try
        {
            using AnimaTransformer transformer = new(config);
            transformer.LoadWeights(weights);

            TensorShape latentShape = new(Batch, config.InChannels, LatentHeight, LatentWidth);
            TensorShape textShape = new(Batch, TextSequence, TextWidth);
            float[] latentValues = RandomValues(checked((int)latentShape.ElementCount), seed: 1709, scale: 0.7f);
            float[] textValues = RandomValues(checked((int)textShape.ElementCount), seed: 1721, scale: 0.5f);

            using Tensor latentCpu = TensorFrom(latentValues, latentShape);
            using Tensor textCpu = TensorFrom(textValues, textShape);
            using CpuBackend cpu = new();
            using Tensor expected = transformer.Forward(cpu, latentCpu, timestep: 0.625f, textCpu);
            transformer.ReleaseDeviceCache(cpu);

            TensorShape expectedShape = new(Batch, config.OutChannels, LatentHeight, LatentWidth);
            Assert.Equal(expectedShape, expected.Shape);
            float[] expectedValues = Snapshot(expected);
            AssertAllFinite(expectedValues, "CPU output");
            AssertExact(latentValues, Snapshot(latentCpu), "CPU latent mutation");

            using Tensor latentHost = TensorFrom(latentValues, latentShape);
            using Tensor textHost = TensorFrom(textValues, textShape);
            using Tensor latentCuda = new(latentShape, DType.F32);
            using Tensor textCuda = new(textShape, DType.F32);
            using CudaBackend cuda = new(0, PtxDir());

            // Make both inputs device-authored before the forward. A host implementation of either boundary
            // must therefore increment the D2H counter instead of passing because its input started on the host.
            cuda.Scale(latentCuda, latentHost, 1.0f);
            cuda.Scale(textCuda, textHost, 1.0f);
            cuda.PreloadWeights(transformer.EnumerateWeights());
            cuda.Sync();
            cuda.ResetD2hSyncCount();

            using Tensor actual = transformer.Forward(cuda, latentCuda, timestep: 0.625f, textCuda);
            cuda.Sync();
            transformer.ReleaseDeviceCache(cuda);

            Assert.Equal(0, cuda.GetD2hSyncCount());
            Assert.Equal(expectedShape, actual.Shape);

            // The first intentional host read is the returned output. It must be the only readback so far.
            float[] actualValues = Snapshot(actual);
            Assert.Equal(1, cuda.GetD2hSyncCount());
            AssertAllFinite(actualValues, "CUDA output");
            // 5e-4 (was 3e-4): the driver JIT's fma contraction of the committed PTX shifts reduction results
            // slightly across driver versions (observed 3.2e-4 abs on 580.173 vs passing on the authoring
            // driver). Real boundary defects (layout/scale bugs) miss by orders of magnitude, not ppm.
            AssertClose(expectedValues, actualValues, absoluteTolerance: 5e-4f, relativeTolerance: 5e-4f);

            // Reading the device-authored input is intentional too and proves the boundary did not mutate it.
            AssertExact(latentValues, Snapshot(latentCuda), "CUDA latent mutation");
            Assert.Equal(2, cuda.GetD2hSyncCount());

            _output.WriteLine(
                $"Anima zero-block boundary: B={Batch}, latent={config.InChannels}x{LatentHeight}x{LatentWidth}, " +
                $"grid={LatentHeight / config.PatchSize.H}x{LatentWidth / config.PatchSize.W}, " +
                $"output={config.OutChannels}x{LatentHeight}x{LatentWidth}, intermediate D2H=0");
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void ConstructorAndForward_RejectInvalidImageBoundaryGeometry()
    {
        Assert.Throws<ArgumentException>(() => new AnimaTransformer(TinyConfig with
        {
            PatchSize = (2, 2, 2),
        }));
        Assert.Throws<ArgumentException>(() => new AnimaTransformer(TinyConfig with
        {
            PatchSize = (1, 0, 2),
        }));
        Assert.Throws<ArgumentException>(() => new AnimaTransformer(TinyConfig with
        {
            PatchSize = (1, 2, 1),
        }));

        using AnimaTransformer transformer = new(TinyConfig);
        using CpuBackend cpu = new();
        using Tensor validLatent = new(
            new TensorShape(Batch, TinyConfig.InChannels, LatentHeight, LatentWidth), DType.F32);
        using Tensor validText = new(new TensorShape(Batch, TextSequence, TextWidth), DType.F32);
        using Tensor rankThreeLatent = new(
            new TensorShape(Batch, TinyConfig.InChannels, LatentHeight * LatentWidth), DType.F32);
        using Tensor rankTwoText = new(new TensorShape(Batch, TextSequence * TextWidth), DType.F32);
        using Tensor unalignedLatent = new(
            new TensorShape(Batch, TinyConfig.InChannels, LatentHeight - 1, LatentWidth), DType.F32);
        using Tensor wrongBatchText = new(new TensorShape(Batch - 1, TextSequence, TextWidth), DType.F32);
        using Tensor wrongChannelLatent = new(
            new TensorShape(Batch, TinyConfig.InChannels + 1, LatentHeight, LatentWidth), DType.F32);
        using Tensor wrongWidthText = new(new TensorShape(Batch, TextSequence, TextWidth + 1), DType.F32);
        using Tensor halfLatent = new(validLatent.Shape, DType.F16);
        using Tensor halfText = new(validText.Shape, DType.F16);
        using Tensor emptyLatent = new(
            new TensorShape(Batch, TinyConfig.InChannels, 0, LatentWidth), DType.F32);

        Assert.Throws<ArgumentException>(() => transformer.Forward(cpu, rankThreeLatent, 0.5f, validText));
        Assert.Throws<ArgumentException>(() => transformer.Forward(cpu, validLatent, 0.5f, rankTwoText));
        Assert.Throws<ArgumentException>(() => transformer.Forward(cpu, unalignedLatent, 0.5f, validText));
        Assert.Throws<ArgumentException>(() => transformer.Forward(cpu, validLatent, 0.5f, wrongBatchText));
        Assert.Throws<ArgumentException>(() => transformer.Forward(cpu, wrongChannelLatent, 0.5f, validText));
        Assert.Throws<ArgumentException>(() => transformer.Forward(cpu, validLatent, 0.5f, wrongWidthText));
        Assert.Throws<NotSupportedException>(() => transformer.Forward(cpu, halfLatent, 0.5f, validText));
        Assert.Throws<NotSupportedException>(() => transformer.Forward(cpu, validLatent, 0.5f, halfText));
        Assert.Throws<ArgumentException>(() => transformer.Forward(cpu, emptyLatent, 0.5f, validText));
        Assert.Throws<ArgumentOutOfRangeException>(() => transformer.Forward(cpu, validLatent, float.NaN, validText));
        Assert.Throws<ArgumentOutOfRangeException>(() => transformer.Forward(cpu, validLatent, float.PositiveInfinity, validText));
    }

    private static Dictionary<string, Tensor> BuildWeights(AnimaConfig config)
    {
        int patchFeatures = (config.InChannels + (config.ConcatPaddingMask ? 1 : 0))
            * config.PatchSize.H * config.PatchSize.W;
        int finalFeatures = config.OutChannels * config.PatchSize.H * config.PatchSize.W;

        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal)
        {
            ["x_embedder.proj.1.weight"] = RandomTensor(
                new TensorShape(config.HiddenSize, patchFeatures), seed: 1801, scale: 0.16f),
            ["t_embedder.1.linear_1.weight"] = RandomTensor(
                new TensorShape(config.HiddenSize, config.HiddenSize), seed: 1811, scale: 0.12f),
            ["t_embedder.1.linear_2.weight"] = RandomTensor(
                new TensorShape(config.ConditionDim, config.HiddenSize), seed: 1823, scale: 0.10f),
            ["t_embedding_norm.weight"] = PositiveTensor(
                config.HiddenSize, seed: 1831, center: 1.0f, spread: 0.15f),
            ["final_layer.adaln_modulation.1.weight"] = RandomTensor(
                new TensorShape(config.AdaLnLoraDim, config.HiddenSize), seed: 1847, scale: 0.10f),
            ["final_layer.adaln_modulation.2.weight"] = RandomTensor(
                new TensorShape(2 * config.HiddenSize, config.AdaLnLoraDim), seed: 1861, scale: 0.08f),
            ["final_layer.linear.weight"] = RandomTensor(
                new TensorShape(finalFeatures, config.HiddenSize), seed: 1871, scale: 0.14f),
        };
        return weights;
    }

    private static Tensor RandomTensor(TensorShape shape, int seed, float scale) =>
        TensorFrom(RandomValues(checked((int)shape.ElementCount), seed, scale), shape);

    private static Tensor PositiveTensor(int count, int seed, float center, float spread)
    {
        float[] values = RandomValues(count, seed, spread);
        for (int i = 0; i < values.Length; i++) values[i] += center;
        return TensorFrom(values, new TensorShape(count));
    }

    private static Tensor TensorFrom(float[] values, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static float[] RandomValues(int count, int seed, float scale)
    {
        Random random = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < values.Length; i++)
            values[i] = ((float)random.NextDouble() * 2.0f - 1.0f) * scale;
        return values;
    }

    private static float[] Snapshot(Tensor tensor)
    {
        float[] values = new float[checked((int)tensor.ElementCount)];
        new ReadOnlySpan<float>((void*)tensor.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    private static void AssertAllFinite(float[] values, string label)
    {
        for (int i = 0; i < values.Length; i++)
            Assert.True(float.IsFinite(values[i]), $"{label}[{i}] is non-finite: {values[i]}");
    }

    private static void AssertClose(
        float[] expected,
        float[] actual,
        float absoluteTolerance,
        float relativeTolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            Assert.True(float.IsFinite(e), $"expected[{i}] is non-finite: {e}");
            Assert.True(float.IsFinite(a), $"actual[{i}] is non-finite: {a}");
            float tolerance = absoluteTolerance + relativeTolerance * MathF.Abs(e);
            Assert.True(MathF.Abs(e - a) <= tolerance,
                $"Mismatch at {i}: expected={e:R}, actual={a:R}, tolerance={tolerance:R}");
        }
    }

    private static void AssertExact(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
    }

    private static void DisposeAll(IReadOnlyDictionary<string, Tensor> tensors)
    {
        foreach (Tensor tensor in tensors.Values)
            tensor.Dispose();
    }
}
