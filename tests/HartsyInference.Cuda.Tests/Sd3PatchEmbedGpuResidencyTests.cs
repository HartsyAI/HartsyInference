using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// Independent scalar-parity, device-residency, contract, and derived-weight ownership gates for SD3 PatchEmbed.
/// </summary>
[Collection("CudaSerial")]
public sealed unsafe class Sd3PatchEmbedGpuResidencyTests
{
    private readonly ITestOutputHelper _output;

    public Sd3PatchEmbedGpuResidencyTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    [Trait("Category", "GpuIntegration")]
    public void Forward_MatchesIndependentScalarReference_WithoutIntermediateD2h(
        bool includePositionEmbedding, bool useF16PositionEmbedding)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int batch = 2;
        const int inChannels = 3;
        const int height = 6;
        const int width = 10;
        const int patchSize = 2;
        const int embedDim = 5;
        const int maxGrid = 7;

        float[] inputValues = RandomValues(batch * inChannels * height * width, 1301, 0.7f);
        float[] weightValues = RandomValues(embedDim * inChannels * patchSize * patchSize, 1303, 0.25f);
        float[] biasValues = RandomValues(embedDim, 1307, 0.1f);
        float[]? positionValues = includePositionEmbedding
            ? RandomValues(maxGrid * maxGrid * embedDim, 1319, 0.2f)
            : null;
        float[]? referencePositionValues = positionValues;
        if (positionValues is not null && useF16PositionEmbedding)
        {
            referencePositionValues = new float[positionValues.Length];
            for (int i = 0; i < positionValues.Length; i++)
                referencePositionValues[i] = (float)(Half)positionValues[i];
        }
        float[] expected = ScalarReference(
            inputValues, weightValues, biasValues, referencePositionValues,
            batch, inChannels, height, width, patchSize, embedDim, maxGrid);

        using Tensor input = TensorFrom(inputValues, new TensorShape(batch, inChannels, height, width));
        using Tensor weight = TensorFrom(weightValues, new TensorShape(embedDim, inChannels, patchSize, patchSize));
        using Tensor bias = TensorFrom(biasValues, new TensorShape(embedDim));
        using Tensor? position = positionValues is null
            ? null
            : useF16PositionEmbedding
                ? HalfTensor(positionValues, new TensorShape(1, maxGrid * maxGrid, embedDim))
                : TensorFrom(positionValues, new TensorShape(1, maxGrid * maxGrid, embedDim));
        using CudaBackend cuda = new(0, PtxDir());
        using PatchEmbed layer = new(patchSize, inChannels, embedDim);
        layer.LoadWeights(weight, bias, position);

        cuda.PreloadWeights(layer.EnumerateWeights());
        cuda.Sync();
        cuda.ResetD2hSyncCount();

        using Tensor output = layer.Forward(cuda, input);
        cuda.Sync();

        Assert.Equal(0, cuda.GetD2hSyncCount());
        Assert.Equal(new TensorShape(batch, height / patchSize * (width / patchSize), embedDim), output.Shape);

        float[] actual = Snapshot(output);
        Assert.Equal(1, cuda.GetD2hSyncCount());
        AssertClose(expected, actual, 2e-5f);
        _output.WriteLine(
            $"pos={includePositionEmbedding}, posF16={useF16PositionEmbedding}: " +
            $"{actual.Length} elements, intermediate D2H=0");
    }

    [Fact]
    public void ConstructorLoadForwardAndGrid_RejectMalformedContractsBeforeBackendWork()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PatchEmbed(0, 3, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PatchEmbed(2, 0, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PatchEmbed(2, 3, 0));

        using PatchEmbed layer = new(2, 3, 5);
        using CpuBackend cpu = new();
        using Tensor validInput = new(new TensorShape(1, 3, 6, 6), DType.F32);
        Assert.Throws<InvalidOperationException>(() => layer.Forward(cpu, validInput));

        using Tensor weight = new(new TensorShape(5, 3, 2, 2), DType.F32);
        using Tensor bias = new(new TensorShape(5), DType.F32);
        using Tensor position = new(new TensorShape(1, 9, 5), DType.F32);
        layer.LoadWeights(weight, bias, position);

        using Tensor rankThree = new(new TensorShape(1, 3, 36), DType.F32);
        using Tensor wrongDtype = new(new TensorShape(1, 3, 6, 6), DType.F16);
        using Tensor wrongChannels = new(new TensorShape(1, 4, 6, 6), DType.F32);
        using Tensor unalignedHeight = new(new TensorShape(1, 3, 5, 6), DType.F32);
        using Tensor gridTooLarge = new(new TensorShape(1, 3, 8, 6), DType.F32);
        Assert.Throws<ArgumentException>(() => layer.Forward(cpu, rankThree));
        Assert.Throws<NotSupportedException>(() => layer.Forward(cpu, wrongDtype));
        Assert.Throws<ArgumentException>(() => layer.Forward(cpu, wrongChannels));
        Assert.Throws<ArgumentException>(() => layer.Forward(cpu, unalignedHeight));
        Assert.Throws<ArgumentException>(() => layer.Forward(cpu, gridTooLarge));

        Assert.Throws<ArgumentOutOfRangeException>(() => layer.GetGridSize(0, 6));
        Assert.Throws<ArgumentException>(() => layer.GetGridSize(5, 6));
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.GetGridSize(8, 6));
    }

    [Fact]
    public void LoadWeights_RejectsMalformedTensorsWithoutReplacingValidState()
    {
        using PatchEmbed layer = new(2, 3, 5);
        using Tensor weight = new(new TensorShape(5, 3, 2, 2), DType.F32);
        using Tensor bias = new(new TensorShape(5), DType.F32);
        using Tensor position = new(new TensorShape(1, 9, 5), DType.F32);
        layer.LoadWeights(weight, bias, position);
        Tensor[] validSnapshot = layer.EnumerateWeights().ToArray();

        using Tensor wrongWeightShape = new(new TensorShape(5, 3, 1, 2), DType.F32);
        using Tensor wrongWeightDtype = new(new TensorShape(5, 3, 2, 2), DType.I32);
        using Tensor wrongBiasShape = new(new TensorShape(1, 5), DType.F32);
        using Tensor wrongBiasDtype = new(new TensorShape(5), DType.I32);
        using Tensor wrongPositionRank = new(new TensorShape(9, 5), DType.F32);
        using Tensor nonsquarePosition = new(new TensorShape(1, 8, 5), DType.F32);
        using Tensor wrongPositionWidth = new(new TensorShape(1, 9, 4), DType.F32);
        using Tensor wrongPositionDtype = new(new TensorShape(1, 9, 5), DType.I32);

        Assert.Throws<ArgumentException>(() => layer.LoadWeights(wrongWeightShape, bias, position));
        Assert.Throws<NotSupportedException>(() => layer.LoadWeights(wrongWeightDtype, bias, position));
        Assert.Throws<ArgumentException>(() => layer.LoadWeights(weight, wrongBiasShape, position));
        Assert.Throws<NotSupportedException>(() => layer.LoadWeights(weight, wrongBiasDtype, position));
        Assert.Throws<ArgumentException>(() => layer.LoadWeights(weight, bias, wrongPositionRank));
        Assert.Throws<ArgumentException>(() => layer.LoadWeights(weight, bias, nonsquarePosition));
        Assert.Throws<ArgumentException>(() => layer.LoadWeights(weight, bias, wrongPositionWidth));
        Assert.Throws<NotSupportedException>(() => layer.LoadWeights(weight, bias, wrongPositionDtype));

        Tensor[] afterFailures = layer.EnumerateWeights().ToArray();
        Assert.Equal(validSnapshot.Length, afterFailures.Length);
        for (int i = 0; i < validSnapshot.Length; i++)
            Assert.Same(validSnapshot[i], afterFailures[i]);
    }

    [Fact]
    public void ConvertedPositionEmbedding_HasExactReloadAndDisposeOwnership()
    {
        using Tensor weight = new(new TensorShape(5, 3, 2, 2), DType.F32);
        using Tensor bias = new(new TensorShape(5), DType.F32);
        using Tensor firstSource = HalfTensor(new TensorShape(1, 9, 5), 1409);
        using Tensor secondSource = HalfTensor(new TensorShape(1, 9, 5), 1423);
        using Tensor borrowedF32 = new(new TensorShape(1, 9, 5), DType.F32);
        using PatchEmbed layer = new(2, 3, 5);

        layer.LoadWeights(weight, bias, firstSource);
        Tensor firstDerived = layer.EnumerateWeights().Last();
        Assert.NotSame(firstSource, firstDerived);
        Assert.Equal(DType.F32, firstDerived.DType);

        layer.LoadWeights(weight, bias, firstSource);
        Assert.Same(firstDerived, layer.EnumerateWeights().Last());

        // Preload callers see the derived tensor, so passing that snapshot back through LoadWeights must not
        // turn the layer's owned tensor into a borrowed-then-disposed self-alias.
        layer.LoadWeights(weight, bias, firstDerived);
        Assert.Same(firstDerived, layer.EnumerateWeights().Last());
        Touch(firstDerived);

        layer.LoadWeights(weight, bias, secondSource);
        Tensor secondDerived = layer.EnumerateWeights().Last();
        Assert.NotSame(firstDerived, secondDerived);
        AssertDisposed(firstDerived);
        Touch(firstSource);

        layer.LoadWeights(weight, bias, borrowedF32);
        Assert.Same(borrowedF32, layer.EnumerateWeights().Last());
        AssertDisposed(secondDerived);

        layer.Dispose();
        layer.Dispose();
        Touch(weight);
        Touch(bias);
        Touch(firstSource);
        Touch(secondSource);
        Touch(borrowedF32);
        Assert.Throws<ObjectDisposedException>(() => layer.EnumerateWeights().ToArray());
    }

    private static float[] ScalarReference(
        float[] input,
        float[] weight,
        float[] bias,
        float[]? position,
        int batch,
        int inChannels,
        int height,
        int width,
        int patchSize,
        int embedDim,
        int maxGrid)
    {
        int gridH = height / patchSize;
        int gridW = width / patchSize;
        int patches = gridH * gridW;
        int startH = (maxGrid - gridH) / 2;
        int startW = (maxGrid - gridW) / 2;
        float[] output = new float[batch * patches * embedDim];

        for (int b = 0; b < batch; b++)
        {
            for (int gh = 0; gh < gridH; gh++)
            {
                for (int gw = 0; gw < gridW; gw++)
                {
                    int patch = gh * gridW + gw;
                    for (int oc = 0; oc < embedDim; oc++)
                    {
                        float sum = bias[oc];
                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            for (int kh = 0; kh < patchSize; kh++)
                            {
                                for (int kw = 0; kw < patchSize; kw++)
                                {
                                    int inputIndex = ((b * inChannels + ic) * height + gh * patchSize + kh) * width
                                        + gw * patchSize + kw;
                                    int weightIndex = ((oc * inChannels + ic) * patchSize + kh) * patchSize + kw;
                                    sum += input[inputIndex] * weight[weightIndex];
                                }
                            }
                        }

                        if (position is not null)
                        {
                            int positionRow = (startH + gh) * maxGrid + startW + gw;
                            sum += position[positionRow * embedDim + oc];
                        }
                        output[(b * patches + patch) * embedDim + oc] = sum;
                    }
                }
            }
        }
        return output;
    }

    private static float[] RandomValues(int count, int seed, float scale)
    {
        Random random = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)((random.NextDouble() * 2.0 - 1.0) * scale);
        return values;
    }

    private static Tensor TensorFrom(float[] values, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        values.AsSpan().CopyTo(new Span<float>(tensor.DataPointer, values.Length));
        return tensor;
    }

    private static Tensor HalfTensor(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F16);
        Random random = new(seed);
        Half* values = (Half*)tensor.DataPointer;
        for (long i = 0; i < tensor.ElementCount; i++)
            values[i] = (Half)(random.NextDouble() * 0.4 - 0.2);
        return tensor;
    }

    private static Tensor HalfTensor(float[] values, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F16);
        Half* destination = (Half*)tensor.DataPointer;
        for (int i = 0; i < values.Length; i++)
            destination[i] = (Half)values[i];
        return tensor;
    }

    private static float[] Snapshot(Tensor tensor)
    {
        float[] values = new float[tensor.ElementCount];
        new ReadOnlySpan<float>(tensor.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxError = 0f;
        int maxIndex = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]),
                $"PatchEmbed non-finite value at {i}: expected={expected[i]:G9}, actual={actual[i]:G9}.");
            float error = MathF.Abs(expected[i] - actual[i]);
            Assert.True(float.IsFinite(error),
                $"PatchEmbed non-finite error at {i}: expected={expected[i]:G9}, actual={actual[i]:G9}.");
            if (error > maxError)
            {
                maxError = error;
                maxIndex = i;
            }
        }
        Assert.True(maxError <= tolerance,
            $"PatchEmbed max error {maxError:E6} at {maxIndex}: expected={expected[maxIndex]:G9}, actual={actual[maxIndex]:G9}, tolerance={tolerance:E2}.");
    }

    private static void AssertDisposed(Tensor tensor) =>
        Assert.Throws<ObjectDisposedException>(() => Touch(tensor));

    private static void Touch(Tensor tensor) => _ = *(byte*)tensor.DataPointer;
}
