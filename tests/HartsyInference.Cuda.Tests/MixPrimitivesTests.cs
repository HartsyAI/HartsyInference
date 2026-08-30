using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// Independent math, broadcast-indexing, batching, contract, input-preservation, and residency gates for the two
/// shared image scheduler/inpaint mix primitives. Packed cases deliberately use P=6 so no implementation can pass
/// by hardcoding the production-common 2x2 patch area.
/// </summary>
[Collection("CudaSerial")]
public sealed unsafe class MixPrimitivesTests
{
    private readonly ITestOutputHelper _output;

    public MixPrimitivesTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    public static TheoryData<MaskBroadcastLayout, TensorShape, TensorShape> MaskLayouts() => new()
    {
        { MaskBroadcastLayout.DenseNchwBroadcast, new TensorShape(2, 5, 3, 7), new TensorShape(2, 1, 3, 7) },
        { MaskBroadcastLayout.PackedChannelOuter, new TensorShape(3, 11, 30), new TensorShape(3, 11, 6) },
        { MaskBroadcastLayout.PackedChannelOuter, new TensorShape(11, 30), new TensorShape(11, 6) },
        { MaskBroadcastLayout.PackedChannelInner, new TensorShape(3, 11, 30), new TensorShape(3, 11, 6) },
        { MaskBroadcastLayout.PackedChannelInner, new TensorShape(11, 30), new TensorShape(11, 6) },
        { MaskBroadcastLayout.Rows, new TensorShape(33, 30), new TensorShape(33) },
    };

    [Fact]
    public void Cpu_AffineMix_MatchesIndependentReference_AndPreservesInputs()
    {
        TensorShape shape = new(2, 3, 5, 7);
        float[] xValues = Values((int)shape.ElementCount, 101, 2.5f);
        float[] yValues = Values((int)shape.ElementCount, 103, 0.75f);
        const float xScale = -0.625f, yScale = 1.375f;
        float[] expected = AffineReference(xValues, yValues, xScale, yScale);

        using Tensor x = TensorFrom(xValues, shape);
        using Tensor y = TensorFrom(yValues, shape);
        using Tensor output = new(shape, DType.F32);
        using IBackend cpu = new CpuBackend();
        cpu.AffineMix(output, x, y, xScale, yScale);

        AssertClose(expected, Snapshot(output), 1e-6f, "CPU affine output");
        AssertExact(xValues, Snapshot(x), "CPU affine x mutation");
        AssertExact(yValues, Snapshot(y), "CPU affine y mutation");

        // Read-only inputs may alias: this is useful for scalar rescaling without manufacturing a second tensor.
        using Tensor aliasedOutput = new(shape, DType.F32);
        cpu.AffineMix(aliasedOutput, x, x, -2f, 0.5f);
        AssertClose(xValues.Select(static value => -1.5f * value).ToArray(), Snapshot(aliasedOutput), 1e-6f,
            "CPU affine read-only input alias");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_AffineMix_MatchesIndependentReference_PreservesInputs_AndStaysResident()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        TensorShape shape = new(2, 9, 17, 19);
        float[] xValues = Values((int)shape.ElementCount, 107, 1.5f);
        float[] yValues = Values((int)shape.ElementCount, 109, 3f);
        const float xScale = 0.3125f, yScale = -1.75f;
        float[] expected = AffineReference(xValues, yValues, xScale, yScale);

        using Tensor xHost = TensorFrom(xValues, shape);
        using Tensor yHost = TensorFrom(yValues, shape);
        using Tensor x = new(shape, DType.F32);
        using Tensor y = new(shape, DType.F32);
        using Tensor output = new(shape, DType.F32);
        using CudaBackend cuda = new(0, PtxDir());
        cuda.Scale(x, xHost, 1f);
        cuda.Scale(y, yHost, 1f);
        cuda.Sync();
        cuda.ResetD2hSyncCount();

        cuda.AffineMix(output, x, y, xScale, yScale);
        cuda.Sync();
        Assert.Equal(0, cuda.GetD2hSyncCount());

        AssertClose(expected, Snapshot(output), 2e-6f, "CUDA affine output");
        AssertExact(xValues, Snapshot(x), "CUDA affine x mutation");
        AssertExact(yValues, Snapshot(y), "CUDA affine y mutation");
    }

    [Theory]
    [MemberData(nameof(MaskLayouts))]
    public void Cpu_MaskedAffineMix_MatchesIndependentLayoutOracle_WithoutClamping(
        MaskBroadcastLayout layout, TensorShape targetShape, TensorShape maskShape)
    {
        using IBackend cpu = new CpuBackend();
        RunMaskedCase(cpu, null, layout, targetShape, maskShape, withNoise: true);
        RunMaskedCase(cpu, null, layout, targetShape, maskShape, withNoise: false);
    }

    [Fact]
    public void RowMaskSupportsRepeatedAffineReplacementWithoutMutatingInputs()
    {
        TensorShape shape = new TensorShape(3, 2);
        using Tensor target = TensorFrom([10f, 11f, 20f, 21f, 30f, 31f], shape);
        using Tensor source = TensorFrom([2f, 4f, 6f, 8f, 10f, 12f], shape);
        using Tensor fixedNoise = TensorFrom([-2f, -4f, -6f, -8f, -10f, -12f], shape);
        using Tensor mask = TensorFrom([0f, 0.5f, 1f], new TensorShape(3));
        using IBackend cpu = new CpuBackend();

        cpu.MaskedAffineMixInPlace(target, source, fixedNoise, mask,
            sourceScale: 0.25f, noiseScale: 0.75f, layout: MaskBroadcastLayout.Rows);
        AssertExact([-1f, -2f, 8.5f, 8.5f, 30f, 31f], Snapshot(target), "first row mix");

        // Reusing the same source/noise inputs with new coefficients must remain deterministic. Gray stays a
        // continuous blend and white keeps the target state, while the final noise-free replacement is also valid.
        cpu.MaskedAffineMixInPlace(target, source, fixedNoise, mask,
            sourceScale: 0.75f, noiseScale: 0.25f, layout: MaskBroadcastLayout.Rows);
        AssertExact([1f, 2f, 5.75f, 6.25f, 30f, 31f], Snapshot(target), "second row mix");

        cpu.MaskedAffineMixInPlace(target, source, null, mask,
            sourceScale: 1f, noiseScale: 0f, layout: MaskBroadcastLayout.Rows);
        AssertExact([2f, 4f, 5.875f, 7.125f, 30f, 31f], Snapshot(target), "noise-free row mix");
    }

    [Theory]
    [MemberData(nameof(MaskLayouts))]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_MaskedAffineMix_MatchesIndependentLayoutOracle_PreservesInputs_AndStaysResident(
        MaskBroadcastLayout layout, TensorShape targetShape, TensorShape maskShape)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend cuda = new(0, PtxDir());
        RunMaskedCase(cuda, cuda, layout, targetShape, maskShape, withNoise: true);
        RunMaskedCase(cuda, cuda, layout, targetShape, maskShape, withNoise: false);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_TwoStepMaskedEuler_RebindsAlternatingScratchWithoutHostSync()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int rows = 2, features = 8, patchArea = 4;
        TensorShape shape = new(rows, features);
        float[] initial = Enumerable.Range(1, rows * features).Select(static value => (float)value).ToArray();
        float[] sourceValues = Enumerable.Range(0, rows * features).Select(static i => 20f - i).ToArray();
        float[] injectionValues = Enumerable.Range(0, rows * features).Select(static i => i - 4f).ToArray();
        float[] velocityValues = Enumerable.Range(0, rows * features)
            .Select(static i => (i + 1) * (i % 2 == 0 ? 0.5f : -0.5f)).ToArray();
        float[] tokenValues = [0.5f, 1f];
        float[] rawValues = [0f, 0.2f, 0.4f, 0.5f, 0.1f, 0.4f, 0.8f, 1f];
        (float Current, float Next)[] schedule = [(0.8f, 0.4f), (0.4f, 0.1f)];

        float[] expected = initial.ToArray();
        foreach ((float current, float next) in schedule)
        {
            float[] nextValues = new float[expected.Length];
            float stateStrength = next / current;
            for (int i = 0; i < expected.Length; i++)
            {
                int row = i / features;
                float q = tokenValues[row] * expected[i] + (1f - tokenValues[row]) * injectionValues[i];
                float dModel = q + current * velocityValues[i];
                float raw = rawValues[row * patchArea + i % patchArea];
                float denoised = raw * dModel + (1f - raw) * sourceValues[i];
                nextValues[i] = stateStrength * expected[i] + (1f - stateStrength) * denoised;
            }
            expected = nextValues;
        }

        using Tensor initialHost = TensorFrom(initial, shape);
        using Tensor sourceHost = TensorFrom(sourceValues, shape);
        using Tensor injectionHost = TensorFrom(injectionValues, shape);
        using Tensor velocityHost = TensorFrom(velocityValues, shape);
        using Tensor tokenHost = TensorFrom(tokenValues, new TensorShape(rows));
        using Tensor rawHost = TensorFrom(rawValues, new TensorShape(rows, patchArea));
        using Tensor stateA = new(shape, DType.F32);
        using Tensor stateB = new(shape, DType.F32);
        using Tensor source = new(shape, DType.F32);
        using Tensor injection = new(shape, DType.F32);
        using Tensor velocity = new(shape, DType.F32);
        using Tensor tokenMask = new(new TensorShape(rows), DType.F32);
        using Tensor rawMask = new(new TensorShape(rows, patchArea), DType.F32);
        using Tensor denoisedScratch = new(shape, DType.F32);
        using CudaBackend cuda = new(0, PtxDir());
        cuda.Scale(stateA, initialHost, 1f);
        cuda.Scale(source, sourceHost, 1f);
        cuda.Scale(injection, injectionHost, 1f);
        cuda.Scale(velocity, velocityHost, 1f);
        cuda.Scale(tokenMask, tokenHost, 1f);
        cuda.Scale(rawMask, rawHost, 1f);
        cuda.Sync();
        cuda.ResetD2hSyncCount();

        Tensor state = stateA;
        Tensor modelScratch = stateB;
        foreach ((float current, float next) in schedule)
        {
            cuda.Scale(modelScratch, state, 1f);
            cuda.MaskedAffineMixInPlace(
                modelScratch, injection, null, tokenMask, 1f, 0f, MaskBroadcastLayout.Rows);
            cuda.AffineMix(denoisedScratch, modelScratch, velocity, 1f, current);
            cuda.MaskedAffineMixInPlace(
                denoisedScratch, source, null, rawMask, 1f, 0f, MaskBroadcastLayout.PackedChannelOuter);
            float stateStrength = next / current;
            cuda.AffineMix(modelScratch, state, denoisedScratch, stateStrength, 1f - stateStrength);
            (state, modelScratch) = (modelScratch, state);
        }
        cuda.Sync();

        Assert.Equal(0, cuda.GetD2hSyncCount());
        AssertClose(expected, Snapshot(state), 3e-6f, "CUDA two-step masked Euler state");
    }

    [Fact]
    public void Cpu_MalformedContractsAndOverlappingStorage_AreRejectedBeforeDataAccess()
    {
        using IBackend cpu = new CpuBackend();
        AssertMalformedContracts(cpu);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_MalformedContractsAndOverlappingStorage_AreRejectedBeforeDispatch()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }
        using CudaBackend cuda = new(0, PtxDir());
        AssertMalformedContracts(cuda);
    }

    private static void RunMaskedCase(
        IBackend backend,
        CudaBackend? cuda,
        MaskBroadcastLayout layout,
        TensorShape targetShape,
        TensorShape maskShape,
        bool withNoise)
    {
        int count = checked((int)targetShape.ElementCount);
        int maskCount = checked((int)maskShape.ElementCount);
        float[] initial = Values(count, 211 + (int)layout * 17, 2f);
        float[] sourceValues = Values(count, 223 + (int)layout * 17, 1.25f);
        float[] noiseValues = Values(count, 227 + (int)layout * 17, 0.5f);
        // Includes values below zero and above one: the primitive is linear algebra, not mask preprocessing.
        float[] maskValues = Enumerable.Range(0, maskCount).Select(i => (i % 11 - 2) / 7f).ToArray();
        const float sourceScale = 0.6875f;
        float noiseScale = withNoise ? -0.375f : 0f;
        float[] expected = MaskedReference(
            initial, sourceValues, withNoise ? noiseValues : null, maskValues,
            targetShape, maskShape, sourceScale, noiseScale, layout);

        using Tensor initialHost = TensorFrom(initial, targetShape);
        using Tensor sourceHost = TensorFrom(sourceValues, targetShape);
        using Tensor noiseHost = TensorFrom(noiseValues, targetShape);
        using Tensor maskHost = TensorFrom(maskValues, maskShape);
        using Tensor target = cuda is null ? TensorFrom(initial, targetShape) : new(targetShape, DType.F32);
        using Tensor source = cuda is null ? TensorFrom(sourceValues, targetShape) : new(targetShape, DType.F32);
        using Tensor noise = cuda is null ? TensorFrom(noiseValues, targetShape) : new(targetShape, DType.F32);
        using Tensor mask = cuda is null ? TensorFrom(maskValues, maskShape) : new(maskShape, DType.F32);

        if (cuda is not null)
        {
            cuda.Scale(target, initialHost, 1f);
            cuda.Scale(source, sourceHost, 1f);
            cuda.Scale(noise, noiseHost, 1f);
            cuda.Scale(mask, maskHost, 1f);
            cuda.Sync();
            cuda.ResetD2hSyncCount();
        }

        backend.MaskedAffineMixInPlace(
            target, source, withNoise ? noise : null, mask, sourceScale, noiseScale, layout);
        cuda?.Sync();
        if (cuda is not null) Assert.Equal(0, cuda.GetD2hSyncCount());

        AssertClose(expected, Snapshot(target), 3e-6f, $"{backend.GetType().Name} {layout} output");
        AssertExact(sourceValues, Snapshot(source), $"{backend.GetType().Name} {layout} source mutation");
        if (withNoise)
            AssertExact(noiseValues, Snapshot(noise), $"{backend.GetType().Name} {layout} noise mutation");
        AssertExact(maskValues, Snapshot(mask), $"{backend.GetType().Name} {layout} mask mutation");

        if (cuda is not null)
            Assert.Equal(withNoise ? 4 : 3, cuda.GetD2hSyncCount());
    }

    private static void AssertMalformedContracts(IBackend backend)
    {
        using Tensor affine = new(new TensorShape(2, 3), DType.F32);
        using Tensor affineOther = new(new TensorShape(2, 3), DType.F32);
        using Tensor affineMismatch = new(new TensorShape(3, 2), DType.F32);
        using Tensor affineF16 = new(new TensorShape(2, 3), DType.F16);
        using Tensor affineEmpty = new(new TensorShape(2, 0), DType.F32);
        Assert.Throws<ArgumentException>(() => backend.AffineMix(affine, affine, affineOther, 1f, 1f));
        Assert.Throws<ArgumentException>(() => backend.AffineMix(affine, affineOther, affine, 1f, 1f));
        Assert.Throws<ArgumentException>(() => backend.AffineMix(affine, affineOther, affineMismatch, 1f, 1f));
        Assert.Throws<NotSupportedException>(() => backend.AffineMix(affineF16, affineF16, affineF16, 1f, 1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.AffineMix(affineEmpty, affineEmpty, affineEmpty, 1f, 1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.AffineMix(affine, affineOther, affineOther, float.NaN, 1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.AffineMix(affine, affineOther, affineOther, 1f, float.PositiveInfinity));

        using Tensor dense = new(new TensorShape(2, 3, 4, 5), DType.F32);
        using Tensor denseSource = new(dense.Shape, DType.F32);
        using Tensor denseNoise = new(dense.Shape, DType.F32);
        using Tensor denseMask = new(new TensorShape(2, 1, 4, 5), DType.F32);
        using Tensor wrongDenseSource = new(new TensorShape(1, 3, 4, 5), DType.F32);
        using Tensor wrongDenseMask = new(new TensorShape(1, 1, 4, 5), DType.F32);
        using Tensor rank3Mask = new(new TensorShape(2, 4, 5), DType.F32);
        using Tensor f16Mask = new(denseMask.Shape, DType.F16);
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            dense, wrongDenseSource, null, denseMask, 1f, 0f, MaskBroadcastLayout.DenseNchwBroadcast));
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            dense, denseSource, null, wrongDenseMask, 1f, 0f, MaskBroadcastLayout.DenseNchwBroadcast));
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            dense, denseSource, null, rank3Mask, 1f, 0f, MaskBroadcastLayout.DenseNchwBroadcast));
        Assert.Throws<NotSupportedException>(() => backend.MaskedAffineMixInPlace(
            dense, denseSource, null, f16Mask, 1f, 0f, MaskBroadcastLayout.DenseNchwBroadcast));
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            dense, denseSource, null, denseMask, 1f, 0.25f, MaskBroadcastLayout.DenseNchwBroadcast));
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            dense, denseSource, denseNoise, denseMask, 1f, 0f, MaskBroadcastLayout.DenseNchwBroadcast));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.MaskedAffineMixInPlace(
            dense, denseSource, null, denseMask, float.NaN, 0f, MaskBroadcastLayout.DenseNchwBroadcast));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.MaskedAffineMixInPlace(
            dense, denseSource, denseNoise, denseMask, 1f, float.NegativeInfinity, MaskBroadcastLayout.DenseNchwBroadcast));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.MaskedAffineMixInPlace(
            dense, denseSource, null, denseMask, 1f, 0f, (MaskBroadcastLayout)999));
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            dense, dense, null, denseMask, 1f, 0f, MaskBroadcastLayout.DenseNchwBroadcast));
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            dense, denseSource, dense, denseMask, 1f, 1f, MaskBroadcastLayout.DenseNchwBroadcast));

        using Tensor packed = new(new TensorShape(2, 7, 30), DType.F32);
        using Tensor packedSource = new(packed.Shape, DType.F32);
        using Tensor packedMask = new(new TensorShape(2, 7, 6), DType.F32);
        using Tensor nonDivisible = new(new TensorShape(2, 7, 31), DType.F32);
        using Tensor nonDivisibleSource = new(nonDivisible.Shape, DType.F32);
        using Tensor wrongPackedMask = new(new TensorShape(2, 6, 6), DType.F32);
        using Tensor rank2Packed = new(new TensorShape(7, 30), DType.F32);
        using Tensor rank2PackedSource = new(rank2Packed.Shape, DType.F32);
        using Tensor rank2PackedMask = new(new TensorShape(7, 6), DType.F32);
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            packed, packedSource, null, wrongPackedMask, 1f, 0f, MaskBroadcastLayout.PackedChannelOuter));
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            nonDivisible, nonDivisibleSource, null, packedMask, 1f, 0f, MaskBroadcastLayout.PackedChannelInner));
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            rank2Packed, rank2PackedSource, null, packedMask, 1f, 0f, MaskBroadcastLayout.PackedChannelOuter));
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            packed, packedSource, null, rank2PackedMask, 1f, 0f, MaskBroadcastLayout.PackedChannelInner));
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            dense, denseSource, null, denseMask, 1f, 0f, MaskBroadcastLayout.PackedChannelOuter));

        using Tensor rowTarget = new(new TensorShape(7, 30), DType.F32);
        using Tensor rowSource = new(rowTarget.Shape, DType.F32);
        using Tensor rowMask = new(new TensorShape(7), DType.F32);
        using Tensor wrongRowMask = new(new TensorShape(6), DType.F32);
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            rowTarget, rowSource, null, wrongRowMask, 1f, 0f, MaskBroadcastLayout.Rows));
        Assert.Throws<ArgumentException>(() => backend.MaskedAffineMixInPlace(
            packed, packedSource, null, rowMask, 1f, 0f, MaskBroadcastLayout.Rows));

        // Reshape and borrowed overlaps are separate Tensor objects but share storage. Validation must catch both
        // without consulting DataPointer and accidentally draining a resident target.
        using Tensor owner = TensorFrom(Values(32, 401, 1f), new TensorShape(32));
        using Tensor ownerView = owner.Reshape(new TensorShape(32));
        using Tensor independent = TensorFrom(Values(32, 409, 1f), new TensorShape(32));
        Assert.Throws<ArgumentException>(() => backend.AffineMix(ownerView, owner, independent, 1f, 1f));

        float* ownerPointer = (float*)owner.DataPointer;
        using Tensor overlapTarget = new(ownerPointer, new TensorShape(16), DType.F32);
        using Tensor overlapSource = new(ownerPointer + 8, new TensorShape(16), DType.F32);
        Assert.Throws<ArgumentException>(() => backend.AffineMix(overlapTarget, overlapSource, overlapSource, 1f, 1f));

        // Tiny borrowed sentinels prove geometry/byte-span overflow is rejected before allocation or dereference.
        using Tensor overflow = Borrowed(new TensorShape(long.MaxValue, 2), 2);
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.AffineMix(overflow, overflow, overflow, 1f, 1f));
        using Tensor byteSpanOverflow = Borrowed(new TensorShape(long.MaxValue / 2), 3);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            backend.AffineMix(byteSpanOverflow, byteSpanOverflow, byteSpanOverflow, 1f, 1f));
    }

    private static float[] AffineReference(float[] x, float[] y, float xScale, float yScale)
    {
        float[] output = new float[x.Length];
        for (int i = 0; i < output.Length; i++)
            output[i] = xScale * x[i] + yScale * y[i];
        return output;
    }

    private static float[] MaskedReference(
        float[] initial,
        float[] source,
        float[]? noise,
        float[] mask,
        TensorShape targetShape,
        TensorShape maskShape,
        float sourceScale,
        float noiseScale,
        MaskBroadcastLayout layout)
    {
        float[] output = (float[])initial.Clone();
        long spatial = layout == MaskBroadcastLayout.DenseNchwBroadcast
            ? targetShape[2] * targetShape[3]
            : 0;
        long batchPlane = layout == MaskBroadcastLayout.DenseNchwBroadcast
            ? targetShape[1] * spatial
            : 0;
        long featureDimension = layout switch
        {
            MaskBroadcastLayout.DenseNchwBroadcast => 0,
            MaskBroadcastLayout.Rows => targetShape[1],
            _ => targetShape[targetShape.Rank - 1],
        };
        long patchArea = layout switch
        {
            MaskBroadcastLayout.DenseNchwBroadcast => 0,
            MaskBroadcastLayout.Rows => 1,
            _ => maskShape[maskShape.Rank - 1],
        };
        long channels = layout == MaskBroadcastLayout.DenseNchwBroadcast ? 0 : featureDimension / patchArea;

        for (long i = 0; i < output.LongLength; i++)
        {
            long maskIndex;
            if (layout == MaskBroadcastLayout.DenseNchwBroadcast)
            {
                long batch = i / batchPlane;
                maskIndex = batch * spatial + i % spatial;
            }
            else
            {
                long feature = i % featureDimension;
                long token = i / featureDimension;
                long patchIndex = layout is MaskBroadcastLayout.PackedChannelOuter or MaskBroadcastLayout.Rows
                    ? feature % patchArea
                    : feature / channels;
                maskIndex = token * patchArea + patchIndex;
            }

            float replacement = sourceScale * source[i];
            if (noise is not null) replacement += noiseScale * noise[i];
            float maskValue = mask[maskIndex];
            output[i] = output[i] * maskValue + replacement * (1f - maskValue);
        }
        return output;
    }

    private static float[] Values(int count, int seed, float scale)
    {
        Random random = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = ((float)random.NextDouble() * 2f - 1f) * scale;
        return values;
    }

    private static Tensor TensorFrom(float[] values, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static Tensor Borrowed(TensorShape shape, nint pointer) =>
        new((void*)pointer, shape, DType.F32);

    private static float[] Snapshot(Tensor tensor) => tensor.AsReadOnlySpan<float>().ToArray();

    private static void AssertExact(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToUInt32Bits(expected[i]) == BitConverter.SingleToUInt32Bits(actual[i]),
                $"{label}: index {i}, expected {expected[i]:R}, got {actual[i]:R}");
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float error = MathF.Abs(expected[i] - actual[i]);
            Assert.True(error <= tolerance || error <= tolerance * MathF.Max(1f, MathF.Abs(expected[i])),
                $"{label}: index {i}, expected {expected[i]:R}, got {actual[i]:R}, error {error:R}");
        }
    }
}
