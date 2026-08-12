using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// Guards Anima's transformer-owned GPU RoPE tables. The cached tensors are borrowed by every
/// self-attention block and must stay device-resident until the transformer phase boundary.
/// </summary>
[Collection("CudaSerial")]
public sealed unsafe class AnimaRopeGpuTableLifecycleTests
{
    private const int Frames = 1;
    private const int GridHeight = 3;
    private const int GridWidth = 5;
    private const int Sequence = Frames * GridHeight * GridWidth;
    private const int Heads = 2;
    private const int HeadDim = 12;
    private const long TableBytes = 2L * Sequence * HeadDim * sizeof(float);

    private readonly ITestOutputHelper _output;

    public AnimaRopeGpuTableLifecycleTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void GetOrCreateTables_ReusesOneUpload_AndApplyRopeMatchesCpuWithoutD2h()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        AnimaRope rope = CreateRope();
        using CudaBackend cuda = new(0, PtxDir());
        using IBackend cpu = new CpuBackend();
        (Tensor Cos, Tensor Sin) referenceTables = rope.BuildFreqs(Frames, GridHeight, GridWidth);
        using Tensor referenceCos = referenceTables.Cos;
        using Tensor referenceSin = referenceTables.Sin;
        using Tensor qExpected = Input(offset: 0.25f);
        using Tensor kExpected = Input(offset: -0.75f);
        using Tensor qHost = Input(offset: 0.25f);
        using Tensor kHost = Input(offset: -0.75f);
        using Tensor qFirst = new(qHost.Shape, DType.F32);
        using Tensor kFirst = new(kHost.Shape, DType.F32);
        using Tensor qSecond = new(qHost.Shape, DType.F32);
        using Tensor kSecond = new(kHost.Shape, DType.F32);

        cpu.ApplyRope(qExpected, kExpected, referenceCos, referenceSin);
        float[] expectedQ = Snapshot(qExpected);
        float[] expectedK = Snapshot(kExpected);

        cuda.Scale(qFirst, qHost, 1.0f);
        cuda.Scale(kFirst, kHost, 1.0f);
        cuda.Scale(qSecond, qHost, 1.0f);
        cuda.Scale(kSecond, kHost, 1.0f);
        cuda.Sync();
        cuda.ResetD2hSyncCount();

        try
        {
            long cachedBefore = cuda.GetGpuCacheStats().cachedBytes;
            (Tensor Cos, Tensor Sin) first =
                rope.GetOrCreateTables(cuda, Frames, GridHeight, GridWidth);
            long cachedAfterFirst = cuda.GetGpuCacheStats().cachedBytes;

            Assert.Equal(new TensorShape(Sequence, HeadDim), first.Cos.Shape);
            Assert.Equal(new TensorShape(Sequence, HeadDim), first.Sin.Shape);
            Assert.True(GpuTransferHelper.IsWeightCached(first.Cos));
            Assert.True(GpuTransferHelper.IsWeightCached(first.Sin));
            Assert.Equal(TableBytes, cachedAfterFirst - cachedBefore);

            cuda.ApplyRope(qFirst, kFirst, first.Cos, first.Sin);

            (Tensor Cos, Tensor Sin) second =
                rope.GetOrCreateTables(cuda, Frames, GridHeight, GridWidth);
            cuda.ApplyRope(qSecond, kSecond, second.Cos, second.Sin);
            cuda.Sync();

            Assert.Same(first.Cos, second.Cos);
            Assert.Same(first.Sin, second.Sin);
            Assert.Equal(cachedAfterFirst, cuda.GetGpuCacheStats().cachedBytes);
            Assert.Equal(0, cuda.GetD2hSyncCount());

            float[] firstQ = Snapshot(qFirst);
            float[] firstK = Snapshot(kFirst);
            float[] secondQ = Snapshot(qSecond);
            float[] secondK = Snapshot(kSecond);
            Assert.Equal(4, cuda.GetD2hSyncCount());

            AssertClose(expectedQ, firstQ, 2e-6f, "first Q");
            AssertClose(expectedK, firstK, 2e-6f, "first K");
            AssertExact(firstQ, secondQ, "reused Q");
            AssertExact(firstK, secondK, "reused K");

            _output.WriteLine(
                $"Anima RoPE cached one {Sequence}x{HeadDim} cos/sin pair ({TableBytes} bytes); " +
                "matching-geometry reuse added no upload and both rotations completed with intermediate D2H=0.");
        }
        finally
        {
            rope.ReleaseGpuTables(cuda);
        }
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void ReleaseDeviceCache_EvictsAndRecreatesEquivalentTables_IdempotentlyWithoutD2h()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using AnimaTransformer transformer = new(TinyConfig());
        AnimaRope rope = TransformerRope(transformer);
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor qHost = Input(offset: 1.25f);
        using Tensor kHost = Input(offset: -1.5f);
        using Tensor qFirst = new(qHost.Shape, DType.F32);
        using Tensor kFirst = new(kHost.Shape, DType.F32);
        using Tensor qSecond = new(qHost.Shape, DType.F32);
        using Tensor kSecond = new(kHost.Shape, DType.F32);

        cuda.Scale(qFirst, qHost, 1.0f);
        cuda.Scale(kFirst, kHost, 1.0f);
        cuda.Scale(qSecond, qHost, 1.0f);
        cuda.Scale(kSecond, kHost, 1.0f);
        cuda.Sync();
        cuda.ResetD2hSyncCount();

        try
        {
            (Tensor Cos, Tensor Sin) first =
                rope.GetOrCreateTables(cuda, Frames, GridHeight, GridWidth);
            cuda.ApplyRope(qFirst, kFirst, first.Cos, first.Sin);
            cuda.Sync();
            Assert.Equal(0, cuda.GetD2hSyncCount());

            float[] expectedQ = Snapshot(qFirst);
            float[] expectedK = Snapshot(kFirst);
            Assert.Equal(2, cuda.GetD2hSyncCount());

            long cachedBeforeRelease = cuda.GetGpuCacheStats().cachedBytes;
            cuda.ResetD2hSyncCount();
            transformer.ReleaseDeviceCache(cuda);

            long cachedAfterRelease = cuda.GetGpuCacheStats().cachedBytes;
            Assert.False(GpuTransferHelper.IsWeightCached(first.Cos));
            Assert.False(GpuTransferHelper.IsWeightCached(first.Sin));
            Assert.Equal(TableBytes, cachedBeforeRelease - cachedAfterRelease);
            Assert.Equal(0, cuda.GetD2hSyncCount());

            transformer.ReleaseDeviceCache(cuda);
            Assert.Equal(cachedAfterRelease, cuda.GetGpuCacheStats().cachedBytes);
            Assert.Equal(0, cuda.GetD2hSyncCount());

            cuda.ResetD2hSyncCount();

            (Tensor Cos, Tensor Sin) recreated =
                rope.GetOrCreateTables(cuda, Frames, GridHeight, GridWidth);
            Assert.NotSame(first.Cos, recreated.Cos);
            Assert.NotSame(first.Sin, recreated.Sin);
            Assert.True(GpuTransferHelper.IsWeightCached(recreated.Cos));
            Assert.True(GpuTransferHelper.IsWeightCached(recreated.Sin));
            Assert.Equal(TableBytes, cuda.GetGpuCacheStats().cachedBytes - cachedAfterRelease);

            cuda.ApplyRope(qSecond, kSecond, recreated.Cos, recreated.Sin);
            cuda.Sync();
            Assert.Equal(0, cuda.GetD2hSyncCount());

            AssertExact(expectedQ, Snapshot(qSecond), "recreated Q");
            AssertExact(expectedK, Snapshot(kSecond), "recreated K");
            Assert.Equal(2, cuda.GetD2hSyncCount());

            _output.WriteLine(
                $"Anima transformer released {TableBytes} cached RoPE bytes, tolerated a duplicate release, " +
                "and recreated bit-identical rotations with intermediate D2H=0.");
        }
        finally
        {
            transformer.ReleaseDeviceCache(cuda);
        }
    }

    private static AnimaRope CreateRope() =>
        new(HeadDim, theta: 10_000.0f, ropeScale: (2.0f, 1.25f, 0.75f));

    private static AnimaConfig TinyConfig() => new()
    {
        HiddenSize = Heads * HeadDim,
        NumHeads = Heads,
        HeadDim = HeadDim,
        NumLayers = 0,
        MlpRatio = 2.0f,
        InChannels = 2,
        OutChannels = 3,
        PatchSize = (1, 2, 2),
        MaxSize = (Frames, GridHeight, GridWidth),
        RopeScale = (2.0f, 1.25f, 0.75f),
        RopeTheta = 10_000.0f,
        ConcatPaddingMask = true,
        ConditionDim = 3 * Heads * HeadDim,
        EmbeddedTimestepDim = Heads * HeadDim,
        AdaLnLoraDim = 4,
        RmsNormEps = 1e-6f,
        QkNormEps = 1e-6f,
        LlmAdapter = new AnimaLlmAdapterConfig
        {
            HiddenSize = 8,
            NumHeads = 2,
            HeadDim = 4,
            FfnHiddenSize = 16,
            NumLayers = 0,
            CodebookVocab = 32,
            RmsNormEps = 1e-6f,
            QkNormEps = 1e-6f,
        },
    };

    private static AnimaRope TransformerRope(AnimaTransformer transformer)
    {
        FieldInfo field = typeof(AnimaTransformer).GetField(
            "_rope", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("Missing AnimaTransformer._rope.");
        return Assert.IsType<AnimaRope>(field.GetValue(transformer));
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(
                HartsyInference.Tests.Common.RepoRoot.Path,
                "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Input(float offset)
    {
        Tensor tensor = new(new TensorShape(1, Sequence, Heads, HeadDim), DType.F32);
        float* values = (float*)tensor.DataPointer;
        for (long i = 0; i < tensor.ElementCount; i++)
            values[i] = offset + (float)((i * 17 + 5) % 97) * 0.015625f;
        return tensor;
    }

    private static float[] Snapshot(Tensor tensor)
    {
        float[] result = new float[checked((int)tensor.ElementCount)];
        new ReadOnlySpan<float>((void*)tensor.DataPointer, result.Length).CopyTo(result);
        return result;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            Assert.True(float.IsFinite(e), $"{label}: expected[{i}] is non-finite: {e}");
            Assert.True(float.IsFinite(a), $"{label}: actual[{i}] is non-finite: {a}");
            Assert.True(MathF.Abs(e - a) <= tolerance,
                $"{label}[{i}]: expected={e:R}, actual={a:R}, tolerance={tolerance:R}");
        }
    }

    private static void AssertExact(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            Assert.True(float.IsFinite(e), $"{label}: expected[{i}] is non-finite: {e}");
            Assert.True(float.IsFinite(a), $"{label}: actual[{i}] is non-finite: {a}");
            Assert.Equal(BitConverter.SingleToInt32Bits(e), BitConverter.SingleToInt32Bits(a));
        }
    }
}
