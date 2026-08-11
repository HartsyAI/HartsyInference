using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Independent parity, contract, input-preservation, and residency gates for Lumina-2's
/// last-dimension-normalized CFG plus flow-match Euler update.</summary>
[Collection("CudaSerial")]
public sealed unsafe class LuminaCfgNormalizedEulerTests
{
    private readonly ITestOutputHelper _output;

    public LuminaCfgNormalizedEulerTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    public static TheoryData<TensorShape, float, float> Shapes() => new()
    {
        // Lumina's unpatchified velocity layout: normalization reduces independently over W.
        { new TensorShape(1, 16, 64, 64), 4.0f, 0.0375f },
        // Batch-two rectangular layout and a multi-warp row tail.
        { new TensorShape(2, 3, 7, 257), 7.5f, 0.0125f },
        // More rows than the launch grid cap, proving the block-level row grid-stride loop.
        { new TensorShape(65_537, 3), 2.25f, -0.025f },
        { new TensorShape(9, 1), 0.0f, 0.125f },
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    [Trait("Category", "GpuIntegration")]
    public void CpuAndCuda_MatchIndependentRowReference_PreserveInputs_AndStayResident(
        TensorShape shape, float guidance, float delta)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        int count = checked((int)shape.ElementCount);
        int lastDim = checked((int)shape[shape.Rank - 1]);
        float[] initial = RandomValues(count, 101 + lastDim, 1.5f);
        float[] condValues = RandomValues(count, 211 + lastDim, 2.0f);
        float[] uncondValues = RandomValues(count, 307 + lastDim, 1.0f);
        float[] expected = Reference(initial, condValues, uncondValues, lastDim, guidance, delta, 1e-12f);

        using Tensor zCpu = TensorFrom(initial, shape);
        using Tensor condCpu = TensorFrom(condValues, shape);
        using Tensor uncondCpu = TensorFrom(uncondValues, shape);
        using (IBackend cpu = new CpuBackend())
            cpu.CfgNormalizedEulerStep(zCpu, condCpu, uncondCpu, guidance, delta);
        AssertClose(expected, Snapshot(zCpu), 3e-6f, "CPU");
        AssertExact(condValues, Snapshot(condCpu), "CPU cond mutation");
        AssertExact(uncondValues, Snapshot(uncondCpu), "CPU uncond mutation");

        using Tensor zHost = TensorFrom(initial, shape);
        using Tensor condHost = TensorFrom(condValues, shape);
        using Tensor uncondHost = TensorFrom(uncondValues, shape);
        using Tensor zCuda = new(shape, DType.F32);
        using Tensor condCuda = new(shape, DType.F32);
        using Tensor uncondCuda = new(shape, DType.F32);
        using CudaBackend cuda = new(0, PtxDir());
        cuda.Scale(zCuda, zHost, 1f);
        cuda.Scale(condCuda, condHost, 1f);
        cuda.Scale(uncondCuda, uncondHost, 1f);
        cuda.Sync();
        cuda.ResetD2hSyncCount();

        cuda.CfgNormalizedEulerStep(zCuda, condCuda, uncondCuda, guidance, delta);
        cuda.Sync();
        Assert.Equal(0, cuda.GetD2hSyncCount());

        float[] actualCond = Snapshot(condCuda);
        float[] actualUncond = Snapshot(uncondCuda);
        float[] actual = Snapshot(zCuda);
        Assert.Equal(3, cuda.GetD2hSyncCount());
        AssertExact(condValues, actualCond, "CUDA cond mutation");
        AssertExact(uncondValues, actualUncond, "CUDA uncond mutation");
        AssertClose(expected, actual, 4e-5f, "CUDA");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void ZeroPredictions_AndPredictionAlias_RemainFiniteAndResident()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        TensorShape shape = new(5, 33);
        float[] initial = RandomValues(165, 419, 1f);
        using Tensor initialHost = TensorFrom(initial, shape);
        using Tensor zeroHost = TensorFrom(new float[165], shape);
        using Tensor z = new(shape, DType.F32);
        using Tensor prediction = new(shape, DType.F32);
        using CudaBackend cuda = new(0, PtxDir());
        cuda.Scale(z, initialHost, 1f);
        cuda.Scale(prediction, zeroHost, 1f);
        cuda.ResetD2hSyncCount();
        cuda.CfgNormalizedEulerStep(z, prediction, prediction, 4f, 0.1f);
        cuda.Sync();
        Assert.Equal(0, cuda.GetD2hSyncCount());
        AssertExact(initial, Snapshot(z), "zero/aliased prediction update");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void ZeroEps_WithAllZeroGuidedRow_DoesNotPoisonZWithNaN()
    {
        // eps=0 with an all-zero guided row makes the normalization denominator exactly zero; the ratio must
        // resolve to 0 (the row contributes nothing either way), not 0/0 -> NaN poisoning z through 0·NaN.
        TensorShape shape = new(3, 17);
        float[] initial = RandomValues(51, 733, 1f);
        using Tensor zCpu = TensorFrom(initial, shape);
        using Tensor zeroPrediction = TensorFrom(new float[51], shape);
        using (IBackend cpu = new CpuBackend())
            cpu.CfgNormalizedEulerStep(zCpu, zeroPrediction, zeroPrediction, 4f, 0.1f, eps: 0f);
        AssertExact(initial, Snapshot(zCpu), "CPU zero-eps zero-row update");

        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }
        using Tensor zCuda = new(shape, DType.F32);
        using Tensor zeroHost = TensorFrom(new float[51], shape);
        using Tensor prediction = new(shape, DType.F32);
        using Tensor initialHost = TensorFrom(initial, shape);
        using CudaBackend cuda = new(0, PtxDir());
        cuda.Scale(zCuda, initialHost, 1f);
        cuda.Scale(prediction, zeroHost, 1f);
        cuda.CfgNormalizedEulerStep(zCuda, prediction, prediction, 4f, 0.1f, eps: 0f);
        cuda.Sync();
        AssertExact(initial, Snapshot(zCuda), "CUDA zero-eps zero-row update");
    }

    [Fact]
    public void FusedUpdate_MatchesLegacyLuminaNormalizeNegateAndFlowStep()
    {
        TensorShape shape = new(1, 2, 3, 7);
        float[] initial = RandomValues(42, 521, 1.25f);
        float[] condValues = RandomValues(42, 523, 2f);
        float[] uncondValues = RandomValues(42, 527, 0.75f);
        const float guidance = 4f;
        const int step = 3;

        FlowMatchEulerDiscreteScheduler scheduler = new(shift: 6f);
        scheduler.SetTimesteps(12);
        using Tensor legacyZ = TensorFrom(initial, shape);
        using Tensor cond = TensorFrom(condValues, shape);
        using Tensor uncond = TensorFrom(uncondValues, shape);
        using Tensor guided = CfgHelper.ApplyCfgNormalized(uncond, cond, guidance);
        float* guidedPtr = (float*)guided.DataPointer;
        for (long i = 0; i < guided.ElementCount; i++) guidedPtr[i] = -guidedPtr[i];
        using Tensor legacyOutput = new(shape, DType.F32);
        scheduler.Step(legacyOutput, guided, legacyZ, step);

        using Tensor fused = TensorFrom(initial, shape);
        using IBackend cpu = new CpuBackend();
        cpu.CfgNormalizedEulerStep(fused, cond, uncond, guidance, -scheduler.Dt(step));
        AssertClose(Snapshot(legacyOutput), Snapshot(fused), 3e-6f, "Lumina fused sign/scheduler equivalence");
    }

    [Fact]
    public void MalformedContracts_AreRejectedBeforeDispatch()
    {
        using Tensor z = new(new TensorShape(2, 7), DType.F32);
        using Tensor first = new(new TensorShape(2, 7), DType.F32);
        using Tensor second = new(new TensorShape(2, 7), DType.F32);
        using Tensor mismatch = new(new TensorShape(1, 14), DType.F32);
        using Tensor f16 = new(new TensorShape(2, 7), DType.F16);
        using Tensor emptyZ = new(new TensorShape(2, 0), DType.F32);
        using Tensor emptyFirst = new(new TensorShape(2, 0), DType.F32);
        using Tensor emptySecond = new(new TensorShape(2, 0), DType.F32);
        using Tensor zView = z.Reshape(new TensorShape(2, 7));
        using IBackend cpu = new CpuBackend();

        AssertMalformed(cpu, z, first, second, mismatch, f16, emptyZ, emptyFirst, emptySecond, zView);
        cpu.CfgNormalizedEulerStep(z, first, first, 1f, 0.1f);

        if (!CudaContext.IsAvailable()) return;
        using CudaBackend cuda = new(0, PtxDir());
        AssertMalformed(cuda, z, first, second, mismatch, f16, emptyZ, emptyFirst, emptySecond, zView);
        cuda.CfgNormalizedEulerStep(z, first, first, 1f, 0.1f);
        cuda.Sync();
    }

    private static void AssertMalformed(
        IBackend backend, Tensor z, Tensor first, Tensor second, Tensor mismatch, Tensor f16,
        Tensor emptyZ, Tensor emptyFirst, Tensor emptySecond, Tensor zView)
    {
        Assert.Throws<NotSupportedException>(() => backend.CfgNormalizedEulerStep(f16, first, second, 4f, 0.1f));
        Assert.Throws<ArgumentException>(() => backend.CfgNormalizedEulerStep(z, mismatch, second, 4f, 0.1f));
        Assert.Throws<ArgumentException>(() => backend.CfgNormalizedEulerStep(emptyZ, emptyFirst, emptySecond, 4f, 0.1f));
        Assert.Throws<ArgumentException>(() => backend.CfgNormalizedEulerStep(z, z, second, 4f, 0.1f));
        Assert.Throws<ArgumentException>(() => backend.CfgNormalizedEulerStep(z, first, z, 4f, 0.1f));
        Assert.Throws<ArgumentException>(() => backend.CfgNormalizedEulerStep(z, zView, second, 4f, 0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.CfgNormalizedEulerStep(z, first, second, float.NaN, 0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.CfgNormalizedEulerStep(z, first, second, 4f, float.PositiveInfinity));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.CfgNormalizedEulerStep(z, first, second, 4f, 0.1f, -1e-6f));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.CfgNormalizedEulerStep(z, first, second, 4f, 0.1f, float.NaN));
    }

    private static float[] Reference(
        float[] initial, float[] cond, float[] uncond, int lastDim,
        float guidance, float delta, float eps)
    {
        float[] output = (float[])initial.Clone();
        int rows = output.Length / lastDim;
        for (int row = 0; row < rows; row++)
        {
            int offset = row * lastDim;
            double condSq = 0.0;
            double guidedSq = 0.0;
            for (int d = 0; d < lastDim; d++)
            {
                int i = offset + d;
                float guided = uncond[i] + guidance * (cond[i] - uncond[i]);
                condSq += (double)cond[i] * cond[i];
                guidedSq += (double)guided * guided;
            }
            float ratio = (float)(Math.Sqrt(condSq) / (Math.Sqrt(guidedSq) + eps));
            for (int d = 0; d < lastDim; d++)
            {
                int i = offset + d;
                float guided = uncond[i] + guidance * (cond[i] - uncond[i]);
                output[i] += delta * (guided * ratio);
            }
        }
        return output;
    }

    private static float[] RandomValues(int count, int seed, float scale)
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
        values.AsSpan().CopyTo(new Span<float>((float*)tensor.DataPointer, values.Length));
        return tensor;
    }

    private static float[] Snapshot(Tensor tensor)
    {
        float[] values = new float[checked((int)tensor.ElementCount)];
        new ReadOnlySpan<float>((float*)tensor.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    private static void AssertExact(float[] expected, float[] actual, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{name} at {i}: expected {expected[i]:G9}, actual {actual[i]:G9}.");
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxError = 0f;
        int maxIndex = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]),
                $"{name}: non-finite at {i}; expected {expected[i]:G9}, actual {actual[i]:G9}.");
            float error = MathF.Abs(expected[i] - actual[i]);
            Assert.True(float.IsFinite(error), $"{name}: non-finite error at {i}.");
            if (error > maxError) { maxError = error; maxIndex = i; }
        }
        Assert.True(maxError <= tolerance,
            $"{name}: max error {maxError:E6} at {maxIndex} exceeds {tolerance:E2}.");
    }
}
