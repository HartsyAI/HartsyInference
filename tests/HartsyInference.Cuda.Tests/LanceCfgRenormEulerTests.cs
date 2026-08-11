using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// Contract, numerical-parity, and residency gates for Lance's global-renormalized CFG Euler step.
/// The CUDA assertion is made before any result is inspected, so a hidden host fallback cannot pass on numerics.
/// </summary>
[Collection("CudaSerial")]
public sealed unsafe class LanceCfgRenormEulerTests
{
    private readonly ITestOutputHelper _output;

    public LanceCfgRenormEulerTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    public static TheoryData<int, float, float, float> TailAndClampCases() => new()
    {
        // Launch boundaries plus Lance's 512x512 token latent (32*32*48).
        { 1,      4.0f, -0.03125f, 0.0f },
        { 33,     0.0f, -0.09000f, 0.0f }, // cond norm > uncond norm: exercises the upper clamp at 1
        { 255,    7.5f, -0.07000f, 0.0f },
        { 256,    3.0f, -0.12500f, 0.2f },
        { 257,   12.0f, -0.01500f, 0.85f },
        { 49_152, 4.0f, -0.04000f, 0.0f },
        // 1024x1024 Lance image latent: 64*64 tokens * 48 channels. This exceeds the 256-block
        // grid cap and therefore exercises every stage's grid-stride loop, including the far tail.
        { 196_608, 6.0f, -0.02000f, 0.35f },
    };

    [Theory]
    [MemberData(nameof(TailAndClampCases))]
    [Trait("Category", "GpuIntegration")]
    public void CpuAndCuda_MatchIndependentReference_PreserveInputs_AndStayResident(
        int count, float guidance, float delta, float renormMin)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        float[] zValues = RandomValues(count, seed: 101 + count, scale: 1.5f);
        float[] condValues = RandomValues(count, seed: 211 + count, scale: 2.0f);
        float[] uncondValues = RandomValues(count, seed: 307 + count, scale: 1.0f);
        float[] expected = Reference(zValues, condValues, uncondValues, guidance, delta, renormMin, out float scale);

        // Exercise the interface's host fallback independently of the CUDA implementation.
        using Tensor zCpu = TensorFrom(zValues);
        using Tensor condCpu = TensorFrom(condValues);
        using Tensor uncondCpu = TensorFrom(uncondValues);
        using (IBackend cpu = new CpuBackend())
            cpu.CfgRenormEulerStep(zCpu, condCpu, uncondCpu, guidance, delta, renormMin);
        AssertClose(expected, Snapshot(zCpu), 2e-6f, "CPU");
        AssertExact(condValues, Snapshot(condCpu), "CPU cond mutation");
        AssertExact(uncondValues, Snapshot(uncondCpu), "CPU uncond mutation");

        using Tensor zHost = TensorFrom(zValues);
        using Tensor condHost = TensorFrom(condValues);
        using Tensor uncondHost = TensorFrom(uncondValues);
        using Tensor zCuda = new(new TensorShape(count), DType.F32);
        using Tensor condCuda = new(new TensorShape(count), DType.F32);
        using Tensor uncondCuda = new(new TensorShape(count), DType.F32);
        using CudaBackend cuda = new(0, PtxDir());

        // Materialize every operand as a CUDA activation before the measured region. If the new operation reads
        // DataPointer, the lazy transfer counter will expose the full-tensor D2H synchronization immediately.
        cuda.Scale(zCuda, zHost, 1f);
        cuda.Scale(condCuda, condHost, 1f);
        cuda.Scale(uncondCuda, uncondHost, 1f);
        cuda.Sync();
        cuda.ResetD2hSyncCount();

        cuda.CfgRenormEulerStep(zCuda, condCuda, uncondCuda, guidance, delta, renormMin);
        cuda.Sync();
        Assert.Equal(0, cuda.GetD2hSyncCount());

        float[] actualCond = Snapshot(condCuda);
        float[] actualUncond = Snapshot(uncondCuda);
        float[] actualZ = Snapshot(zCuda);
        Assert.Equal(3, cuda.GetD2hSyncCount());
        AssertExact(condValues, actualCond, "CUDA cond mutation");
        AssertExact(uncondValues, actualUncond, "CUDA uncond mutation");
        AssertClose(expected, actualZ, 3e-5f, "CUDA");
        _output.WriteLine($"N={count}, guidance={guidance}, renormMin={renormMin}, reference scale={scale:G9}");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void AllZeroPredictions_LeaveLatentFiniteAndUnchanged()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int count = 513;
        float[] initial = RandomValues(count, 353, 1.0f);
        using Tensor initialHost = TensorFrom(initial);
        using Tensor zeroHost = TensorFrom(new float[count]);
        using Tensor z = new(new TensorShape(count), DType.F32);
        using Tensor cond = new(new TensorShape(count), DType.F32);
        using Tensor uncond = new(new TensorShape(count), DType.F32);
        using CudaBackend cuda = new(0, PtxDir());
        cuda.Scale(z, initialHost, 1f);
        cuda.Scale(cond, zeroHost, 1f);
        cuda.Scale(uncond, zeroHost, 1f);
        cuda.ResetD2hSyncCount();
        cuda.CfgRenormEulerStep(z, cond, uncond, 7f, -0.1f, 0.6f);
        cuda.Sync();
        Assert.Equal(0, cuda.GetD2hSyncCount());
        AssertExact(initial, Snapshot(z), "all-zero prediction update");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void RepeatedCalls_DoNotReadBackOrRetainScratch()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        // Maxes the reduction's 256-block scratch and crosses its element boundary by one. Two hundred fifty-six
        // calls are over 8x a production Lance denoise loop. NVIDIA Racecheck retains enough instrumentation
        // metadata per async allocation to kill the testhost at that count, so sanitizer runs use a bounded loop;
        // the ordinary run remains the authoritative allocator-drift stress gate.
        const int count = 65_537;
        bool underComputeSanitizer = Environment.GetEnvironmentVariable("NV_SANITIZER_INJECTION_PORT_BASE") is not null;
        int repeats = underComputeSanitizer ? 16 : 256;
        using Tensor zHost = TensorFrom(RandomValues(count, 401, 1.0f));
        using Tensor condHost = TensorFrom(RandomValues(count, 409, 1.5f));
        using Tensor uncondHost = TensorFrom(RandomValues(count, 419, 0.75f));
        using Tensor z = new(new TensorShape(count), DType.F32);
        using Tensor cond = new(new TensorShape(count), DType.F32);
        using Tensor uncond = new(new TensorShape(count), DType.F32);
        using CudaBackend cuda = new(0, PtxDir());
        cuda.Scale(z, zHost, 1f);
        cuda.Scale(cond, condHost, 1f);
        cuda.Scale(uncond, uncondHost, 1f);

        // Warm the stream-ordered allocator before taking the memory baseline.
        cuda.CfgRenormEulerStep(z, cond, uncond, 4f, -0.001f, 0f);
        cuda.TrimMemoryPool();
        long cachedBefore = cuda.GetGpuCacheStats().cachedBytes;
        long freeBefore = cuda.FreeMemoryBytes();
        cuda.ResetD2hSyncCount();

        for (int i = 0; i < repeats; i++)
            cuda.CfgRenormEulerStep(z, cond, uncond, 4f, -0.001f, 0f);

        cuda.TrimMemoryPool();
        long cachedAfter = cuda.GetGpuCacheStats().cachedBytes;
        long freeAfter = cuda.FreeMemoryBytes();
        Assert.Equal(0, cuda.GetD2hSyncCount());
        Assert.Equal(cachedBefore, cachedAfter);
        if (!underComputeSanitizer)
            Assert.True(freeAfter >= freeBefore - 1024 * 1024,
                $"Per-call CFG scratch drifted device memory: before={freeBefore}, after={freeAfter}.");
        _output.WriteLine($"{repeats} calls: cached bytes {cachedBefore}->{cachedAfter}, free bytes {freeBefore}->{freeAfter}");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void CfgEulerStep_PinnedLatentSurvivesPerStepActivationSweeps_WithoutReadback()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int count = 16_387; // deliberately crosses many 256-thread blocks with a three-element tail
        const int steps = 7;
        const float guidance = 5.5f;
        const float delta = -0.0375f;
        float[] initial = RandomValues(count, 503, 1.25f);
        float[] posValues = RandomValues(count, 509, 0.9f);
        float[] negValues = RandomValues(count, 521, 0.6f);
        float[] expected = (float[])initial.Clone();
        for (int step = 0; step < steps; step++)
            for (int i = 0; i < count; i++)
            {
                float velocity = guidance * posValues[i] + (1f - guidance) * negValues[i];
                expected[i] += velocity * delta;
            }

        using Tensor initialHost = TensorFrom(initial);
        using Tensor posHost = TensorFrom(posValues);
        using Tensor negHost = TensorFrom(negValues);
        using Tensor z = new(new TensorShape(count), DType.F32);
        using CudaBackend cuda = new(0, PtxDir());
        cuda.Scale(z, initialHost, 1f);
        cuda.PinActivation(z);
        cuda.Sync();
        cuda.ResetD2hSyncCount();

        for (int step = 0; step < steps; step++)
        {
            using Tensor pos = new(new TensorShape(count), DType.F32);
            using Tensor neg = new(new TensorShape(count), DType.F32);
            cuda.Scale(pos, posHost, 1f);
            cuda.Scale(neg, negHost, 1f);
            cuda.CfgEulerStep(z, pos, neg, guidance, delta);
            cuda.PinActivation(z);
            // Mirrors the SD3 per-step cleanup: prediction activations go away, the carried latent does not.
            cuda.FreeActivations(trimPool: false);
        }

        cuda.Sync();
        Assert.Equal(0, cuda.GetD2hSyncCount());
        cuda.UnpinActivation(z);
        float[] actual = Snapshot(z);
        Assert.Equal(1, cuda.GetD2hSyncCount());
        AssertClose(expected, actual, 2e-5f, "pinned CfgEuler latent");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void CfgEulerStep_WithGpuResidentPredictions_MatchesHostCfgAndFlowReference()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int count = 4_099;
        const float guidance = 4.25f;
        float[] latentValues = RandomValues(count, 601, 1.0f);
        float[] condValues = RandomValues(count, 607, 0.8f);
        float[] uncondValues = RandomValues(count, 613, 0.7f);
        using Tensor latentHost = TensorFrom(latentValues);
        using Tensor condHost = TensorFrom(condValues);
        using Tensor uncondHost = TensorFrom(uncondValues);
        using CudaBackend cuda = new(0, PtxDir());

        FlowMatchEulerDiscreteScheduler scheduler = new(shift: 3f);
        scheduler.SetTimesteps(6);
        const int stepIndex = 2;

        // Host reference versus the device-resident endpoint. Current SD3 still CPU-unpatchifies its predictions,
        // so this transfer delta becomes representative only after that transformer-glue migration; it is not an
        // assertion about the current end-to-end SD3 forward.
        using Tensor oldCond = new(new TensorShape(count), DType.F32);
        using Tensor oldUncond = new(new TensorShape(count), DType.F32);
        cuda.Scale(oldCond, condHost, 1f);
        cuda.Scale(oldUncond, uncondHost, 1f);
        cuda.Sync();
        cuda.ResetD2hSyncCount();
        using Tensor guided = CfgHelper.ApplyCfg(oldUncond, oldCond, guidance);
        Assert.Equal(2, cuda.GetD2hSyncCount());
        using Tensor expected = new(new TensorShape(count), DType.F32);
        scheduler.Step(expected, guided, latentHost, stepIndex);
        Assert.Equal(2, cuda.GetD2hSyncCount());

        // Resident replacement: the same CFG algebra and scheduler delta fold into one device update.
        using Tensor latent = new(new TensorShape(count), DType.F32);
        using Tensor cond = new(new TensorShape(count), DType.F32);
        using Tensor uncond = new(new TensorShape(count), DType.F32);
        cuda.Scale(latent, latentHost, 1f);
        cuda.Scale(cond, condHost, 1f);
        cuda.Scale(uncond, uncondHost, 1f);
        cuda.Sync();
        cuda.ResetD2hSyncCount();
        cuda.CfgEulerStep(latent, cond, uncond, guidance, scheduler.Dt(stepIndex));
        cuda.Sync();
        Assert.Equal(0, cuda.GetD2hSyncCount());
        float[] actual = Snapshot(latent);
        Assert.Equal(1, cuda.GetD2hSyncCount());
        AssertClose(Snapshot(expected), actual, 2e-5f, "legacy CFG+scheduler vs resident CfgEuler");
    }

    [Fact]
    public void MalformedContracts_AreRejectedBeforeDispatch()
    {
        using Tensor z = new(new TensorShape(7), DType.F32);
        using Tensor first = new(new TensorShape(7), DType.F32);
        using Tensor second = new(new TensorShape(7), DType.F32);
        using Tensor shapeMismatch = new(new TensorShape(1, 7), DType.F32);
        using Tensor f16 = new(new TensorShape(7), DType.F16);
        using Tensor emptyZ = new(new TensorShape(0), DType.F32);
        using Tensor emptyFirst = new(new TensorShape(0), DType.F32);
        using Tensor emptySecond = new(new TensorShape(0), DType.F32);
        using Tensor zView = z.Reshape(new TensorShape(7));
        using IBackend cpu = new CpuBackend();

        Assert.Throws<NotSupportedException>(() => cpu.CfgRenormEulerStep(f16, first, second, 4f, -0.1f, 0f));
        Assert.Throws<ArgumentException>(() => cpu.CfgRenormEulerStep(z, shapeMismatch, second, 4f, -0.1f, 0f));
        Assert.Throws<ArgumentException>(() => cpu.CfgRenormEulerStep(emptyZ, emptyFirst, emptySecond, 4f, -0.1f, 0f));
        Assert.Throws<ArgumentException>(() => cpu.CfgRenormEulerStep(z, z, second, 4f, -0.1f, 0f));
        Assert.Throws<ArgumentException>(() => cpu.CfgRenormEulerStep(z, first, z, 4f, -0.1f, 0f));
        Assert.Throws<ArgumentException>(() => cpu.CfgRenormEulerStep(z, zView, second, 4f, -0.1f, 0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.CfgRenormEulerStep(z, first, second, float.NaN, -0.1f, 0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.CfgRenormEulerStep(z, first, second, 4f, float.PositiveInfinity, 0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.CfgRenormEulerStep(z, first, second, 4f, -0.1f, -0.01f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.CfgRenormEulerStep(z, first, second, 4f, -0.1f, 1.01f));

        Assert.Throws<NotSupportedException>(() => cpu.CfgEulerStep(f16, first, second, 4f, -0.1f));
        Assert.Throws<ArgumentException>(() => cpu.CfgEulerStep(z, shapeMismatch, second, 4f, -0.1f));
        Assert.Throws<ArgumentException>(() => cpu.CfgEulerStep(emptyZ, emptyFirst, emptySecond, 4f, -0.1f));
        Assert.Throws<ArgumentException>(() => cpu.CfgEulerStep(z, z, second, 4f, -0.1f));
        Assert.Throws<ArgumentException>(() => cpu.CfgEulerStep(z, first, z, 4f, -0.1f));
        Assert.Throws<ArgumentException>(() => cpu.CfgEulerStep(z, zView, second, 4f, -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.CfgEulerStep(z, first, second, float.NaN, -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.CfgEulerStep(z, first, second, 4f, float.PositiveInfinity));

        // Prediction aliases are intentional for conditional-only sampling and must remain supported.
        cpu.CfgEulerStep(z, first, first, 1f, -0.1f);
        cpu.CfgRenormEulerStep(z, first, first, 1f, -0.1f, 0f);

        if (!CudaContext.IsAvailable()) return;
        using CudaBackend cuda = new(0, PtxDir());
        Assert.Throws<NotSupportedException>(() => cuda.CfgRenormEulerStep(f16, first, second, 4f, -0.1f, 0f));
        Assert.Throws<ArgumentException>(() => cuda.CfgRenormEulerStep(z, shapeMismatch, second, 4f, -0.1f, 0f));
        Assert.Throws<ArgumentException>(() => cuda.CfgRenormEulerStep(emptyZ, emptyFirst, emptySecond, 4f, -0.1f, 0f));
        Assert.Throws<ArgumentException>(() => cuda.CfgRenormEulerStep(z, z, second, 4f, -0.1f, 0f));
        Assert.Throws<ArgumentException>(() => cuda.CfgRenormEulerStep(z, first, z, 4f, -0.1f, 0f));
        Assert.Throws<ArgumentException>(() => cuda.CfgRenormEulerStep(z, zView, second, 4f, -0.1f, 0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cuda.CfgRenormEulerStep(z, first, second, float.NaN, -0.1f, 0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cuda.CfgRenormEulerStep(z, first, second, 4f, float.PositiveInfinity, 0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cuda.CfgRenormEulerStep(z, first, second, 4f, -0.1f, -0.01f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cuda.CfgRenormEulerStep(z, first, second, 4f, -0.1f, 1.01f));

        Assert.Throws<NotSupportedException>(() => cuda.CfgEulerStep(f16, first, second, 4f, -0.1f));
        Assert.Throws<ArgumentException>(() => cuda.CfgEulerStep(z, shapeMismatch, second, 4f, -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cuda.CfgEulerStep(emptyZ, emptyFirst, emptySecond, 4f, -0.1f));
        Assert.Throws<ArgumentException>(() => cuda.CfgEulerStep(z, z, second, 4f, -0.1f));
        Assert.Throws<ArgumentException>(() => cuda.CfgEulerStep(z, first, z, 4f, -0.1f));
        Assert.Throws<ArgumentException>(() => cuda.CfgEulerStep(z, zView, second, 4f, -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cuda.CfgEulerStep(z, first, second, float.NaN, -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cuda.CfgEulerStep(z, first, second, 4f, float.PositiveInfinity));

        cuda.CfgEulerStep(z, first, first, 1f, -0.1f);
        cuda.CfgRenormEulerStep(z, first, first, 1f, -0.1f, 0f);
        cuda.Sync();
    }

    private static float[] Reference(
        float[] z, float[] cond, float[] uncond, float guidance, float delta, float renormMin, out float scale)
    {
        double condSq = 0.0;
        double guidedSq = 0.0;
        for (int i = 0; i < z.Length; i++)
        {
            float guided = uncond[i] + guidance * (cond[i] - uncond[i]);
            condSq += (double)cond[i] * cond[i];
            guidedSq += (double)guided * guided;
        }
        scale = (float)Math.Clamp(Math.Sqrt(condSq) / (Math.Sqrt(guidedSq) + 1e-8), renormMin, 1.0);
        float[] output = (float[])z.Clone();
        for (int i = 0; i < output.Length; i++)
        {
            float guided = uncond[i] + guidance * (cond[i] - uncond[i]);
            guided *= scale;
            output[i] += guided * delta;
        }
        return output;
    }

    private static float[] RandomValues(int count, int seed, float scale)
    {
        Random rng = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
        return values;
    }

    private static Tensor TensorFrom(float[] values)
    {
        Tensor tensor = new(new TensorShape(values.Length), DType.F32);
        values.AsSpan().CopyTo(new Span<float>((float*)tensor.DataPointer, values.Length));
        return tensor;
    }

    private static float[] Snapshot(Tensor tensor)
    {
        float[] values = new float[tensor.ElementCount];
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
                $"{name}: non-finite value at {i}; expected {expected[i]:G9}, actual {actual[i]:G9}.");
            float error = MathF.Abs(expected[i] - actual[i]);
            Assert.True(float.IsFinite(error),
                $"{name}: non-finite error at {i}; expected {expected[i]:G9}, actual {actual[i]:G9}.");
            if (error > maxError) { maxError = error; maxIndex = i; }
        }
        Assert.True(maxError <= tolerance,
            $"{name}: max error {maxError:E6} at {maxIndex} exceeds {tolerance:E2}.");
    }
}
