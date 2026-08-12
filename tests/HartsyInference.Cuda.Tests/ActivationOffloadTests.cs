using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Covers the activation-offload policy layer (docs/Research/MEMORY_SCHEDULING_SERVING.md §9): the named
/// per-tensor <see cref="CudaBackend.OffloadActivation"/> and the bulk <see cref="CudaBackend.OffloadActivations"/>.
/// The failure modes here are all silent — an offload that copies stale bytes, a reload that returns different bytes
/// than were paged out, and above all an offloaded tensor that weight auto-promotion quietly makes device-resident
/// again, which turns the whole lever into a no-op that still pays the D2H.</summary>
[Collection("CudaSerial")]
public sealed unsafe class ActivationOffloadTests
{
    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Ramp(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    /// <summary>Produces a fresh device-resident activation of <paramref name="shape"/> (Scale caches its output).</summary>
    private static Tensor Activation(CudaBackend cuda, Tensor source, TensorShape shape)
    {
        Tensor output = new Tensor(shape, DType.F32);
        cuda.Scale(output, source, 2.0f);
        return output;
    }

    [Fact]
    public void OffloadActivation_MaterializesHostDataAndReleasesDevice()
    {
        TensorShape shape = new TensorShape(512, 1024);
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor source = Ramp(shape, 7);
        using Tensor activation = Activation(cuda, source, shape);

        GpuTransferHelper.State state = GpuTransferHelper.CurrentState;
        Assert.True(state.ActivationCache.TryGetValue(activation, out (ulong gpuPtr, nuint bytes) entry),
            "Scale must leave its output in the activation cache for this test to mean anything.");
        ulong devicePtr = entry.gpuPtr;

        cuda.OffloadActivation(activation);

        Assert.False(state.ActivationCache.ContainsKey(activation));
        Assert.DoesNotContain(devicePtr, state.CachedPointers);
        Assert.DoesNotContain(activation, state.PinnedActivations);

        float* src = (float*)source.DataPointer;
        float* got = (float*)activation.DataPointer;
        for (long i = 0; i < shape.ElementCount; i++)
        {
            Assert.Equal(BitConverter.SingleToInt32Bits(src[i] * 2.0f), BitConverter.SingleToInt32Bits(got[i]));
        }
    }

    [Fact]
    public void OffloadedActivation_ReloadsWithIdenticalBytes()
    {
        TensorShape shape = new TensorShape(512, 1024);
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor source = Ramp(shape, 11);
        using Tensor activation = Activation(cuda, source, shape);

        cuda.OffloadActivation(activation);
        float[] paged = new float[shape.ElementCount];
        new Span<float>(activation.DataPointer, (int)shape.ElementCount).CopyTo(paged);

        // Next use is a cache miss on both caches, so it re-uploads the host copy.
        using Tensor reloaded = new Tensor(shape, DType.F32);
        cuda.Scale(reloaded, activation, 1.0f);
        float* got = (float*)reloaded.DataPointer;
        for (long i = 0; i < shape.ElementCount; i++)
        {
            Assert.Equal(BitConverter.SingleToInt32Bits(paged[i]), BitConverter.SingleToInt32Bits(got[i]));
        }
    }

    /// <param name="bulk">the bulk path blocks auto-promotion; the per-tensor path deliberately does not (it is the
    /// named spelling of the cross-step-cache idiom, which relies on promotion) — so this doubles as the positive
    /// control proving the assertion can observe a promotion at all</param>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AutoPromotion_DoesNotResurrectBulkOffloadedActivation(bool bulk)
    {
        TensorShape shape = new TensorShape(512, 1024);   // 2 MB — above AutoPromoteMinBytes
        // Promotion is skipped when it would eat the free-VRAM headroom, which would make the control vacuous.
        if (CudaDriverApi.cuMemGetInfo(out nuint free, out _) != 0 || (long)free < (1536L + 64L) << 20) return;

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor source = Ramp(shape, 13);
        using Tensor activation = Activation(cuda, source, shape);

        if (bulk)
        {
            Assert.True(cuda.OffloadActivations(1) > 0);
        }
        else
        {
            cuda.OffloadActivation(activation);
        }

        // Every read after an offload is an H2D re-upload; TryAutoPromote fires on the second one.
        for (int i = 0; i < 3; i++)
        {
            using Tensor consumer = new Tensor(shape, DType.F32);
            cuda.Scale(consumer, activation, 1.0f);
            _ = consumer.DataPointer;
        }

        bool promoted = GpuTransferHelper.CurrentState.WeightCache.ContainsKey(activation);
        Assert.Equal(!bulk, promoted);
    }

    [Fact]
    public void OffloadActivations_ReportsBytesFreedAndStopsAtTarget()
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor source = Ramp(new TensorShape(1024, 1024), 17);
        using Tensor big = Activation(cuda, source, new TensorShape(1024, 1024));      // 4 MB
        using Tensor mid = Activation(cuda, source, new TensorShape(512, 1024));       // 2 MB
        using Tensor small = Activation(cuda, source, new TensorShape(256, 1024));     // 1 MB

        GpuTransferHelper.State state = GpuTransferHelper.CurrentState;
        Assert.True(state.ActivationCache.ContainsKey(big));
        Assert.True(state.ActivationCache.ContainsKey(mid));
        Assert.True(state.ActivationCache.ContainsKey(small));

        Assert.Equal(0, cuda.OffloadActivations(0));

        // Largest-first: the 4 MB entry alone covers a 3 MB target, so the walk must stop there.
        long freed = cuda.OffloadActivations(3L << 20);
        Assert.Equal(4L << 20, freed);
        Assert.False(state.ActivationCache.ContainsKey(big));
        Assert.True(state.ActivationCache.ContainsKey(mid));
        Assert.True(state.ActivationCache.ContainsKey(small));

        // Over-large target drains what is left and reports exactly that, not the target.
        long rest = cuda.OffloadActivations(1L << 30);
        Assert.Equal(3L << 20, rest);
        Assert.False(state.ActivationCache.ContainsKey(mid));
        Assert.False(state.ActivationCache.ContainsKey(small));
        Assert.Equal(0, cuda.OffloadActivations(1L << 30));
    }

    /// <summary>The one real caller: <see cref="DeviceFeatureCache"/>'s cross-step residual and indicator under
    /// <c>HARTSY_STEP_CACHE_OFFLOAD=1</c>. Both are pinned when created, so an empty pin set afterwards is proof the
    /// page-out ran (the pin is cleared by the D2H); the residual must still reconstruct the block-stack output,
    /// which is what makes paging it survivable. Also exercises the pinned-first tie-break — at the
    /// <c>StoreResidual</c> call the residual shares its size class with the scratch negated input.</summary>
    [Fact]
    public void StepCacheCrossStepState_PagesToHostAndStillReconstructs()
    {
        // DeviceFeatureCache latches the switch into a static readonly on first touch, so it must be set before the
        // type is first used — this is the only test in this assembly that touches the class.
        Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_OFFLOAD", "1");
        try
        {
            TensorShape shape = new TensorShape(512, 1024);
            using CudaBackend cuda = new CudaBackend(0, PtxDir());
            using DeviceFeatureCache cache = new DeviceFeatureCache(threshold: 0.10f);
            using Tensor blockIn = Ramp(shape, 3);
            using Tensor blockOut = Ramp(shape, 5);

            Assert.True(cache.ShouldCompute(cuda, blockIn));
            cache.StoreResidual(cuda, blockIn, blockOut);

            Assert.Empty(GpuTransferHelper.CurrentState.PinnedActivations);

            using Tensor applied = cache.ApplyResidual(cuda, blockIn);
            float* expected = (float*)blockOut.DataPointer;
            float* got = (float*)applied.DataPointer;
            for (long i = 0; i < shape.ElementCount; i++)
            {
                Assert.True(Math.Abs(expected[i] - got[i]) <= 1e-5f,
                    $"element {i}: expected {expected[i]}, got {got[i]} after paging the residual to host.");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_STEP_CACHE_OFFLOAD", null);
        }
    }
}
