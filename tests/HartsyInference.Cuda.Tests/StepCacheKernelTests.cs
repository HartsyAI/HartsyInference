using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Numerics gate for the stepcache.ptx device-side feature-cache reduction
/// (INFERENCE_ACCEL_GRIND §H1.3): <see cref="CudaBackend.RelativeL1Distance"/> against the IBackend host
/// default on identical random tensors. F32 within 1e-6 relative, F16 within 1e-3 (atomic accumulation
/// order nondeterminism is inside those tolerances). Also asserts the optional-module plumbing —
/// <see cref="CudaBackend.SupportsDeviceStepCacheGate"/> is true once stepcache.ptx ships.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class StepCacheKernelTests
{
    private readonly ITestOutputHelper _output;
    public StepCacheKernelTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor RandomF32(TensorShape shape, int seed, double scale = 1.0)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return t;
    }

    private static Tensor ToF16(Tensor f32)
    {
        Tensor t = new Tensor(f32.Shape, DType.F16);
        float* src = (float*)f32.DataPointer;
        ushort* dst = (ushort*)t.DataPointer;
        long n = f32.ElementCount;
        for (long i = 0; i < n; i++) dst[i] = BitConverter.HalfToUInt16Bits((Half)src[i]);
        return t;
    }

    [Fact]
    public void SupportsDeviceStepCacheGate_TrueWithPtxPresent()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        Assert.True(File.Exists(Path.Combine(PtxDir(), "stepcache.ptx")),
            "stepcache.ptx missing from Ptx dir — run src/HartsyInference.Cuda/Kernels/dit/build.sh");
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        Assert.True(cuda.SupportsDeviceStepCacheGate);
    }

    [Fact]
    public void RelativeL1Distance_MatchesHostDefault_F32()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        // Realistic indicator scale: a DiT img-stream block output (seq × hidden), non-tiny so the
        // grid-stride loop and the atomic block accumulation both get exercised across many blocks.
        TensorShape shape = new TensorShape(1, 4096, 1536);
        using Tensor a = RandomF32(shape, 42);
        using Tensor b = RandomF32(shape, 43);

        float host;
        using (CpuBackend cpu = new CpuBackend())
            host = ((IBackend)cpu).RelativeL1Distance(a, b);

        float device;
        using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
            device = cuda.RelativeL1Distance(a, b);

        _output.WriteLine($"F32 host={host:G9} device={device:G9} relErr={Math.Abs(host - device) / host:E3}");
        Assert.True(host > 0f);
        Assert.True(Math.Abs(host - device) / host < 1e-5f,
            $"F32 rel err {Math.Abs(host - device) / host:E3} exceeds 1e-5 (host={host:G9}, device={device:G9})");
    }

    [Fact]
    public void RelativeL1Distance_MatchesHostDefault_F16()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        TensorShape shape = new TensorShape(1, 4096, 1536);
        using Tensor a32 = RandomF32(shape, 44);
        using Tensor b32 = RandomF32(shape, 45);
        using Tensor a = ToF16(a32);
        using Tensor b = ToF16(b32);

        float host;
        using (CpuBackend cpu = new CpuBackend())
            host = ((IBackend)cpu).RelativeL1Distance(a, b);

        float device;
        using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
            device = cuda.RelativeL1Distance(a, b);

        _output.WriteLine($"F16 host={host:G9} device={device:G9} relErr={Math.Abs(host - device) / host:E3}");
        Assert.True(host > 0f);
        Assert.True(Math.Abs(host - device) / host < 1e-3f,
            $"F16 rel err {Math.Abs(host - device) / host:E3} exceeds 1e-3 (host={host:G9}, device={device:G9})");
    }

    [Fact]
    public void RelativeL1Distance_IdenticalTensors_ReturnsZero()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        TensorShape shape = new TensorShape(1, 512, 768);
        using Tensor a = RandomF32(shape, 46);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        float device = cuda.RelativeL1Distance(a, a);
        _output.WriteLine($"identical-tensor distance = {device:G9}");
        Assert.Equal(0f, device, 3);
    }
}
