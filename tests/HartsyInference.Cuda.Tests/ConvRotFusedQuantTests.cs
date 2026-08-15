using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Pins the fused ConvRot+per-row-int8-quant kernel against the EXPLICIT unfused pair it replaces
/// (<c>convrot_rotate</c> then <c>w8a8_quant_rowwise</c>) — not against a second copy of the fused logic — so a
/// wrong rotation stage, a wrong reduction, or a changed rounding rule fails here instead of as slightly-off
/// video. Non-square shapes throughout, so a transposed axis cannot pass.</summary>
[Collection("CudaSerial")]
public sealed unsafe class ConvRotFusedQuantTests
{
    private readonly ITestOutputHelper _output;
    public ConvRotFusedQuantTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Theory]
    [InlineData(37, 4096, 256)]      // LTX video width, ragged row count
    [InlineData(129, 2048, 256)]     // LTX audio width
    [InlineData(8, 1024, 256)]
    [InlineData(5, 512, 64)]         // smaller group (must be a power of FOUR: kron(h4,...,h4))
    public void FusedConvRotQuant_MatchesRotateThenQuant(int rows, int cols, int group)
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        CudaKernels k = cuda.Kernels!;
        Assert.True(k.HasFusedConvRotQuant(cols, group), $"fused path unavailable for cols={cols}");

        Tensor xF32 = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* px = (float*)xF32.DataPointer;
        Random rng = new Random(1234);
        for (long i = 0; i < (long)rows * cols; i++) px[i] = (float)(rng.NextDouble() * 6 - 3);
        using Tensor x = xF32.CastTo(DType.F16);
        xF32.Dispose();

        long n = (long)rows * cols;
        using Tensor qFused = new Tensor(new TensorShape(rows, cols), DType.I8);
        using Tensor qRef = new Tensor(new TensorShape(rows, cols), DType.I8);
        using Tensor sFused = new Tensor(new TensorShape(rows), DType.F32);
        using Tensor sRef = new Tensor(new TensorShape(rows), DType.F32);

        ulong dx = GpuTransferHelper.CopyToDevice(x);
        ulong dRot = GpuTransferHelper.AllocateDevice((nuint)(n * 2));
        ulong dQFused = GpuTransferHelper.AllocateDevice((nuint)n);
        ulong dQRef = GpuTransferHelper.AllocateDevice((nuint)n);
        ulong dSFused = GpuTransferHelper.AllocateDevice((nuint)(rows * 4));
        ulong dSRef = GpuTransferHelper.AllocateDevice((nuint)(rows * 4));
        try
        {
            // Reference: the two kernels the fused one replaces.
            k.LaunchConvRotRotate(dRot, dx, n, group, 0, srcF16: true);
            k.LaunchW8A8QuantRowwise(dQRef, dSRef, dRot, rows, cols, 0, srcF16: true);
            // Fused.
            k.LaunchConvRotQuantRowwise(dQFused, dSFused, dx, rows, cols, group, 0, srcF16: true);
            cuda.Sync();

            GpuTransferHelper.CopyToHost(qFused, dQFused, (nuint)n);
            GpuTransferHelper.CopyToHost(qRef, dQRef, (nuint)n);
            GpuTransferHelper.CopyToHost(sFused, dSFused, (nuint)(rows * 4));
            GpuTransferHelper.CopyToHost(sRef, dSRef, (nuint)(rows * 4));

            float* pf = (float*)sFused.DataPointer, pr = (float*)sRef.DataPointer;
            for (int r = 0; r < rows; r++)
                Assert.True(pr[r] == pf[r], $"row {r} scale: fused {pf[r]} vs unfused {pr[r]}");
            sbyte* qf = (sbyte*)qFused.DataPointer, qr2 = (sbyte*)qRef.DataPointer;
            long diffs = 0;
            for (long i = 0; i < n; i++) if (qf[i] != qr2[i]) diffs++;
            _output.WriteLine($"rows={rows} cols={cols} group={group}: {diffs} of {n} int8 codes differ");
            Assert.Equal(0, diffs);
        }
        finally
        {
            foreach (ulong p in new[] { dx, dRot, dQFused, dQRef, dSFused, dSRef }) GpuTransferHelper.FreeDevice(p);
        }
    }

    /// <summary>Same contract for the wide (f16-staged) variant, which serves the row widths the f32-staged
    /// kernel above refuses. 16384 is LTX-2.5's FFN-down activation width — the shape this kernel exists for.</summary>
    [Theory]
    [InlineData(37, 16384, 64)]
    [InlineData(11, 12288, 256)]
    [InlineData(3, 12288, 4096)]     // group wider than the default float tile
    [InlineData(129, 10240, 1024)]
    public void WideConvRotQuant_MatchesRotateThenQuant(int rows, int cols, int group)
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        CudaKernels k = cuda.Kernels!;
        Assert.False(k.HasFusedConvRotQuant(cols, group), $"cols={cols} should be too wide for the f32-staged kernel");
        Assert.True(k.HasWideConvRotQuant(cols, group, srcF16: true), $"wide path unavailable for cols={cols}");

        Tensor xF32 = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* px = (float*)xF32.DataPointer;
        Random rng = new Random(99);
        for (long i = 0; i < (long)rows * cols; i++) px[i] = (float)(rng.NextDouble() * 6 - 3);
        using Tensor x = xF32.CastTo(DType.F16);
        xF32.Dispose();

        long n = (long)rows * cols;
        using Tensor qWide = new Tensor(new TensorShape(rows, cols), DType.I8);
        using Tensor qRef = new Tensor(new TensorShape(rows, cols), DType.I8);
        using Tensor sWide = new Tensor(new TensorShape(rows), DType.F32);
        using Tensor sRef = new Tensor(new TensorShape(rows), DType.F32);

        ulong dx = GpuTransferHelper.CopyToDevice(x);
        ulong dRot = GpuTransferHelper.AllocateDevice((nuint)(n * 2));
        ulong dQWide = GpuTransferHelper.AllocateDevice((nuint)n);
        ulong dQRef = GpuTransferHelper.AllocateDevice((nuint)n);
        ulong dSWide = GpuTransferHelper.AllocateDevice((nuint)(rows * 4));
        ulong dSRef = GpuTransferHelper.AllocateDevice((nuint)(rows * 4));
        try
        {
            k.LaunchConvRotRotate(dRot, dx, n, group, 0, srcF16: true);
            k.LaunchW8A8QuantRowwise(dQRef, dSRef, dRot, rows, cols, 0, srcF16: true);
            k.LaunchConvRotQuantRowwiseWide(dQWide, dSWide, dx, rows, cols, group, 0);
            cuda.Sync();

            GpuTransferHelper.CopyToHost(qWide, dQWide, (nuint)n);
            GpuTransferHelper.CopyToHost(qRef, dQRef, (nuint)n);
            GpuTransferHelper.CopyToHost(sWide, dSWide, (nuint)(rows * 4));
            GpuTransferHelper.CopyToHost(sRef, dSRef, (nuint)(rows * 4));

            float* pw = (float*)sWide.DataPointer, pr = (float*)sRef.DataPointer;
            for (int r = 0; r < rows; r++)
                Assert.True(pr[r] == pw[r], $"row {r} scale: wide {pw[r]} vs unfused {pr[r]}");
            sbyte* qw = (sbyte*)qWide.DataPointer, qr2 = (sbyte*)qRef.DataPointer;
            long diffs = 0;
            for (long i = 0; i < n; i++) if (qw[i] != qr2[i]) diffs++;
            _output.WriteLine($"rows={rows} cols={cols} group={group}: {diffs} of {n} int8 codes differ");
            Assert.Equal(0, diffs);
        }
        finally
        {
            foreach (ulong p in new[] { dx, dRot, dQWide, dQRef, dSWide, dSRef }) GpuTransferHelper.FreeDevice(p);
        }
    }
}
