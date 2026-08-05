using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Equivalence tests for the row-indexed DiT modulation ops, which fold the modulation-table
/// gather (and the <c>1 +</c> on the scale) into the affine/gate kernel so the caller never materializes
/// the <c>[seq, hidden]</c> gathered tables. The reference is the unfused sequence they replace —
/// <c>AddScalar → RowGather → AffineBroadcastLastDim</c> and <c>RowGather → GatedResidualLastDim</c>.
/// On CPU both paths are the same F32 operations in the same order, so they must agree BIT-EXACTLY;
/// CUDA is compared at KERNEL.md's 1e-5 F32 tolerance because nvcc contracts <c>x*s + shift</c> into a
/// single FMA that the two-step CPU path does not. Skips cleanly when CUDA is unavailable.</summary>
[Collection("CudaSerial")]
public sealed unsafe class DitRowIndexedKernelTests
{
    private const int Seq = 37, Hidden = 64, ModRows = 6;

    private readonly ITestOutputHelper _output;
    public DitRowIndexedKernelTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Random(TensorShape shape, int seed, float lo = -1f, float hi = 1f)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * (hi - lo) + lo);
        return t;
    }

    /// <summary>One modulation row per token, deliberately non-monotonic so a kernel that ignored the index
    /// (or used <c>row / seqLen</c> like the lastdim twin) could not pass.</summary>
    private static Tensor RowIndex(int rows, int modRows, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows), DType.I32);
        int* p = (int*)t.DataPointer;
        Random rng = new Random(seed);
        for (int i = 0; i < rows; i++) p[i] = rng.Next(modRows);
        return t;
    }

    /// <summary>The unfused CPU reference for the fused affine: <c>1+scale</c>, gather both tables, then affine.</summary>
    private static void UnfusedAffine(IBackend b, Tensor output, Tensor input, Tensor scaleTable, Tensor? shiftTable, Tensor rowIndex)
    {
        int rows = (int)rowIndex.ElementCount;
        using Tensor onePlus = new Tensor(scaleTable.Shape, DType.F32);
        b.AddScalar(onePlus, scaleTable, 1f);
        using Tensor scaleRows = new Tensor(new TensorShape(rows, Hidden), DType.F32);
        b.RowGather(scaleRows, onePlus, rowIndex, rows, Hidden);
        if (shiftTable is null)
        {
            b.AffineBroadcastLastDim(output, input, scaleRows, null);
            return;
        }
        using Tensor shiftRows = new Tensor(new TensorShape(rows, Hidden), DType.F32);
        b.RowGather(shiftRows, shiftTable, rowIndex, rows, Hidden);
        b.AffineBroadcastLastDim(output, input, scaleRows, shiftRows);
    }

    /// <summary>The unfused CPU reference for the fused gate: gather the gate table, then gated residual.</summary>
    private static void UnfusedGate(IBackend b, Tensor output, Tensor residual, Tensor value, Tensor gateTable, Tensor rowIndex)
    {
        int rows = (int)rowIndex.ElementCount;
        using Tensor gateRows = new Tensor(new TensorShape(rows, Hidden), DType.F32);
        b.RowGather(gateRows, gateTable, rowIndex, rows, Hidden);
        b.GatedResidualLastDim(output, residual, value, gateRows);
    }

    private void AssertBitExact(Tensor expected, Tensor actual, string name)
    {
        float* a = (float*)expected.DataPointer;
        float* b = (float*)actual.DataPointer;
        long n = expected.ElementCount;
        for (long i = 0; i < n; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
                Assert.Fail($"{name}: bit mismatch at {i} — unfused {a[i]:R} vs fused {b[i]:R}");
        }
        _output.WriteLine($"{name}: bit-exact over {n} elems");
    }

    private void AssertClose(Tensor expected, Tensor actual, float tol, string name)
    {
        float* a = (float*)expected.DataPointer;
        long n = expected.ElementCount;
        double maxErr = 0;
        long maxIdx = 0;
        for (long i = 0; i < n; i++)
        {
            double e = Math.Abs(a[i] - ReadAs(actual, i));
            if (e > maxErr) { maxErr = e; maxIdx = i; }
        }
        _output.WriteLine($"{name}: max_err={maxErr:E3} @ {maxIdx} over {n} elems (tol {tol:E0})");
        Assert.True(maxErr <= tol, $"{name}: max_err {maxErr:E3} exceeds tol {tol:E0} at index {maxIdx}");
    }

    /// <summary>Host read of one element as float, for F32/F16/BF16 result tensors.</summary>
    private static float ReadAs(Tensor t, long i)
    {
        if (t.DType == DType.F32) return ((float*)t.DataPointer)[i];
        if (t.DType == DType.F16) return (float)((Half*)t.DataPointer)[i];
        return BitConverter.Int32BitsToSingle(((ushort*)t.DataPointer)[i] << 16);
    }

    /// <summary>Runs a CUDA op and forces the lazy device-to-host sync while the backend is still alive.</summary>
    private static void RunCuda(Action<CudaBackend> op, params Tensor[] outputs)
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        op(cuda);
        cuda.Sync();
        foreach (Tensor t in outputs)
            _ = *(byte*)t.DataPointer;
    }

    [Fact]
    public void AffineBroadcastRowIndexed_Cpu_IsBitExact_Vs_UnfusedGatherSequence()
    {
        using Tensor input = Random(new TensorShape(Seq, 1, Hidden), seed: 41);
        using Tensor scaleTable = Random(new TensorShape(ModRows, Hidden), seed: 42);
        using Tensor shiftTable = Random(new TensorShape(ModRows, Hidden), seed: 43);
        using Tensor rowIndex = RowIndex(Seq, ModRows, seed: 44);

        foreach (bool withShift in new[] { true, false })
        {
            Tensor? shift = withShift ? shiftTable : null;
            using Tensor unfused = new Tensor(input.Shape, DType.F32);
            using Tensor fused = new Tensor(input.Shape, DType.F32);

            IBackend cpu = new CpuBackend();
            UnfusedAffine(cpu, unfused, input, scaleTable, shift, rowIndex);
            cpu.AffineBroadcastRowIndexed(fused, input, scaleTable, shift, rowIndex);
            cpu.Dispose();

            AssertBitExact(unfused, fused, $"Affine(shift={withShift})");
        }
    }

    [Fact]
    public void GatedResidualRowIndexed_Cpu_IsBitExact_Vs_UnfusedGatherSequence()
    {
        using Tensor residual = Random(new TensorShape(Seq, 1, Hidden), seed: 51);
        using Tensor value = Random(new TensorShape(Seq, 1, Hidden), seed: 52);
        using Tensor gateTable = Random(new TensorShape(ModRows, Hidden), seed: 53);
        using Tensor rowIndex = RowIndex(Seq, ModRows, seed: 54);
        using Tensor unfused = new Tensor(residual.Shape, DType.F32);
        using Tensor fused = new Tensor(residual.Shape, DType.F32);

        IBackend cpu = new CpuBackend();
        UnfusedGate(cpu, unfused, residual, value, gateTable, rowIndex);
        cpu.GatedResidualRowIndexed(fused, residual, value, gateTable, rowIndex);
        cpu.Dispose();

        AssertBitExact(unfused, fused, "GatedResidual");
    }

    [Fact]
    public void RowIndexedKernels_Cuda_MatchUnfusedCpuReference()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using Tensor input = Random(new TensorShape(Seq, 1, Hidden), seed: 61);
        using Tensor value = Random(new TensorShape(Seq, 1, Hidden), seed: 62);
        using Tensor scaleTable = Random(new TensorShape(ModRows, Hidden), seed: 63);
        using Tensor shiftTable = Random(new TensorShape(ModRows, Hidden), seed: 64);
        using Tensor gateTable = Random(new TensorShape(ModRows, Hidden), seed: 65);
        using Tensor rowIndex = RowIndex(Seq, ModRows, seed: 66);

        foreach (bool withShift in new[] { true, false })
        {
            Tensor? shift = withShift ? shiftTable : null;
            using Tensor cpuOut = new Tensor(input.Shape, DType.F32);
            using Tensor cudaOut = new Tensor(input.Shape, DType.F32);

            IBackend cpu = new CpuBackend();
            UnfusedAffine(cpu, cpuOut, input, scaleTable, shift, rowIndex);
            cpu.Dispose();

            RunCuda(c => c.AffineBroadcastRowIndexed(cudaOut, input, scaleTable, shift, rowIndex), cudaOut);
            AssertClose(cpuOut, cudaOut, 1e-5f, $"CudaAffine(shift={withShift})");
        }

        using (Tensor cpuOut = new Tensor(input.Shape, DType.F32))
        using (Tensor cudaOut = new Tensor(input.Shape, DType.F32))
        {
            IBackend cpu = new CpuBackend();
            UnfusedGate(cpu, cpuOut, input, value, gateTable, rowIndex);
            cpu.Dispose();

            RunCuda(c => c.GatedResidualRowIndexed(cudaOut, input, value, gateTable, rowIndex), cudaOut);
            AssertClose(cpuOut, cudaOut, 1e-5f, "CudaGatedResidual");
        }
    }

    [Fact]
    public void RowIndexedKernels_F16AndBf16_MatchF32Twins()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using Tensor input = Random(new TensorShape(Seq, 1, Hidden), seed: 71);
        using Tensor value = Random(new TensorShape(Seq, 1, Hidden), seed: 72);
        using Tensor scaleTable = Random(new TensorShape(ModRows, Hidden), seed: 73);
        using Tensor shiftTable = Random(new TensorShape(ModRows, Hidden), seed: 74);
        using Tensor gateTable = Random(new TensorShape(ModRows, Hidden), seed: 75);
        using Tensor rowIndex = RowIndex(Seq, ModRows, seed: 76);

        // The 16-bit inputs are the F32 masters rounded, so the only error is I/O rounding (kernels
        // accumulate in F32): ~2^-11 relative for F16, ~2^-8 for BF16, over |values| ≲ 3.
        using Tensor input16 = input.CastTo(DType.F16);
        using Tensor inputBf = input.CastTo(DType.BF16);
        using Tensor value16 = value.CastTo(DType.F16);
        using Tensor valueBf = value.CastTo(DType.BF16);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend b = cuda;

        using (Tensor o32 = new Tensor(input.Shape, DType.F32))
        using (Tensor o16 = new Tensor(input.Shape, DType.F16))
        using (Tensor oBf = new Tensor(input.Shape, DType.BF16))
        {
            b.AffineBroadcastRowIndexed(o32, input, scaleTable, shiftTable, rowIndex);
            b.AffineBroadcastRowIndexed(o16, input16, scaleTable, shiftTable, rowIndex);
            b.AffineBroadcastRowIndexed(oBf, inputBf, scaleTable, shiftTable, rowIndex);
            cuda.Sync();
            AssertClose(o32, o16, 4e-3f, "Affine F16");
            AssertClose(o32, oBf, 3e-2f, "Affine BF16");
        }

        using (Tensor o32 = new Tensor(input.Shape, DType.F32))
        using (Tensor o16 = new Tensor(input.Shape, DType.F16))
        using (Tensor oBf = new Tensor(input.Shape, DType.BF16))
        {
            b.GatedResidualRowIndexed(o32, input, value, gateTable, rowIndex);
            b.GatedResidualRowIndexed(o16, input16, value16, gateTable, rowIndex);
            b.GatedResidualRowIndexed(oBf, inputBf, valueBf, gateTable, rowIndex);
            cuda.Sync();
            AssertClose(o32, o16, 4e-3f, "GatedResidual F16");
            AssertClose(o32, oBf, 3e-2f, "GatedResidual BF16");
        }
    }

    /// <summary>The BF16 twins of the non-row-indexed adaLN pair, which a BF16 DiT body still needs where the
    /// modulation is one broadcast row per call (a final layer's per-segment head) rather than one per token.</summary>
    [Fact]
    public void LastDimKernels_Bf16_MatchF32Twins()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int B = 2, S = 9;
        TensorShape shape = new TensorShape(B, S, Hidden);
        using Tensor input = Random(shape, seed: 91);
        using Tensor value = Random(shape, seed: 92);
        using Tensor scale = Random(new TensorShape(B, Hidden), seed: 93);
        using Tensor shift = Random(new TensorShape(B, Hidden), seed: 94);
        using Tensor gate = Random(new TensorShape(B, Hidden), seed: 95);
        using Tensor inputBf = input.CastTo(DType.BF16);
        using Tensor valueBf = value.CastTo(DType.BF16);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend b = cuda;

        foreach (bool withShift in new[] { true, false })
        {
            Tensor? sh = withShift ? shift : null;
            using Tensor o32 = new Tensor(shape, DType.F32);
            using Tensor oBf = new Tensor(shape, DType.BF16);
            b.AffineBroadcastLastDim(o32, input, scale, sh);
            b.AffineBroadcastLastDim(oBf, inputBf, scale, sh);
            cuda.Sync();
            AssertClose(o32, oBf, 3e-2f, $"LastDim Affine BF16(shift={withShift})");
        }

        using (Tensor o32 = new Tensor(shape, DType.F32))
        using (Tensor oBf = new Tensor(shape, DType.BF16))
        {
            b.GatedResidualLastDim(o32, input, value, gate);
            b.GatedResidualLastDim(oBf, inputBf, valueBf, gate);
            cuda.Sync();
            AssertClose(o32, oBf, 3e-2f, "LastDim GatedResidual BF16");
        }
    }

    [Fact]
    public void RowIndexed_RejectsBadIndexTensor()
    {
        using Tensor input = Random(new TensorShape(Seq, 1, Hidden), seed: 81);
        using Tensor scaleTable = Random(new TensorShape(ModRows, Hidden), seed: 82);
        using Tensor output = new Tensor(input.Shape, DType.F32);
        using Tensor shortIndex = RowIndex(Seq - 1, ModRows, seed: 83);
        using Tensor f32Index = new Tensor(new TensorShape(Seq), DType.F32);

        IBackend cpu = new CpuBackend();
        Assert.Throws<ArgumentException>(() => cpu.AffineBroadcastRowIndexed(output, input, scaleTable, null, shortIndex));
        Assert.Throws<NotSupportedException>(() => cpu.AffineBroadcastRowIndexed(output, input, scaleTable, null, f32Index));
        cpu.Dispose();
    }
}
