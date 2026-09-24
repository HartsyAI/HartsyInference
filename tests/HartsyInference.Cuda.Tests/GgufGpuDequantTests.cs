using System.Linq;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Gguf.Codecs;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>GPU-side dequant accuracy tests. For each supported quant DType, builds a synthetic block with known values, dequantizes via both the CPU codec (canonical) and the GPU PTX kernel, then compares element-wise. Tolerance for GPU vs CPU is avg_abs_err &lt; 1e-3 (F16-precision; the underlying math is identical, only F32 → F16 narrowing differs).
///
/// <para>Tests skip cleanly when CUDA is unavailable. Each test creates its own <see cref="CudaBackend"/> + <see cref="CudaKernels"/> — running them in a separate xunit collection forces serial execution to avoid context contention with other CUDA test classes.</para></summary>
[Collection("CudaSerial")]
public sealed class GgufGpuDequantTests
{
    private readonly ITestOutputHelper _output;
    public GgufGpuDequantTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void Q8_0_GpuDequant_MatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int blocks = 4;
        const int totalElems = blocks * 32;
        Tensor src = new Tensor(new TensorShape(totalElems), DType.Q8_0);
        try
        {
            FillQ8_0(src, blocks);
            using Tensor cpuRef = GgufDequantizer.Dequantize(src, DType.F16);
            using Tensor gpuOut = RunGpuDequant(src, totalElems);
            CompareF16(cpuRef, gpuOut, totalElems, tolerance: 1e-3f);
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q4_0_GpuDequant_MatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int blocks = 4;
        const int totalElems = blocks * 32;
        Tensor src = new Tensor(new TensorShape(totalElems), DType.Q4_0);
        try
        {
            FillQ4_0(src, blocks);
            using Tensor cpuRef = GgufDequantizer.Dequantize(src, DType.F16);
            using Tensor gpuOut = RunGpuDequant(src, totalElems);
            CompareF16(cpuRef, gpuOut, totalElems, tolerance: 1e-3f);
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q5_0_GpuDequant_MatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int blocks = 4;
        const int totalElems = blocks * 32;
        Tensor src = new Tensor(new TensorShape(totalElems), DType.Q5_0);
        try
        {
            FillQ5_0(src, blocks);
            using Tensor cpuRef = GgufDequantizer.Dequantize(src, DType.F16);
            using Tensor gpuOut = RunGpuDequant(src, totalElems);
            CompareF16(cpuRef, gpuOut, totalElems, tolerance: 1e-3f);
        }
        finally { src.Dispose(); }
    }

    /// <summary>Q2_K is the smallest K-quant, and the reason it is worth a kernel: a 6.7 GB MiniMax-H3 build against
    /// 21 GB at Q8_0. Widened on the host instead it expands roughly sixfold, which puts a model that would fit a
    /// 12 GB card out of reach of a 24 GB one.</summary>
    [Fact]
    public unsafe void Q2_K_GpuDequant_MatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int superBlocks = 3;
        const int totalElems = superBlocks * 256;
        Tensor src = new Tensor(new TensorShape(totalElems), DType.Q2_K);
        try
        {
            FillQ2_K(src, superBlocks);
            // A fixture whose mins were all zero would certify a kernel that dropped the dmin term entirely.
            byte* scales = (byte*)src.DataPointer;
            Assert.Contains(Enumerable.Range(0, 16), i => (scales[i] >> 4) != 0);
            using Tensor cpuRef = GgufDequantizer.Dequantize(src, DType.F16);
            using Tensor gpuOut = RunGpuDequant(src, totalElems);
            CompareF16(cpuRef, gpuOut, totalElems, tolerance: 1e-3f);
        }
        finally { src.Dispose(); }
    }

    /// <summary>Q3_K's high bit is stored INVERTED — a set mask bit means "do not subtract 4" — which is the single
    /// easiest thing to get backwards in this layout, and a sign error there is not visibly wrong, just wrong.</summary>
    [Fact]
    public unsafe void Q3_K_GpuDequant_MatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int superBlocks = 3;
        const int totalElems = superBlocks * 256;
        Tensor src = new Tensor(new TensorShape(totalElems), DType.Q3_K);
        try
        {
            FillQ3_K(src, superBlocks);
            AssertQ3KFixtureExercisesBothHighBitBranches(src, superBlocks);
            using Tensor cpuRef = GgufDequantizer.Dequantize(src, DType.F16);
            using Tensor gpuOut = RunGpuDequant(src, totalElems);
            CompareF16(cpuRef, gpuOut, totalElems, tolerance: 1e-3f);
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q4_K_GpuDequant_MatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int superBlocks = 2;
        const int totalElems = superBlocks * 256;
        Tensor src = new Tensor(new TensorShape(totalElems), DType.Q4_K);
        try
        {
            FillQ4_K(src, superBlocks);
            using Tensor cpuRef = GgufDequantizer.Dequantize(src, DType.F16);
            using Tensor gpuOut = RunGpuDequant(src, totalElems);
            CompareF16(cpuRef, gpuOut, totalElems, tolerance: 1e-3f);
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void IQ4_XS_GpuDequant_MatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int superBlocks = 3;
        const int totalElems = superBlocks * 256;
        Tensor src = new Tensor(new TensorShape(totalElems), DType.IQ4_XS);
        try
        {
            Random rng = new Random(53);
            byte* p = (byte*)src.DataPointer;
            for (int sb = 0; sb < superBlocks; sb++)
            {
                byte* block = p + sb * 136;
                ushort d = (ushort)(0x3400 + rng.Next(0x800));   // F16 in [0.25, 2)
                block[0] = (byte)d; block[1] = (byte)(d >> 8);
                for (int i = 2; i < 136; i++) block[i] = (byte)rng.Next(256);
            }
            using Tensor cpuRef = GgufDequantizer.Dequantize(src, DType.F16);
            using Tensor gpuOut = RunGpuDequant(src, totalElems);
            CompareF16(cpuRef, gpuOut, totalElems, tolerance: 1e-3f);
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void IQ4_NL_GpuDequant_MatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int blocks = 24;
        const int totalElems = blocks * 32;
        Tensor src = new Tensor(new TensorShape(totalElems), DType.IQ4_NL);
        try
        {
            Random rng = new Random(59);
            byte* p = (byte*)src.DataPointer;
            for (int b = 0; b < blocks; b++)
            {
                byte* block = p + b * 18;
                ushort d = (ushort)(0x3400 + rng.Next(0x800));   // F16 in [0.25, 2)
                block[0] = (byte)d; block[1] = (byte)(d >> 8);
                for (int i = 2; i < 18; i++) block[i] = (byte)rng.Next(256);
            }
            using Tensor cpuRef = GgufDequantizer.Dequantize(src, DType.F16);
            using Tensor gpuOut = RunGpuDequant(src, totalElems);
            CompareF16(cpuRef, gpuOut, totalElems, tolerance: 1e-3f);
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q5_K_GpuDequant_MatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int superBlocks = 2;
        const int totalElems = superBlocks * 256;
        Tensor src = new Tensor(new TensorShape(totalElems), DType.Q5_K);
        try
        {
            FillQ5_K(src, superBlocks);
            using Tensor cpuRef = GgufDequantizer.Dequantize(src, DType.F16);
            using Tensor gpuOut = RunGpuDequant(src, totalElems);
            CompareF16(cpuRef, gpuOut, totalElems, tolerance: 1e-3f);
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void Q6_K_GpuDequant_MatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int superBlocks = 2;
        const int totalElems = superBlocks * 256;
        Tensor src = new Tensor(new TensorShape(totalElems), DType.Q6_K);
        try
        {
            FillQ6_K(src, superBlocks);
            using Tensor cpuRef = GgufDequantizer.Dequantize(src, DType.F16);
            using Tensor gpuOut = RunGpuDequant(src, totalElems);
            CompareF16(cpuRef, gpuOut, totalElems, tolerance: 1e-3f);
        }
        finally { src.Dispose(); }
    }

    [Fact]
    public unsafe void EndToEnd_QuantizeOnCpu_GpuDequant_MatchesOriginal()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int totalElems = 1024;
        Tensor original = new Tensor(new TensorShape(totalElems), DType.F32);
        Tensor q4 = new Tensor(new TensorShape(totalElems), DType.Q4_K);
        try
        {
            float* op = (float*)original.DataPointer;
            Random rng = new Random(7);
            for (int i = 0; i < totalElems; i++) op[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            IGgufCodec codec = GgufCodecRegistry.Get(DType.Q4_K);
            codec.QuantizeFromF32(op, (byte*)q4.DataPointer, totalElems);

            using Tensor cpuRef = GgufDequantizer.Dequantize(q4, DType.F16);
            using Tensor gpuOut = RunGpuDequant(q4, totalElems);
            CompareF16(cpuRef, gpuOut, totalElems, tolerance: 1e-3f);
        }
        finally
        {
            original.Dispose();
            q4.Dispose();
        }
    }

    private static unsafe Tensor RunGpuDequant(Tensor src, int totalElems)
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        using CudaBackend backend = new CudaBackend(0, ptxDir);
        using CudaKernels kernels = new CudaKernels(ptxDir);

        nuint srcBytes = (nuint)src.DType.ComputeByteCount(totalElems);
        nuint dstBytes = (nuint)(totalElems * DType.F16.SizeInBytes);

        ulong devSrc = CudaMemory.Allocate(srcBytes);
        ulong devDst = CudaMemory.Allocate(dstBytes);
        try
        {
            CudaMemory.CopyHostToDevice(devSrc, (void*)src.DataPointer, srcBytes);
            nint stream = backend.Stream.Handle;

            kernels.LaunchGgufDequantToF16(src.DType, devDst, devSrc, totalElems, stream);
            backend.Sync();

            Tensor result = new Tensor(new TensorShape(totalElems), DType.F16);
            CudaMemory.CopyDeviceToHost((void*)result.DataPointer, devDst, dstBytes);
            return result;
        }
        finally
        {
            CudaMemory.Free(devSrc);
            CudaMemory.Free(devDst);
        }
    }

    private static unsafe void CompareF16(Tensor cpuRef, Tensor gpuOut, int count, float tolerance)
    {
        Half* cp = (Half*)cpuRef.DataPointer;
        Half* gp = (Half*)gpuOut.DataPointer;
        double sumAbs = 0.0;
        float maxAbs = 0f;
        for (int i = 0; i < count; i++)
        {
            float c = (float)cp[i];
            float g = (float)gp[i];
            float err = MathF.Abs(c - g);
            sumAbs += err;
            if (err > maxAbs) maxAbs = err;
        }
        float avgErr = (float)(sumAbs / count);
        Assert.True(avgErr < tolerance, $"GPU vs CPU avg_err {avgErr:E3} exceeds tolerance {tolerance:E3}; max_err = {maxAbs:E3}");
    }

    private static unsafe void FillQ8_0(Tensor t, int blocks)
    {
        byte* p = (byte*)t.DataPointer;
        for (int b = 0; b < blocks; b++)
        {
            byte* block = p + b * 34;
            *(Half*)block = (Half)(0.5f + 0.1f * b);
            sbyte* qs = (sbyte*)(block + 2);
            for (int i = 0; i < 32; i++) qs[i] = (sbyte)((i + b) - 16);
        }
    }

    private static unsafe void FillQ4_0(Tensor t, int blocks)
    {
        byte* p = (byte*)t.DataPointer;
        for (int b = 0; b < blocks; b++)
        {
            byte* block = p + b * 18;
            *(Half*)block = (Half)(0.25f + 0.1f * b);
            byte* qs = block + 2;
            for (int i = 0; i < 16; i++) qs[i] = (byte)((i * 7 + b * 13) & 0xFF);   // mix both nibbles
        }
    }

    private static unsafe void FillQ5_0(Tensor t, int blocks)
    {
        byte* p = (byte*)t.DataPointer;
        for (int b = 0; b < blocks; b++)
        {
            byte* block = p + b * 22;
            *(Half*)block = (Half)(0.3f + 0.05f * b);
            uint qh = 0xA5A5_5A5Au ^ (uint)(b * 0x1234567);   // exercise the 5th-bit path across all 32 positions
            block[2] = (byte)qh; block[3] = (byte)(qh >> 8); block[4] = (byte)(qh >> 16); block[5] = (byte)(qh >> 24);
            byte* qs = block + 6;
            for (int i = 0; i < 16; i++) qs[i] = (byte)((i * 11 + b * 17) & 0xFF);
        }
    }

    /// <summary>Proves the Q3_K fixture actually reaches both sides of the inverted high bit. Without this a kernel
    /// that always subtracted 4 — or never did — could match the CPU codec on a fixture that only ever took one
    /// branch, and the test would certify a sign error.</summary>
    private static unsafe void AssertQ3KFixtureExercisesBothHighBitBranches(Tensor t, int superBlocks)
    {
        byte* p = (byte*)t.DataPointer;
        bool sawSet = false, sawClear = false;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            byte* hmask = p + sb * 110;
            for (int element = 0; element < 256; element++)
            {
                int h = element >> 7, r = element & 127, j = r >> 5, o = r & 31;
                bool bit = (hmask[16 * (o >> 4) + (o & 15)] & (1u << (4 * h + j))) != 0;
                if (bit) sawSet = true; else sawClear = true;
            }
        }
        Assert.True(sawSet && sawClear, "The Q3_K fixture must produce both high-bit branches.");
    }

    /// <summary>Q2_K super-block: 16 scale bytes (4-bit scale | 4-bit min), 64 quant bytes, then d and dmin.
    /// Varied per super-block so a kernel that ignored the block index would not pass.</summary>
    private static unsafe void FillQ2_K(Tensor t, int superBlocks)
    {
        byte* p = (byte*)t.DataPointer;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            byte* block = p + sb * 84;
            for (int i = 0; i < 16; i++) block[i] = (byte)(0x13 + i * 7 + sb);
            byte* qs = block + 16;
            for (int i = 0; i < 64; i++) qs[i] = (byte)(0x6C + (i * 5 & 0x7F) + sb);
            *(Half*)(block + 80) = (Half)(0.75f + sb * 0.25f);
            *(Half*)(block + 82) = (Half)(0.125f * (sb + 1));
        }
    }

    /// <summary>Q3_K super-block: 32 hmask bytes, 64 quant bytes, 12 packed 6-bit signed scales, then d. The hmask
    /// pattern deliberately sets a mix of bits so both branches of the inverted high bit are exercised.</summary>
    private static unsafe void FillQ3_K(Tensor t, int superBlocks)
    {
        byte* p = (byte*)t.DataPointer;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            byte* block = p + sb * 110;
            for (int i = 0; i < 32; i++) block[i] = (byte)(0x5A + i * 3 + sb);
            byte* qs = block + 32;
            for (int i = 0; i < 64; i++) qs[i] = (byte)(0x27 + (i * 11 & 0x7F) + sb);
            byte* scales = block + 96;
            for (int i = 0; i < 12; i++) scales[i] = (byte)(0x1D + i * 13 + sb);
            *(Half*)(block + 108) = (Half)(0.5f + sb * 0.125f);
        }
    }

    private static unsafe void FillQ4_K(Tensor t, int superBlocks)
    {
        byte* p = (byte*)t.DataPointer;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            byte* block = p + sb * 144;
            *(Half*)block = (Half)2.0f;
            *(Half*)(block + 2) = (Half)0.5f;
            byte* scales = block + 4;
            for (int i = 0; i < 12; i++) scales[i] = (byte)(0x21 + i);
            byte* qs = block + 16;
            for (int i = 0; i < 128; i++) qs[i] = (byte)(0x53 + (i & 0x3F));
        }
    }

    private static unsafe void FillQ5_K(Tensor t, int superBlocks)
    {
        byte* p = (byte*)t.DataPointer;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            byte* block = p + sb * 176;
            *(Half*)block = (Half)1.5f;
            *(Half*)(block + 2) = (Half)0.25f;
            byte* scales = block + 4;
            for (int i = 0; i < 12; i++) scales[i] = (byte)(0x12 + i);
            byte* highBits = block + 16;
            for (int i = 0; i < 32; i++) highBits[i] = (byte)(i & 0xFF);
            byte* lowBits = block + 48;
            for (int i = 0; i < 128; i++) lowBits[i] = (byte)((0x37 + i) & 0xFF);
        }
    }

    private static unsafe void FillQ6_K(Tensor t, int superBlocks)
    {
        byte* p = (byte*)t.DataPointer;
        for (int sb = 0; sb < superBlocks; sb++)
        {
            byte* block = p + sb * 210;
            byte* ql = block;
            byte* qh = block + 128;
            sbyte* scales = (sbyte*)(block + 192);
            for (int i = 0; i < 128; i++) ql[i] = (byte)((0x12 + i) & 0xFF);
            for (int i = 0; i < 64; i++) qh[i] = (byte)((0x34 + i) & 0xFF);
            for (int i = 0; i < 16; i++) scales[i] = (sbyte)(i + 1);
            *(Half*)(block + 208) = (Half)0.125f;
        }
    }
}

[CollectionDefinition("CudaSerial", DisableParallelization = true)]
public class CudaSerialCollection { }
