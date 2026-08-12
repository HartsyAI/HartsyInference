using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Nvfp4;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Parity for the RESIDENT nvfp4 <c>Linear</c> path — an <see cref="DType.F4E2M1"/> weight kept packed at
/// 0.5 byte/param and unpacked transiently by <c>dequant_nvfp4_to_f16</c> — against the host reference already
/// validated on <c>qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors</c>.
///
/// <para><b>How the weight is observed.</b> The kernel writes into a transient device buffer that no public API
/// hands back, so every case feeds the GEMM a <c>K×K</c> IDENTITY activation:
/// <c>out[i, n] = Σ_k I[i,k]·W[n,k] = W[n,i]</c>, making the F32 output the transposed dequantized weight the
/// backend actually used. That exercises the whole dispatch (attach → relabel → scale upload → kernel → cuBLAS)
/// rather than the kernel in isolation.</para>
///
/// <para><b>Signed zero.</b> The readback cannot preserve a negative zero — cuBLAS accumulates <c>-0.0·1.0</c> into
/// a <c>+0.0</c> accumulator and IEEE gives <c>+0.0</c> — and E2M1 nibble 8 is <c>-0.0</c>, so ~1/16 of a random
/// packing lands there. Those are counted separately and their total must equal the reference's own count of
/// <c>-0.0</c> words, so a kernel that zeroed something else is still caught. The sign itself is gated on the host
/// by <c>Nvfp4ResidentCodecParityTests</c>.</para></summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class Nvfp4ResidentCudaParityTests
{
    private readonly ITestOutputHelper _output;

    public Nvfp4ResidentCudaParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>BF16 bit pattern of negative zero.</summary>
    private const ushort NegativeZeroBf16 = 0x8000;

    private static string Qwen3VlNvfp4Path => Path.Combine(TestPaths.ModelsDir, "text_encoders",
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors");

    /// <summary>Size of the complete checkpoint. A partial download still passes <c>File.Exists</c> and would fail
    /// as a parity error rather than as the missing asset it is.</summary>
    private const long Qwen3VlNvfp4Bytes = 15_687_142_551L;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    /// <summary>Opens a backend, or returns null after logging why the case is skipped.</summary>
    /// <remarks>The PTX check is not redundant with the device check: without the module <c>CanRunResidentNvfp4</c>
    /// silently reroutes to the host dequant, which compares equal against the same reference — the test would pass
    /// while never running the kernel it exists to gate.</remarks>
    private CudaBackend? TryOpenBackend()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return null; }
        string ptxDir = PtxDir();
        if (!File.Exists(Path.Combine(ptxDir, "dequant_nvfp4_to_f16.ptx")))
        {
            _output.WriteLine($"SKIPPED: dequant_nvfp4_to_f16.ptx not in {ptxDir}");
            return null;
        }
        CudaBackend backend = new CudaBackend(0, ptxDir);
        _output.WriteLine($"device: {backend.Context.DeviceName} "
            + $"(SM {backend.Context.ComputeCapabilityMajor}.{backend.Context.ComputeCapabilityMinor})");
        return backend;
    }

    [Theory]
    [InlineData(100, 48)]     // rows pad 100->128 AND block columns pad 3->4
    [InlineData(100, 256)]    // padded rows only
    [InlineData(128, 48)]     // padded block columns only
    [InlineData(256, 256)]    // sweeps all 256 E4M3 scale bytes
    public void GpuDequant_IsBitIdenticalToTheHostReference(int n, int k)
    {
        using CudaBackend? backend = TryOpenBackend();
        if (backend is null) return;   // tier-lint: guarded
        using PackedWeight weight = PackedWeight.Synthetic(n, k, seed: 20260812 + n * 31 + k);
        AssertBitIdentical(backend, weight, $"synthetic [{n}x{k}]");
    }

    [Theory]
    [InlineData("model.layers.0.self_attn.k_proj", 1024, 5120)]
    [InlineData("model.layers.10.self_attn.q_proj", 8192, 5120)]
    public void RealCheckpointWeight_GpuDequant_IsBitIdenticalToTheHostReference(string prefix, int n, int k)
    {
        using CudaBackend? backend = TryOpenBackend();
        if (backend is null) return;   // tier-lint: guarded
        using SafeTensorsLoader? loader = TryOpenCheckpoint();
        if (loader is null) return;    // tier-lint: guarded
        using PackedWeight weight = PackedWeight.FromCheckpoint(loader, prefix, n, k);
        AssertBitIdentical(backend, weight, $"real {prefix} [{n}x{k}]");
    }

    [Theory]
    [InlineData(100, 48)]
    [InlineData(256, 256)]
    public void F16GemmPath_MatchesTheExactHostDequant_WithinF16Rounding(int n, int k)
    {
        using CudaBackend? backend = TryOpenBackend();
        if (backend is null) return;   // tier-lint: guarded
        using PackedWeight weight = PackedWeight.Synthetic(n, k, seed: 4242 + n + k);

        float[] exact = Nvfp4HostReference.ExactF32(weight.Packed, weight.BlockScale, weight.GlobalScale);
        float[] got = IdentityGemm(backend, weight, DType.F16);

        double squaredError = 0, squaredReference = 0, maxAbs = 0;
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < k; col++)
            {
                double diff = got[(long)col * n + row] - exact[(long)row * k + col];
                squaredError += diff * diff;
                squaredReference += (double)exact[(long)row * k + col] * exact[(long)row * k + col];
                maxAbs = Math.Max(maxAbs, Math.Abs(diff));
            }
        }
        double relL2 = squaredReference > 0 ? Math.Sqrt(squaredError / squaredReference) : Math.Sqrt(squaredError);
        _output.WriteLine($"nvfp4 F16 gemm [{n}x{k}]: relL2={relL2:E3} maxAbs={maxAbs:E3} (vs the un-narrowed host dequant)");
        // F16 keeps 11 significant bits, so an exactly-rounded store is bounded by 2^-11 relative per element; the
        // gate sits an order of magnitude above the RMS of that and far below what a dropped scale term would give.
        Assert.True(relL2 < 1e-3, $"[{n}x{k}] F16 dequant relL2={relL2:E3} exceeds 1e-3");
    }

    [Fact]
    public void LinearWeightRows_TakesTheHostFallback_AndMatchesTheReferenceRows()
    {
        using CudaBackend? backend = TryOpenBackend();
        if (backend is null) return;   // tier-lint: guarded
        const int N = 256, K = 256, RowOffset = 64, RowCount = 64;
        using PackedWeight weight = PackedWeight.Synthetic(N, K, seed: 909);
        ushort[] reference = Nvfp4HostReference.Bf16Words(weight.Packed, weight.BlockScale, weight.GlobalScale);

        using Tensor resident = weight.AttachResident();
        using Tensor identity = Identity(K, DType.F32);
        using Tensor output = new Tensor(new TensorShape(K, RowCount), DType.F32);
        try
        {
            // A row range is a shape CanRunResidentNvfp4 refuses, so this drives the host-dequant branch at the top
            // of LinearImpl rather than the kernel — the path that keeps such a layer runnable at all.
            backend.LinearWeightRows(output, identity, resident, null, RowOffset, RowCount);
            backend.Sync();
            ReadOnlySpan<float> got = output.AsReadOnlySpan<float>();

            double maxAbs = 0;
            for (int row = 0; row < RowCount; row++)
                for (int col = 0; col < K; col++)
                {
                    float expected = Bf16ToF32(reference[(RowOffset + row) * K + col]);
                    maxAbs = Math.Max(maxAbs, Math.Abs(got[col * RowCount + row] - expected));
                }
            _output.WriteLine($"nvfp4 host-dequant fallback rows [{RowOffset}, {RowOffset + RowCount}): maxAbs={maxAbs:E3}");
            Assert.True(maxAbs == 0.0, $"host-dequant fallback rows differ from the reference: maxAbs={maxAbs:E3}");
        }
        finally
        {
            backend.FreeWeights([resident]);
        }
    }

    [Fact]
    public void CachedWeightCast_ServesASecondLinearWithTheSameResidentWeight()
    {
        using CudaBackend? backend = TryOpenBackend();
        if (backend is null) return;   // tier-lint: guarded
        const int N = 256, K = 256;
        using PackedWeight weight = PackedWeight.Synthetic(N, K, seed: 1717);
        ushort[] reference = Nvfp4HostReference.Bf16Words(weight.Packed, weight.BlockScale, weight.GlobalScale);

        using Tensor resident = weight.AttachResident();
        try
        {
            // Preloading is what makes LinearImpl take the CACHED weight-cast branch: the transient branch would
            // re-run the kernel every call and the second pass would prove nothing about the cache.
            backend.PreloadWeights([resident]);
            Assert.True(GpuTransferHelper.IsWeightCached(resident), "PreloadWeights did not register the packed weight.");

            using Tensor identity = Identity(K, DType.F32);
            for (int pass = 1; pass <= 2; pass++)
            {
                using Tensor output = new Tensor(new TensorShape(K, N), DType.F32);
                backend.Linear(output, identity, resident, null);
                backend.Sync();
                bool cast = GpuTransferHelper.TryGetWeightCast(resident, out ulong _);
                (long mismatches, long signedZeroOnly) = CompareBitExact(output.AsReadOnlySpan<float>(), reference, N, K);
                _output.WriteLine($"cached-cast pass {pass}: cast-cached={cast} real mismatches={mismatches} "
                    + $"signed-zero-only={signedZeroOnly}");
                Assert.Equal(0L, mismatches);
            }
        }
        finally
        {
            backend.FreeWeights([resident]);
        }
    }

    private void AssertBitIdentical(CudaBackend backend, PackedWeight weight, string label)
    {
        ushort[] reference = Nvfp4HostReference.Bf16Words(weight.Packed, weight.BlockScale, weight.GlobalScale);
        float[] got = IdentityGemm(backend, weight, DType.F32);

        (long mismatches, long signedZeroOnly) = CompareBitExact(got, reference, weight.OutFeatures, weight.InFeatures);
        long negativeZeros = 0;
        foreach (ushort word in reference) if (word == NegativeZeroBf16) negativeZeros++;

        _output.WriteLine($"nvfp4 BF16 gemm {label}: real mismatches={mismatches} / {reference.LongLength}, "
            + $"signed-zero-only={signedZeroOnly}, reference -0.0 words={negativeZeros}");
        Assert.Equal(0L, mismatches);
        Assert.Equal(negativeZeros, signedZeroOnly);
    }

    /// <summary>Bit-exact comparison of an identity-GEMM readback against the reference BF16 words, returning the
    /// real mismatch count and (separately) the negative zeros the GEMM cannot carry.</summary>
    private static (long Mismatches, long SignedZeroOnly) CompareBitExact(ReadOnlySpan<float> got, ushort[] reference,
        int n, int k)
    {
        long mismatches = 0, signedZeroOnly = 0;
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < k; col++)
            {
                ushort expected = reference[(long)row * k + col];
                float value = got[col * n + row];
                if (BitConverter.SingleToUInt32Bits(value) >> 16 == expected) continue;
                if (expected == NegativeZeroBf16 && value == 0f) signedZeroOnly++;
                else mismatches++;
            }
        }
        return (mismatches, signedZeroOnly);
    }

    /// <summary>Runs <c>identity · Wᵀ</c> and returns the F32 output <c>[K, N]</c>, which IS the dequantized weight
    /// the backend used, transposed.</summary>
    private static float[] IdentityGemm(CudaBackend backend, PackedWeight weight, DType activationDType)
    {
        using Tensor resident = weight.AttachResident();
        try
        {
            using Tensor identity = Identity(weight.InFeatures, activationDType);
            using Tensor output = new Tensor(new TensorShape(weight.InFeatures, weight.OutFeatures), DType.F32);
            backend.Linear(output, identity, resident, null);
            backend.Sync();
            return output.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            backend.FreeWeights([resident]);
        }
    }

    private static Tensor Identity(int size, DType dtype)
    {
        Tensor identity = new Tensor(new TensorShape(size, size), dtype);
        if (dtype == DType.F32)
        {
            float* p = (float*)identity.DataPointer;
            for (int i = 0; i < size; i++) p[(long)i * size + i] = 1f;
        }
        else
        {
            Half* p = (Half*)identity.DataPointer;
            for (int i = 0; i < size; i++) p[(long)i * size + i] = (Half)1f;
        }
        return identity;
    }

    private static float Bf16ToF32(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);

    private SafeTensorsLoader? TryOpenCheckpoint()
    {
        string path = Qwen3VlNvfp4Path;
        if (!RealWeightGate.Require(_output.WriteLine, path)) return null;
        long length = new FileInfo(path).Length;
        if (length < Qwen3VlNvfp4Bytes)
        {
            _output.WriteLine($"SKIPPED: {path} is {length} bytes, expected {Qwen3VlNvfp4Bytes} — partial download");
            return null;
        }
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        return loader;
    }

    /// <summary>A packed nvfp4 weight plus its two scale companions, in the on-disk U8 <c>[N, K/2]</c> form.</summary>
    private sealed class PackedWeight : IDisposable
    {
        public required Tensor Packed { get; init; }
        public required Tensor BlockScale { get; init; }
        public required Tensor GlobalScale { get; init; }
        public required int OutFeatures { get; init; }
        public required int InFeatures { get; init; }

        public static PackedWeight Synthetic(int n, int k, int seed)
        {
            int paddedRows = (n + 127) / 128 * 128;
            int paddedCols = (k / Nvfp4Codec.GroupSize + 3) / 4 * 4;
            byte[] packedBytes = new byte[(long)n * (k / 2)];
            new Random(seed).NextBytes(packedBytes);
            byte[] scaleBytes = new byte[(long)paddedRows * paddedCols];
            // Every E4M3 byte value, so the kernel's hand-rolled decode (it cannot use the SM 8.9 fp8 intrinsic) is
            // exercised on subnormals and on the 480 maximum, not just the band a checkpoint happens to use.
            for (int i = 0; i < scaleBytes.Length; i++) scaleBytes[i] = (byte)(i & 0xFF);

            return new PackedWeight
            {
                Packed = FromBytes(packedBytes, new TensorShape(n, k / 2), DType.U8),
                BlockScale = FromBytes(scaleBytes, new TensorShape(paddedRows, paddedCols), DType.F8E4M3),
                GlobalScale = Scalar(0.37f),
                OutFeatures = n,
                InFeatures = k,
            };
        }

        public static PackedWeight FromCheckpoint(SafeTensorsLoader loader, string prefix, int n, int k)
        {
            Tensor packed = loader.GetTensor($"{prefix}.weight");
            Tensor blockScale = loader.GetTensor($"{prefix}.weight_scale");
            if (packed.DType != DType.U8 || packed.Shape[0] != n || packed.Shape[1] != k / 2)
                throw new InvalidOperationException($"{prefix}.weight is {packed.DType} {packed.Shape}, expected U8 [{n}, {k / 2}].");
            if (blockScale.DType != DType.F8E4M3)
                throw new InvalidOperationException($"{prefix}.weight_scale is {blockScale.DType}, expected F8E4M3.");

            using Tensor scalar = loader.GetTensor($"{prefix}.weight_scale_2");
            return new PackedWeight
            {
                Packed = packed,
                BlockScale = blockScale,
                GlobalScale = Scalar(((float*)scalar.DataPointer)[0]),
                OutFeatures = n,
                InFeatures = k,
            };
        }

        /// <summary>Relabels through the production entry point, so a gate that changed its mind about these shapes
        /// fails here instead of quietly measuring the eager path.</summary>
        public Tensor AttachResident()
        {
            if (!Nvfp4Codec.TryAttachResident(Packed, BlockScale, GlobalScale, hasPreQuantScale: false, out Tensor resident))
                throw new InvalidOperationException("Nvfp4Codec.TryAttachResident refused a weight the resident path must serve.");
            return resident;
        }

        public void Dispose()
        {
            Packed.Dispose();
            BlockScale.Dispose();
            GlobalScale.Dispose();
        }

        private static Tensor FromBytes(ReadOnlySpan<byte> source, TensorShape shape, DType dtype)
        {
            Tensor tensor = new Tensor(shape, dtype);
            source.CopyTo(new Span<byte>(tensor.DataPointer, source.Length));
            return tensor;
        }

        private static Tensor Scalar(float value)
        {
            Tensor tensor = new Tensor(new TensorShape(1), DType.F32);
            ((float*)tensor.DataPointer)[0] = value;
            return tensor;
        }
    }
}
