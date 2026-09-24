using System.Diagnostics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

/// <summary>Which kernel family a GEMM ran on; the engagement counters and their tests read it.</summary>
internal enum GemmKernel { CoopMat2, CoopMat, Tiled }

/// <summary>One GEMM on device buffers — <c>C[M,N] = alpha · op(A) · op(B) + beta · C (+ bias)</c>, every operand a buffer
/// handle plus an element offset — so a batched product, an attention score tile and a column tile of a convolution are
/// the same call as a Linear layer. <see cref="Dtype"/> is what A and B hold and what the tiled kernel writes; the
/// cooperative-matrix kernels write F16 or F32 per <see cref="OutputDtype"/>.</summary>
internal readonly record struct GemmOperands(
    ulong A, ulong B, ulong C, long M, long N, long K, bool TransposeA, bool TransposeB, DType Dtype, DType OutputDtype)
{
    public uint AOffset { get; init; }
    public uint BOffset { get; init; }
    public uint COffset { get; init; }
    public float Alpha { get; init; } = 1f;
    public float Beta { get; init; }
    /// <summary>Leading dimension of C; a column tile of a wider output sets it to the full width.</summary>
    public long Ldc { get; init; } = N;
    /// <summary>Bias as F32, the form the cooperative-matrix kernels read; 0 for none.</summary>
    public ulong BiasF32 { get; init; }
    /// <summary>Bias in <see cref="Dtype"/>, the form the tiled kernel reads; 0 for none.</summary>
    public ulong BiasOut { get; init; }
    public bool HasBias => BiasF32 != 0 || BiasOut != 0;
    public long Lda => TransposeA ? M : K;
    public long Ldb => TransposeB ? K : N;
}

public sealed partial class VulkanBackend
{
    /// <summary>Runs one GEMM on the fastest kernel its operands admit: coopmat2 (F16, the Linear-layer transpose pair,
    /// no alpha or beta), then coopmat (F16, N and K multiples of 16, any transpose pair; an unaligned M only without
    /// transposeA and without beta), then the tiled kernel. A bias rides in whichever form the chosen kernel reads, so a caller with one
    /// supplies both forms. The tiled kernel writes <see cref="GemmOperands.Dtype"/>, so an output of another dtype
    /// must reach a cooperative-matrix kernel.</summary>
    internal GemmKernel DispatchGemm(in GemmOperands g)
    {
        bool outputSupported = g.OutputDtype == DType.F16 || g.OutputDtype == DType.F32;
        if (g.Dtype == DType.F16 && outputSupported && (!g.HasBias || g.BiasF32 != 0))
        {
            if (EnableCoopMat2 && Vk.HasCooperativeMatrix2 && !g.TransposeA && g.TransposeB && g.Alpha == 1f && g.Beta == 0f)
            {
                long t0 = _profiler.IsEnabled ? Stopwatch.GetTimestamp() : 0;
                DispatchCoopMat2(in g);
                if (_profiler.IsEnabled) _coopmat2GemmTicks += Stopwatch.GetTimestamp() - t0;
                _coopmat2GemmCount++;
                return GemmKernel.CoopMat2;
            }
            // An M off the 16-row fragment runs the partial-M shader, which takes no transposed A and adds no beta·C.
            if (!_disableCoopmat && Vk.HasCooperativeMatrix && g.N % 16 == 0 && g.K % 16 == 0 && (g.M % 16 == 0 || (!g.TransposeA && g.Beta == 0f)))
            {
                long t0 = _profiler.IsEnabled ? Stopwatch.GetTimestamp() : 0;
                DispatchCoopMat(in g);
                if (_profiler.IsEnabled) _coopmatGemmTicks += Stopwatch.GetTimestamp() - t0;
                _coopmatGemmCount++;
                return GemmKernel.CoopMat;
            }
        }
        if (g.OutputDtype != g.Dtype)
            throw new NotSupportedException(
                $"The tiled GEMM writes {g.Dtype.Name}; a {g.OutputDtype.Name} output needs a cooperative-matrix kernel, which this shape or device does not admit.");
        long tTiled0 = _profiler.IsEnabled ? Stopwatch.GetTimestamp() : 0;
        DispatchTiled(in g);
        if (_profiler.IsEnabled) _tiledGemmTicks += Stopwatch.GetTimestamp() - tTiled0;
        _tiledGemmCount++;
        return GemmKernel.Tiled;
    }

    /// <summary>The 11-word push block every GEMM shader shares: M, N, K, lda, ldb, ldc, alpha, beta, aOffset, bOffset, cOffset (coopmat2 omits alpha and beta).</summary>
    private static void WriteGemmPush(Span<byte> pc, in GemmOperands g, bool withScalars)
    {
        int o = 0;
        BinaryWriteUInt(pc, o, (uint)g.M); o += 4;
        BinaryWriteUInt(pc, o, (uint)g.N); o += 4;
        BinaryWriteUInt(pc, o, (uint)g.K); o += 4;
        BinaryWriteUInt(pc, o, (uint)g.Lda); o += 4;
        BinaryWriteUInt(pc, o, (uint)g.Ldb); o += 4;
        BinaryWriteUInt(pc, o, (uint)g.Ldc); o += 4;
        if (withScalars)
        {
            BinaryWriteFloat(pc, o, g.Alpha); o += 4;
            BinaryWriteFloat(pc, o, g.Beta); o += 4;
        }
        BinaryWriteUInt(pc, o, g.AOffset); o += 4;
        BinaryWriteUInt(pc, o, g.BOffset); o += 4;
        BinaryWriteUInt(pc, o, g.COffset);
    }

    /// <summary>VK_NV_cooperative_matrix2: no alignment requirement (the hardware clamp drops out-of-bounds accesses); the [M,K] × [N,K]ᵀ pair only.</summary>
    private void DispatchCoopMat2(in GemmOperands g)
    {
        uint BM = Vk.CoopMat2MGranularity;
        uint BN = Vk.CoopMat2NGranularity;
        const uint BK = 64u;
        bool outputIsF32 = g.OutputDtype == DType.F32;
        ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
        {
            SpecConstant.UInt(0, Vk.CoopMat2WorkgroupInvocations),
            SpecConstant.UInt(1, 1),
            SpecConstant.UInt(2, 1),
            SpecConstant.UInt(10, BM),
            SpecConstant.UInt(11, BN),
            SpecConstant.UInt(12, BK),
            SpecConstant.Bool(15, outputIsF32),
            SpecConstant.Bool(16, g.BiasF32 != 0),
        };
        VulkanKernel k = GetKernel("matmul_coopmat2", storageBufferCount: 5, spec);
        Span<byte> pc = stackalloc byte[9 * 4];
        WriteGemmPush(pc, in g, withScalars: false);
        // HAS_BIAS=false never reads binding 4; A stands in so the layout stays five buffers.
        Span<ulong> bufs = stackalloc ulong[] { g.A, g.B, g.C, g.C, g.BiasF32 != 0 ? g.BiasF32 : g.A };
        Dispatch(k, bufs, pc, (uint)((g.N + BN - 1) / BN), (uint)((g.M + BM - 1) / BM), 1);
    }

    /// <summary>VK_KHR_cooperative_matrix: 16×16×16 F16 fragments with an F32 accumulator; the workgroup tile shrinks until its subgroups fit the device.</summary>
    private void DispatchCoopMat(in GemmOperands g)
    {
        const uint FRAG = 16;
        bool mAligned = g.M % FRAG == 0;
        uint BM = g.M >= 64 ? 64u : (g.M >= 32 ? 32u : 16u);
        uint BN = g.N >= 64 ? 64u : (g.N >= 32 ? 32u : 16u);
        uint subgroupSize = Vk.SubgroupSize;
        uint maxInvocations = Math.Min(Vk.MaxComputeWorkGroupInvocations, Vk.MaxComputeWorkGroupSizeX);
        while ((BM > FRAG || BN > FRAG) &&
               ((BM / FRAG) * (BN / FRAG) * subgroupSize > maxInvocations ||
                (!mAligned && (BM * FRAG * 2 + BM * BN * 4) > Vk.MaxComputeSharedMemoryBytes)))
        {
            if (BN >= BM && BN > FRAG) BN /= 2;
            else if (BM > FRAG) BM /= 2;
            else BN /= 2;
        }
        uint localX = (BM / FRAG) * (BN / FRAG) * subgroupSize;
        ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
        {
            SpecConstant.UInt(0, localX),
            SpecConstant.UInt(1, 1),
            SpecConstant.UInt(2, 1),
            SpecConstant.UInt(10, BM),
            SpecConstant.UInt(11, BN),
            SpecConstant.UInt(12, subgroupSize),
            SpecConstant.Bool(13, g.TransposeA),
            SpecConstant.Bool(14, g.TransposeB),
            SpecConstant.Bool(15, g.OutputDtype == DType.F32),
            SpecConstant.Bool(16, g.BiasF32 != 0),
        };
        VulkanKernel k = GetKernel(mAligned ? "matmul_coopmat" : "matmul_coopmat_partial_m", storageBufferCount: 5, spec);
        Span<byte> pc = stackalloc byte[11 * 4];
        WriteGemmPush(pc, in g, withScalars: true);
        Span<ulong> bufs = stackalloc ulong[] { g.A, g.B, g.C, g.BiasF32 != 0 ? g.BiasF32 : g.C, g.C };
        Dispatch(k, bufs, pc, (uint)((g.N + BN - 1) / BN), (uint)((g.M + BM - 1) / BM), 1);
    }

    /// <summary>The register-tiled fallback that runs on any device, in F16 or F32.</summary>
    private void DispatchTiled(in GemmOperands g)
    {
        (uint BM, uint BN, uint BK, uint TM, uint TN) = PickMatmulTile(g.M, g.N, g.K);
        ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
        {
            SpecConstant.UInt(0, BN / TN),
            SpecConstant.UInt(1, BM / TM),
            SpecConstant.UInt(2, 1),
            SpecConstant.UInt(10, BM),
            SpecConstant.UInt(11, BN),
            SpecConstant.UInt(12, BK),
            SpecConstant.UInt(13, TM),
            SpecConstant.UInt(14, TN),
            SpecConstant.Bool(15, g.TransposeA),
            SpecConstant.Bool(16, g.TransposeB),
            SpecConstant.Bool(17, g.BiasOut != 0),
            SpecConstant.UInt(18, 0u),    // no fused activation
            SpecConstant.Bool(19, false), // no fused residual
        };
        VulkanKernel k = GetKernel("matmul_tiled" + DtypeSuffix(g.Dtype), storageBufferCount: 5, spec);
        Span<byte> pc = stackalloc byte[11 * 4];
        WriteGemmPush(pc, in g, withScalars: true);
        Span<ulong> bufs = stackalloc ulong[] { g.A, g.B, g.C, g.BiasOut != 0 ? g.BiasOut : g.C, g.C };
        Dispatch(k, bufs, pc, (uint)((g.N + BN - 1) / BN), (uint)((g.M + BM - 1) / BM), 1);
    }
}
