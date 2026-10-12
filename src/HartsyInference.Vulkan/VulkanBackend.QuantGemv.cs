using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan.Authoring;

namespace HartsyInference.Vulkan;

public sealed partial class VulkanBackend
{
    /// <summary>The GGUF block formats <c>gemv_quant</c> reads directly; the loader keeps exactly these compressed on this
    /// backend (<see cref="Core.Backends.BackendCapabilities.QuantizedTypes"/>) and widens every other format to F32.</summary>
    internal static readonly IReadOnlySet<DType> QuantGemvTypes = new HashSet<DType> { DType.Q8_0, DType.Q4_0, DType.Q5_0, DType.Q4_K, DType.Q5_K, DType.Q6_K };

    /// <summary>Most activation rows one <c>gemv_quant</c> dispatch takes (a specialization constant of the shader, one pipeline per count): decode and a
    /// speculative verify (the prompt-lookup draft is 8 tokens, so a round is 9 rows). Longer inputs dequantize the weight per call and run the GEMM.
    /// Sixteen matches <c>MoeFeedForward.IndexedMaxTokens</c>, so a verify round is on the fused path in the dense and the MoE layers alike.</summary>
    internal const int QuantGemvMaxRows = 16;

    private long _quantGemvCount;

    /// <summary>Dispatches of <c>gemv_quant</c> so far; tests read it to prove a Linear took the fused path.</summary>
    internal long QuantGemvCount => _quantGemvCount;

    /// <summary>Elements the shader reads as one unit of an F32 weight row (a vec4); K must be a multiple of it.</summary>
    private const int F32GemvUnitElements = 4;

    /// <summary>Elements of one block of the weight format <c>gemv_quant</c> reads: a GGUF block, or the vec4 unit of an F32 matrix.</summary>
    internal static int GemvBlockElements(DType dtype) => dtype == DType.F32 ? F32GemvUnitElements : dtype.BlockElementCount;

    /// <summary>Linear on a quantized or F32 weight at a decode-sized M: <c>gemv_quant</c> straight from the GGUF blocks (or the
    /// float matrix, which is what a MoE router weight is), F32 in and out. Declines (returns false) anything it does not cover so
    /// the caller takes the dequantize-then-GEMM path.</summary>
    private bool TryDispatchQuantGemv(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        if (!(QuantGemvTypes.Contains(weight.DType) || weight.DType == DType.F32) || weight.Shape.Rank != 2) return false;
        if (input.DType != DType.F32 || output.DType != DType.F32) return false;
        long n = weight.Shape[0], k = weight.Shape[1];
        // K comes from the weight: the decode graph feeds the o-projection the head-major attention output
        // [1, heads, 1, headDim], whose last dim is the head size, not K (the rows are the same bytes as [1, 1, heads*headDim]).
        if (input.ElementCount % k != 0 || k % GemvBlockElements(weight.DType) != 0) return false;
        long m = input.ElementCount / k;
        if (m < 1 || m > QuantGemvMaxRows || output.ElementCount != m * n) return false;
        if (bias is not null && (bias.DType != DType.F32 || bias.ElementCount != n)) return false;

        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(m * n * sizeof(float)));
        try
        {
            DispatchQuantGemv(weight.DType, GetBuffer(weight).Handle, GetBuffer(input).Handle, outBuf.Handle,
                bias is null ? 0 : GetBuffer(bias).Handle, (int)n, (int)k, (int)m);
            CacheOutput(output, outBuf);
            return true;
        }
        catch
        {
            outBuf.Dispose();
            throw;
        }
    }

    /// <summary>Work units per block as <c>gemv_quant_unit.glsl</c> cuts them (<c>UNITS_PER_BLOCK</c>).</summary>
    internal static int GemvUnitsPerBlock(DType dtype) =>
        dtype == DType.F32 ? 1 : dtype == DType.Q8_0 ? 8 : dtype == DType.Q4_0 || dtype == DType.Q5_0 ? 4 : 16;

    /// <summary>Records one <c>gemv_quant_*</c> dispatch: <c>out[m, n] = x[m, :] · W[n, :] (+ bias[n])</c> with W the
    /// <paramref name="n"/> rows of <paramref name="k"/> quantized elements at the start of <paramref name="weight"/>.
    /// <paramref name="bias"/> 0 means none.</summary>
    internal void DispatchQuantGemv(DType dtype, ulong weight, ulong x, ulong output, ulong bias, int n, int k, int m)
    {
        string shader = dtype.Name switch
        {
            "Q8_0" => "gemv_quant_q8_0",
            "Q4_K" => "gemv_quant_q4_k",
            "Q6_K" => "gemv_quant_q6_k",
            "Q4_0" => "gemv_quant_q4_0",
            "Q5_0" => "gemv_quant_q5_0",
            "Q5_K" => "gemv_quant_q5_k",
            "F32" => "gemv_quant_f32",
            _ => throw new NotSupportedException($"gemv_quant has no {dtype.Name} decoder; supported: Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K, F32."),
        };
        if (n < 1 || k < 1) throw new ArgumentOutOfRangeException(nameof(n), $"gemv_quant needs N, K >= 1; got N={n}, K={k}.");
        if (k % GemvBlockElements(dtype) != 0)
            throw new ArgumentException($"gemv_quant {dtype.Name}: K={k} is not a multiple of the {GemvBlockElements(dtype)}-element block.", nameof(k));
        if (m < 1 || m > QuantGemvMaxRows)
            throw new ArgumentOutOfRangeException(nameof(m), $"gemv_quant takes 1..{QuantGemvMaxRows} activation rows; got {m}.");

        // A lane per work unit up to 128, never below a subgroup; units per block as the shader cuts them.
        long units = (long)k / GemvBlockElements(dtype) * GemvUnitsPerBlock(dtype);
        uint subgroup = Math.Max(1u, Vk.SubgroupSize);
        uint wg = subgroup;
        // Lane r writes activation row r's total, so the workgroup has at least M lanes.
        while (wg < 128 && (wg < units || wg < m)) wg *= 2;
        // The shader sums one partial per subgroup through 32 shared slots per activation row.
        wg = Math.Min(wg, 32 * subgroup);

        VulkanKernel kernel = GetKernel(shader, storageBufferCount: 4, new SpecConstant[]
        {
            SpecConstant.UInt(0, wg),
            SpecConstant.Bool(1, bias != 0),
            SpecConstant.UInt(2, (uint)m),
        });
        Span<byte> raw = stackalloc byte[8];
        PushConstants pc = new(raw);
        pc.U32((uint)n);
        pc.U32((uint)k);
        // HAS_BIAS=false never reads binding 3; the output stands in so the layout stays four buffers.
        Span<ulong> bufs = stackalloc ulong[] { weight, x, output, bias != 0 ? bias : output };
        const uint MaxGroupsPerDim = 65535;
        uint gx = (uint)Math.Min(n, MaxGroupsPerDim);
        uint gy = (uint)((n + gx - 1) / gx);
        Dispatch(kernel, bufs, pc.Written, gx, gy, 1);
        _quantGemvCount++;
    }
}
