using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan.Authoring;

namespace HartsyInference.Vulkan;

public sealed partial class VulkanBackend
{
    /// <summary>The GGUF block formats <c>gemv_quant</c> reads directly; the loader keeps exactly these compressed on this
    /// backend (<see cref="Core.Backends.BackendCapabilities.QuantizedTypes"/>) and widens every other format to F32.</summary>
    internal static readonly IReadOnlySet<DType> QuantGemvTypes = new HashSet<DType> { DType.Q8_0, DType.Q4_K, DType.Q6_K };

    /// <summary>Most activation rows one <c>gemv_quant</c> dispatch takes (a specialization constant of the shader, one pipeline per count): decode and a short
    /// speculative verify. Longer inputs dequantize the weight per call and run the GEMM.</summary>
    internal const int QuantGemvMaxRows = 8;

    private long _quantGemvCount;

    /// <summary>Dispatches of <c>gemv_quant</c> so far; tests read it to prove a Linear took the fused path.</summary>
    internal long QuantGemvCount => _quantGemvCount;

    /// <summary>Linear on a quantized weight at a decode-sized M: <c>gemv_quant</c> straight from the GGUF blocks, F32 in
    /// and out. Declines (returns false) anything it does not cover so the caller takes the dequantize-then-GEMM path.</summary>
    private bool TryDispatchQuantGemv(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        if (!QuantGemvTypes.Contains(weight.DType) || weight.Shape.Rank != 2) return false;
        if (input.DType != DType.F32 || output.DType != DType.F32) return false;
        long n = weight.Shape[0], k = weight.Shape[1];
        if (input.Shape[input.Shape.Rank - 1] != k || k % weight.DType.BlockElementCount != 0) return false;
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
            _ => throw new NotSupportedException($"gemv_quant has no {dtype.Name} decoder; supported: Q8_0, Q4_K, Q6_K."),
        };
        if (n < 1 || k < 1) throw new ArgumentOutOfRangeException(nameof(n), $"gemv_quant needs N, K >= 1; got N={n}, K={k}.");
        if (k % dtype.BlockElementCount != 0)
            throw new ArgumentException($"gemv_quant {dtype.Name}: K={k} is not a multiple of the {dtype.BlockElementCount}-element block.", nameof(k));
        if (m < 1 || m > QuantGemvMaxRows)
            throw new ArgumentOutOfRangeException(nameof(m), $"gemv_quant takes 1..{QuantGemvMaxRows} activation rows; got {m}.");

        // A lane per work unit up to 128, never below a subgroup; the shader cuts a Q8_0, Q4_K and Q6_K block into 8, 16 and 16 units.
        int unitsPerBlock = dtype == DType.Q8_0 ? 8 : 16;
        long units = (long)k / dtype.BlockElementCount * unitsPerBlock;
        uint subgroup = Math.Max(1u, Vk.SubgroupSize);
        uint wg = subgroup;
        while (wg < 128 && wg < units) wg *= 2;

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
