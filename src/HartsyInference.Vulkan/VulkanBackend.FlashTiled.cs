using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// The query-tiled F32 flash kernel (sdpa_flash_tiled) behind numerics.vkTiledAttention. A head-dim bucket picks the SPIR-V
// module (MAX_D 64, 128, 256). Each module compiles its query and key tile heights in (build.sh); TiledQueryRows and
// TiledKeyRows mirror them here, so the two change together.
public sealed partial class VulkanBackend
{
    /// <summary>The largest head dim the tiled kernel serves: the 256 bucket.</summary>
    internal const int TiledMaxHeadDim = 256;

    private static readonly string[] TiledModules = ["sdpa_flash_tiled_d64", "sdpa_flash_tiled_d128", "sdpa_flash_tiled_d256"];
    private static readonly string[] TiledBiasModules = ["sdpa_flash_tiled_d64_bias", "sdpa_flash_tiled_d128_bias", "sdpa_flash_tiled_d256_bias"];

    private bool? _tiledModulesPresent;

    /// <summary>Whether dense F32 attention takes the query-tiled kernel ahead of the scalar route.</summary>
    public bool EnableTiledAttention { get; set; } = ParseTiledAttention(EngineKnobs.VkTiledAttention.Value);

    /// <summary>Head-major [B, H, S, D] strides: each head spans <paramref name="tokens"/> rows.</summary>
    internal static TiledStrides HeadMajorStrides(int heads, int tokens, int headDim) =>
        new((uint)headDim, (uint)((long)tokens * headDim), (uint)((long)heads * tokens * headDim));

    /// <summary>Token-major [B, S, H, D] strides: each token spans every head.</summary>
    internal static TiledStrides TokenMajorStrides(int heads, int tokens, int headDim) =>
        new((uint)heads * (uint)headDim, (uint)headDim, (uint)((long)tokens * heads * headDim));

    /// <summary>Runs <c>sdpa_flash_tiled</c> over F32 Q, K, V and an optional F32 additive bias. The strides address every tensor,
    /// so head-major and token-major are the same launch; the output is written with the O strides.</summary>
    internal void DispatchFlashTiled(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? bias, in TiledCall call)
    {
        using OpScope _op = EnterOp();
        int bucket = TiledBucket(call.HeadDim);
        if (bucket < 0)
            throw new NotSupportedException($"sdpa_flash_tiled serves head dims 1..{TiledMaxHeadDim}; got {call.HeadDim}.");

        VulkanBuffer qBuf = GetBuffer(query);
        VulkanBuffer kBuf = GetBuffer(key);
        VulkanBuffer vBuf = GetBuffer(value);
        VulkanBuffer? biasBuf = bias is null ? null : GetBuffer(bias);
        ulong outBytes = (ulong)call.Batch * (ulong)call.Hq * (ulong)call.Sq * (ulong)call.HeadDim * sizeof(float);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outBytes);
        try
        {
            VulkanKernel k = biasBuf is null
                ? GetKernel(TiledModules[bucket], storageBufferCount: 4, ReadOnlySpan<SpecConstant>.Empty)
                : GetKernel(TiledBiasModules[bucket], storageBufferCount: 5, ReadOnlySpan<SpecConstant>.Empty);
            Span<byte> pc = stackalloc byte[biasBuf is null ? 88 : 100];
            WriteTiledPushConstants(pc, in call, bias);
            Span<ulong> bufs = biasBuf is null
                ? stackalloc ulong[] { qBuf.Handle, kBuf.Handle, vBuf.Handle, outBuf.Handle }
                : stackalloc ulong[] { qBuf.Handle, kBuf.Handle, vBuf.Handle, biasBuf.Handle, outBuf.Handle };
            uint rows = (uint)TiledQueryRows(bucket);
            Dispatch(k, bufs, pc, ((uint)call.Sq + rows - 1) / rows, (uint)call.Hq, (uint)call.Batch);
            CacheOutputCastingFrom(output, outBuf, DType.F32, "sdpa_flash_tiled");
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan sdpa_flash_tiled dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    /// <summary>Whether the query-tiled kernel takes this dense call: the knob is auto, every tensor is F32, the head dim has a bucket,
    /// the device has the shared memory and the grid, and the modules are shipped.</summary>
    private bool TiledF32Admits(in AttentionCall c)
    {
        if (!EnableTiledAttention) return false;
        int bucket = TiledBucket(c.QkHeadDim);
        bool f32 = c.Query.DType == DType.F32 && c.Key.DType == DType.F32 && c.Value.DType == DType.F32 && c.Output.DType == DType.F32;
        if (bucket < 0 || !f32 || c.KvHeads <= 0 || c.Heads % c.KvHeads != 0) return false;
        if (c.Mask.Bias is not null && c.Mask.Bias.DType != DType.F32) return false;
        if (TiledSharedBytes(bucket) > Vk.MaxComputeSharedMemoryBytes || !TiledModulesShipped()) return false;
        uint groupsX = ((uint)c.QueryTokens + (uint)TiledQueryRows(bucket) - 1) / (uint)TiledQueryRows(bucket);
        return groupsX <= Vk.MaxComputeWorkGroupCountX && (uint)c.Heads <= Vk.MaxComputeWorkGroupCountY
            && (uint)c.Batch <= Vk.MaxComputeWorkGroupCountZ;
    }

    /// <summary>Head-dim bucket: 0 serves 1..64, 1 serves 65..128, 2 serves 129..256; -1 is outside the kernel.</summary>
    internal static int TiledBucket(int headDim) =>
        headDim <= 0 || headDim > TiledMaxHeadDim ? -1 : headDim <= 64 ? 0 : headDim <= 128 ? 1 : 2;

    private static int TiledQueryRows(int bucket) => bucket == 2 ? 8 : 16;

    private static int TiledKeyRows(int bucket) => bucket == 2 ? 16 : 32;

    /// <summary>Shared memory one workgroup declares: the query tile, the key and value tiles, the score block and the two
    /// reduction rows, with the padded row stride the shader uses.</summary>
    private static ulong TiledSharedBytes(int bucket)
    {
        int br = TiledQueryRows(bucket), tk = TiledKeyRows(bucket), stride = (64 << bucket) + 1;
        return (ulong)(4 * (br * stride + 2 * tk * stride + br * (tk + 1) + 2 * br * 8));
    }

    private static bool ParseTiledAttention(string? value) => value switch
    {
        "auto" => true,
        "off" => false,
        _ => throw new ArgumentException($"numerics.vkTiledAttention must be 'auto' or 'off'; got '{value}'."),
    };

    private bool TiledModulesShipped()
    {
        if (_tiledModulesPresent is null)
        {
            bool present = true;
            for (int i = 0; i < TiledModules.Length; i++)
                present &= _kernels.HasModule(TiledModules[i]) && _kernels.HasModule(TiledBiasModules[i]);
            _tiledModulesPresent = present;
        }
        return _tiledModulesPresent.Value;
    }

    /// <summary>The push block: sdpa_flash's prefix, the Q/K/V/O strides, then the bias strides when a bias is bound.</summary>
    private static void WriteTiledPushConstants(Span<byte> pc, in TiledCall c, Tensor? bias)
    {
        BinaryWriteUInt(pc, 0, (uint)c.Hq);
        BinaryWriteUInt(pc, 4, (uint)c.Hkv);
        BinaryWriteUInt(pc, 8, (uint)c.Sq);
        BinaryWriteUInt(pc, 12, (uint)c.Skv);
        BinaryWriteUInt(pc, 16, (uint)c.Skv);
        BinaryWriteUInt(pc, 20, (uint)c.HeadDim);
        BinaryWriteFloat(pc, 24, c.Scale);
        BinaryWriteUInt(pc, 28, c.Causal ? 1u : 0u);
        BinaryWriteUInt(pc, 32, (uint)c.QOffset);
        BinaryWriteUInt(pc, 36, (uint)c.Window);
        WriteTiledStrides(pc, 40, c.Q);
        WriteTiledStrides(pc, 52, c.K);
        WriteTiledStrides(pc, 64, c.V);
        WriteTiledStrides(pc, 76, c.O);
        if (bias is null) return;
        AttentionMaskLayout m = AttentionMaskLayout.Of(bias, c.Batch, c.Hq, c.Sq, c.Skv);
        BinaryWriteUInt(pc, 88, (uint)m.BatchStride);
        BinaryWriteUInt(pc, 92, (uint)m.HeadStride);
        BinaryWriteUInt(pc, 96, (uint)m.RowStride);
    }

    private static void WriteTiledStrides(Span<byte> pc, int offset, TiledStrides s)
    {
        BinaryWriteUInt(pc, offset, s.Row);
        BinaryWriteUInt(pc, offset + 4, s.Head);
        BinaryWriteUInt(pc, offset + 8, s.Batch);
    }

    /// <summary>Element strides of one tensor: between tokens (rows), between heads and between batch items. The innermost dim is contiguous.</summary>
    internal readonly record struct TiledStrides(uint Row, uint Head, uint Batch);

    /// <summary>One query-tiled launch: the shape, the causal and window mask, the scale and the strides of Q, K, V and O.</summary>
    internal readonly record struct TiledCall(int Batch, int Hq, int Hkv, int Sq, int Skv, int HeadDim, float Scale, bool Causal,
        int QOffset, int Window, TiledStrides Q, TiledStrides K, TiledStrides V, TiledStrides O);
}
