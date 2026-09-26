using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

public sealed partial class VulkanBackend
{
    /// <summary>Upper bound on the first absmax pass's workgroups; each strides over the tensor, the finalize folds them.</summary>
    private const uint Fp8AbsMaxMaxBlocks = 1024;

    /// <summary>Devices whose fp8 status line has been logged; a backend is rebuilt per test and per auto-select probe.</summary>
    private static readonly HashSet<string> _fp8StatusLogged = new();

    /// <summary>Logs once per device whether fp8 Linears run on fp8 cooperative matrices, and why not. Only a constructed Vulkan
    /// backend reaches this, so a CUDA host stays quiet. A forced <c>numerics.vkFp8=true</c> the device cannot honor warns.</summary>
    private void LogFp8Status(bool? knob)
    {
        lock (_fp8StatusLogged)
        {
            if (!_fp8StatusLogged.Add($"{Vk.DeviceUuid}|{Vk.DeviceName}")) return;
        }
        (bool warn, string message) = Fp8StatusLine(knob, Vk.HasFloat8CooperativeMatrix, Vk.Fp8CoopMatUnavailableReason,
            Vk.Fp8CoopMatM, Vk.Fp8CoopMatN, Vk.Fp8CoopMatK);
        if (warn) Logs.Warning($"[Vulkan] {Vk.DeviceName}: {message}");
        else Logs.Info($"[Vulkan] {Vk.DeviceName}: {message}");
    }

    /// <summary>The fp8 status line and whether it is a warning; split out so every branch is testable without the device.</summary>
    internal static (bool Warn, string Message) Fp8StatusLine(bool? knob, bool available, string? reason, uint m, uint n, uint k)
    {
        const string fallback = "fp8 weights are widened to F16 for each Linear instead";
        if (knob == false) return (false, "fp8 Linear off (numerics.vkFp8=false).");
        if (available) return (false, $"fp8 Linear on (E4M3 {m}x{n}x{k} cooperative matrices).");
        return knob == true
            ? (true, $"numerics.vkFp8 is on, but {reason}; {fallback}.")
            : (false, $"fp8 Linear off: {reason}; {fallback}.");
    }

    /// <summary>Whether <see cref="Linear"/> runs an E4M3 weight on fp8 cooperative matrices (see <see cref="TryDispatchFp8Linear"/>).
    /// Follows <c>numerics.vkFp8</c>, unset meaning on wherever the device offers them; settable for tests.</summary>
    public bool EnableFp8Linear { get; set; }

    /// <summary>Quantizes the activation with the weight's checkpoint <c>.input_scale</c> when it carries one, skipping the absmax
    /// passes — <c>numerics.fp8StaticInputScale</c>, the same switch and default CUDA's fp8 Linear reads.</summary>
    public bool EnableStaticFp8InputScale { get; set; }

    /// <summary>The fp8 Linear, CUDA's native fp8 scheme: the weight stays packed E4M3 with its per-tensor scale folded into alpha,
    /// the activation is quantized to E4M3 per tensor each call (a static checkpoint scale, or absmax/448 computed and read on the
    /// device), and the product accumulates in F32 on <c>matmul_fp8_coopmat</c>. Refuses what the kernel cannot take — an E5M2
    /// or block-scaled weight, a pre-quantized input, or N, K off the device's fragment shape (a ragged M is padded) — and the caller falls through
    /// to the F16-cast path.</summary>
    private bool TryDispatchFp8Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        if (!EnableFp8Linear || !Vk.HasFloat8CooperativeMatrix) return false;
        if (weight.DType != DType.F8E4M3 || weight.Shape.Rank != 2 || weight.QuantInfo?.BlockScale is not null) return false;
        if ((input.DType != DType.F32 && input.DType != DType.F16) || (output.DType != DType.F32 && output.DType != DType.F16)) return false;
        long n = weight.Shape[0], k = weight.Shape[1];
        if (input.Shape[input.Shape.Rank - 1] != k) return false;
        long m = input.ElementCount / k;
        if (output.ElementCount != m * n || (bias is not null && bias.ElementCount != n)) return false;
        if (n % Vk.Fp8CoopMatN != 0 || k % Vk.Fp8CoopMatK != 0) return false;
        // A ragged M is quantized into rows padded to the fragment height; the shader indexes elements in uint.
        long mPad = (m + Vk.Fp8CoopMatM - 1) / Vk.Fp8CoopMatM * Vk.Fp8CoopMatM;
        if (mPad * k > uint.MaxValue || n * k > uint.MaxValue || m * n > uint.MaxValue) return false;

        float staticScale = EnableStaticFp8InputScale && weight.Fp8InputScaleFactor > 0f ? weight.Fp8InputScaleFactor : 0f;
        long count = m * k;
        string suffix = DtypeSuffix(input.DType);
        VulkanBuffer xBuf = GetBuffer(input);
        VulkanBuffer xQ = _xfer.AllocateDevice((ulong)(mPad * k));
        VulkanBuffer? scratch = null, biasOwned = null;
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(output.ElementCount * output.DType.SizeInBytes));
        try
        {
            if (staticScale == 0f)
            {
                uint blocks = (uint)Math.Min(GroupCount(count, LocalX1D), Fp8AbsMaxMaxBlocks);
                scratch = _xfer.AllocateDevice((ulong)((blocks + 1) * sizeof(float)));
                DispatchFp8AbsMax(xBuf.Handle, scratch.Handle, suffix, (uint)count, blocks);
            }
            DispatchQuantE4M3(xBuf.Handle, xQ.Handle, scratch?.Handle ?? xQ.Handle, suffix, (uint)(count / 4), (uint)(mPad * k / 4), staticScale);

            ulong biasF32 = 0;
            if (bias is not null)
            {
                (VulkanBuffer biasRes, biasOwned) = CastIfNeeded(bias, GetBuffer(bias), DType.F32);
                biasF32 = biasRes.Handle;
            }
            float alpha = weight.Fp8ScaleFactor * (staticScale == 0f ? 1f : staticScale);
            DispatchFp8Gemm(xQ.Handle, GetBuffer(weight).Handle, outBuf.Handle, biasF32, scratch?.Handle ?? 0,
                (uint)m, (uint)n, (uint)k, alpha, output.DType == DType.F32);
            CacheOutput(output, outBuf);
            return true;
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan fp8 Linear dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            _xfer.FreeDevice(xQ);
            if (scratch is not null) _xfer.FreeDevice(scratch);
            if (biasOwned is not null) _xfer.FreeDevice(biasOwned);
        }
    }

    /// <summary>Test hook: the activation half of the fp8 Linear alone. Returns <paramref name="x"/> (F32 or F16, a multiple of 4
    /// elements) as E4M3 bytes and the dequant scale they were taken at — <paramref name="staticScale"/>, or absmax/448.</summary>
    internal (byte[] Bytes, float Scale) QuantizeE4M3ForTest(Tensor x, float staticScale)
    {
        using OpScope _ = EnterOp();
        long count = x.ElementCount;
        using Tensor q = new(new TensorShape(count), DType.I8);
        using Tensor scratchT = new(new TensorShape(Fp8AbsMaxMaxBlocks + 1), DType.F32);
        VulkanBuffer qBuf = _xfer.AllocateDevice((ulong)count);
        VulkanBuffer scratch = _xfer.AllocateDevice((Fp8AbsMaxMaxBlocks + 1) * sizeof(float));
        string suffix = DtypeSuffix(x.DType);
        if (staticScale == 0f)
            DispatchFp8AbsMax(GetBuffer(x).Handle, scratch.Handle, suffix, (uint)count, (uint)Math.Min(GroupCount(count, LocalX1D), Fp8AbsMaxMaxBlocks));
        DispatchQuantE4M3(GetBuffer(x).Handle, qBuf.Handle, scratch.Handle, suffix, (uint)(count / 4), (uint)(count / 4), staticScale);
        CacheOutput(q, qBuf);
        CacheOutput(scratchT, scratch);
        Sync();
        return (q.AsReadOnlySpan<byte>().ToArray(), staticScale != 0f ? staticScale : scratchT.AsReadOnlySpan<float>()[0]);
    }

    /// <summary>fp8_absmax's two passes: per-workgroup maxes into <c>scratch[1..]</c>, then the dequant scale into <c>scratch[0]</c>.</summary>
    private void DispatchFp8AbsMax(ulong x, ulong scratch, string suffix, uint count, uint blocks)
    {
        Span<byte> pc = stackalloc byte[2 * 4];
        BinaryWriteUInt(pc, 0, count);
        BinaryWriteUInt(pc, 4, blocks);
        Span<ulong> bufs = stackalloc ulong[] { x, scratch };
        Dispatch(GetKernel("fp8_absmax" + suffix, storageBufferCount: 2,
            new[] { SpecConstant.UInt(0, LocalX1D), SpecConstant.Bool(10, false) }), bufs, pc, blocks, 1, 1);
        bufs[0] = scratch;
        Dispatch(GetKernel("fp8_absmax_f32", storageBufferCount: 2,
            new[] { SpecConstant.UInt(0, LocalX1D), SpecConstant.Bool(10, true) }), bufs, pc, 1, 1, 1);
    }

    /// <summary>quant_e4m3: four E4M3 bytes per word at x / scale, the scale static or read from <c>scale[0]</c>, zero words up to <paramref name="paddedWords"/>.</summary>
    private void DispatchQuantE4M3(ulong x, ulong q, ulong scale, string suffix, uint words, uint paddedWords, float staticScale)
    {
        Span<byte> pc = stackalloc byte[4 * 4];
        BinaryWriteUInt(pc, 0, words);
        BinaryWriteFloat(pc, 4, staticScale);
        BinaryWriteUInt(pc, 8, 0u);
        BinaryWriteUInt(pc, 12, paddedWords);
        Span<ulong> bufs = stackalloc ulong[] { x, q, scale };
        Dispatch(GetKernel("quant_e4m3" + suffix, storageBufferCount: 3, _default1DSpec), bufs, pc, GroupCount(paddedWords, LocalX1D));
    }

    /// <summary>matmul_fp8_coopmat on packed E4M3 operands: 2×2 subgroups, each owning 2×2 fragments of the device's shape.</summary>
    private void DispatchFp8Gemm(ulong a, ulong b, ulong c, ulong biasF32, ulong scale, uint m, uint n, uint k, float alpha, bool outputF32)
    {
        const uint WM = 2, WN = 2, SgRows = 2, SgCols = 2;
        uint fm = Vk.Fp8CoopMatM, fn = Vk.Fp8CoopMatN;
        ReadOnlySpan<SpecConstant> spec = new SpecConstant[]
        {
            SpecConstant.UInt(0, SgRows * SgCols * Vk.SubgroupSize),
            SpecConstant.UInt(10, fm),
            SpecConstant.UInt(11, fn),
            SpecConstant.UInt(12, Vk.Fp8CoopMatK),
            SpecConstant.UInt(13, WM),
            SpecConstant.UInt(14, WN),
            SpecConstant.UInt(15, SgCols),
            SpecConstant.Bool(16, outputF32),
            SpecConstant.Bool(17, biasF32 != 0),
            SpecConstant.Bool(18, scale != 0),
            SpecConstant.UInt(19, SgRows),
        };
        Span<byte> pc = stackalloc byte[5 * 4];
        BinaryWriteUInt(pc, 0, m);
        BinaryWriteUInt(pc, 4, n);
        BinaryWriteUInt(pc, 8, k);
        BinaryWriteFloat(pc, 12, alpha);
        BinaryWriteUInt(pc, 16, 0u);
        // Unused bindings take the output buffer so the layout stays six buffers.
        Span<ulong> bufs = stackalloc ulong[] { a, b, c, biasF32 != 0 ? biasF32 : c, c, scale != 0 ? scale : c };
        uint tileM = SgRows * WM * fm, tileN = SgCols * WN * fn;
        Dispatch(GetKernel("matmul_fp8_coopmat", storageBufferCount: 6, spec), bufs, pc, (n + tileN - 1) / tileN, (m + tileM - 1) / tileM, 1);
    }
}
