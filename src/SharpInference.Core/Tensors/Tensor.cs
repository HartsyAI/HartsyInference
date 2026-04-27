using SharpInference.Core.Backends;
using SharpInference.Core.Memory;

namespace SharpInference.Core.Tensors;

/// <summary>Core tensor type holding a pointer to unmanaged memory, shape, dtype, and device.</summary>
public sealed unsafe class Tensor : IDisposable
{
    private NativeBuffer? _ownedBuffer;
    private nint _dataPointer;

    /// <summary>Backend-set callback: copies GPU→CPU then frees GPU pointer. Invoked lazily on first CPU data access.</summary>
    internal Action? _gpuSyncCallback;

    /// <summary>Backend-set callback: frees GPU pointer without D2H copy. Invoked on Dispose when GPU data was never synced to CPU.</summary>
    internal Action? _gpuDisposeCallback;

    /// <summary>Shape and strides of this tensor.</summary>
    public TensorShape Shape { get; }

    /// <summary>Data type of the tensor elements.</summary>
    public DType DType { get; }

    /// <summary>Device this tensor resides on.</summary>
    public DeviceKind Device { get; }

    /// <summary>Total number of elements across all dimensions.</summary>
    public long ElementCount => Shape.ElementCount;

    /// <summary>Whether this tensor owns its memory or borrows it from a mmap/view.</summary>
    public bool OwnsMemory => _ownedBuffer is not null;

    /// <summary>
    /// Per-tensor scaling factor for ComfyUI-style <c>fp8_scaled</c> quantization, where the real value
    /// of each weight element is <c>fp8_byte_decoded * Fp8ScaleFactor</c>. Default is 1.0, meaning
    /// "no extra scaling" (used for plain FP8 or non-quantized tensors). When non-1, GEMM call sites
    /// fold this into cuBLAS' <c>alpha</c> parameter so the scaling is applied for free during the matmul.
    /// Set during checkpoint conversion for fp8_scaled formats; left at 1.0 otherwise.
    /// </summary>
    public float Fp8ScaleFactor { get; set; } = 1.0f;

    /// <summary>Pointer to the raw tensor data. If GPU data is cached, triggers a lazy sync (D2H copy) first.</summary>
    public void* DataPointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            EnsureCpuData();
            nint ptr = _dataPointer;
            if (ptr == 0)
                throw new ObjectDisposedException(nameof(Tensor));
            return (void*)ptr;
        }
    }

    /// <summary>If GPU data is cached on this tensor, syncs it to CPU and clears the callbacks.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EnsureCpuData()
    {
        Action? sync = _gpuSyncCallback;
        if (sync != null)
        {
            _gpuSyncCallback = null;
            _gpuDisposeCallback = null;
            sync();
        }
    }

    /// <summary>Creates a new tensor with freshly allocated unmanaged memory, zeroed.</summary>
    public Tensor(TensorShape shape, DType dtype, DeviceKind device = default)
    {
        Shape = shape;
        DType = dtype;
        Device = device;

        long byteSize = dtype.ComputeByteCount(shape.ElementCount);
        _ownedBuffer = new NativeBuffer((nuint)byteSize);
        _dataPointer = (nint)_ownedBuffer.Pointer;
    }

    /// <summary>Creates a tensor that borrows memory from an external pointer (e.g., mmap'd weights).</summary>
    public Tensor(void* dataPointer, TensorShape shape, DType dtype, DeviceKind device = default)
    {
        _dataPointer = (nint)dataPointer;
        Shape = shape;
        DType = dtype;
        Device = device;
        _ownedBuffer = null;
    }

    /// <summary>Returns a span over the tensor data interpreted as <typeparamref name="T"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan<T>() where T : unmanaged
    {
        EnsureCpuData();
        nint ptr = _dataPointer;
        if (ptr == 0)
            throw new ObjectDisposedException(nameof(Tensor));
        int count = (int)(DType.ComputeByteCount(Shape.ElementCount) / sizeof(T));
        return new Span<T>((void*)ptr, count);
    }

    /// <summary>Returns a read-only span over the tensor data.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AsReadOnlySpan<T>() where T : unmanaged
    {
        EnsureCpuData();
        nint ptr = _dataPointer;
        if (ptr == 0)
            throw new ObjectDisposedException(nameof(Tensor));
        int count = (int)(DType.ComputeByteCount(Shape.ElementCount) / sizeof(T));
        return new ReadOnlySpan<T>((void*)ptr, count);
    }

    /// <summary>Creates a zero-alloc TensorRef view of this tensor for use in kernel implementations.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef AsRef()
    {
        EnsureCpuData();
        nint ptr = _dataPointer;
        if (ptr == 0)
            throw new ObjectDisposedException(nameof(Tensor));
        return new TensorRef(ptr, Shape, DType, Device);
    }

    /// <summary>Creates a view with a different shape but same underlying data. No copy.</summary>
    public Tensor Reshape(TensorShape newShape)
    {
        void* ptr = DataPointer;

        if (newShape.ElementCount != Shape.ElementCount)
            throw new SharpInference.Core.Exceptions.SharpInferenceException(
                $"Cannot reshape {Shape} ({Shape.ElementCount} elements) to {newShape} ({newShape.ElementCount} elements).");

        return new Tensor(ptr, newShape, DType, Device);
    }

    /// <summary>Creates a contiguous copy on the specified device. Cross-device requires IBackend.CopyTo.</summary>
    public Tensor To(DeviceKind targetDevice)
    {
        void* ptr = DataPointer;

        if (targetDevice == Device)
        {
            Tensor copy = new Tensor(Shape, DType, Device);
            long byteSize = DType.ComputeByteCount(Shape.ElementCount);
            Buffer.MemoryCopy(ptr, copy.DataPointer, byteSize, byteSize);
            return copy;
        }

        throw new SharpInference.Core.Exceptions.SharpInferenceException(
            $"Direct tensor copy from {Device} to {targetDevice} is not supported. Use IBackend.CopyTo for cross-device transfers.");
    }

    /// <summary>Creates a copy cast to the specified dtype. Quantized types require GgufDequantizer.</summary>
    public Tensor CastTo(DType targetDtype)
    {
        void* ptr = DataPointer;

        if (targetDtype == DType)
            return To(Device);

        if (DType.IsQuantized || targetDtype.IsQuantized)
            throw new SharpInference.Core.Exceptions.SharpInferenceException(
                $"Quantized dtype conversion ({DType} → {targetDtype}) requires a dedicated dequantizer. Use GgufDequantizer instead.");

        Tensor result = new Tensor(Shape, targetDtype, Device);
        try
        {
            long count = Shape.ElementCount;

            if (DType == DType.F32 && targetDtype == DType.F16)
            {
                ReadOnlySpan<float> src = new ReadOnlySpan<float>(ptr, (int)count);
                Span<Half> dst = new Span<Half>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = (Half)src[i];
            }
            else if (DType == DType.F16 && targetDtype == DType.F32)
            {
                ReadOnlySpan<Half> src = new ReadOnlySpan<Half>(ptr, (int)count);
                Span<float> dst = new Span<float>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = (float)src[i];
            }
            else if (DType == DType.F32 && targetDtype == DType.BF16)
            {
                ReadOnlySpan<float> src = new ReadOnlySpan<float>(ptr, (int)count);
                Span<ushort> dst = new Span<ushort>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                {
                    // BF16: truncate lower 16 bits of F32
                    uint bits = BitConverter.SingleToUInt32Bits(src[i]);
                    dst[i] = (ushort)(bits >> 16);
                }
            }
            else if (DType == DType.BF16 && targetDtype == DType.F32)
            {
                ReadOnlySpan<ushort> src = new ReadOnlySpan<ushort>(ptr, (int)count);
                Span<float> dst = new Span<float>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
            }
            else if (DType == DType.F8E4M3 && targetDtype == DType.F32)
            {
                ReadOnlySpan<byte> src = new ReadOnlySpan<byte>(ptr, (int)count);
                Span<float> dst = new Span<float>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = Fp8E4M3ToFloat(src[i]);
            }
            else if (DType == DType.F32 && targetDtype == DType.F8E4M3)
            {
                ReadOnlySpan<float> src = new ReadOnlySpan<float>(ptr, (int)count);
                Span<byte> dst = new Span<byte>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = FloatToFp8E4M3(src[i]);
            }
            else if (DType == DType.F8E4M3 && targetDtype == DType.F16)
            {
                ReadOnlySpan<byte> src = new ReadOnlySpan<byte>(ptr, (int)count);
                Span<Half> dst = new Span<Half>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = (Half)Fp8E4M3ToFloat(src[i]);
            }
            else if (DType == DType.F16 && targetDtype == DType.F8E4M3)
            {
                ReadOnlySpan<Half> src = new ReadOnlySpan<Half>(ptr, (int)count);
                Span<byte> dst = new Span<byte>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = FloatToFp8E4M3((float)src[i]);
            }
            else if (DType == DType.F8E5M2 && targetDtype == DType.F32)
            {
                ReadOnlySpan<byte> src = new ReadOnlySpan<byte>(ptr, (int)count);
                Span<float> dst = new Span<float>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = Fp8E5M2ToFloat(src[i]);
            }
            else if (DType == DType.F32 && targetDtype == DType.F8E5M2)
            {
                ReadOnlySpan<float> src = new ReadOnlySpan<float>(ptr, (int)count);
                Span<byte> dst = new Span<byte>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = FloatToFp8E5M2(src[i]);
            }
            else if (DType == DType.BF16 && targetDtype == DType.F8E4M3)
            {
                ReadOnlySpan<ushort> src = new ReadOnlySpan<ushort>(ptr, (int)count);
                Span<byte> dst = new Span<byte>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = FloatToFp8E4M3(BitConverter.UInt32BitsToSingle((uint)src[i] << 16));
            }
            else if (DType == DType.BF16 && targetDtype == DType.F16)
            {
                // BF16 → F16: round-trip through F32 (no direct shortcut). BF16 has the same
                // exponent range as F32 (8-bit) so any finite F32 value resulting from the
                // BF16 expand will fit fine into F16 only as long as |x| ≤ 65504 — values
                // outside the F16 range are saturated to ±Inf, matching the standard half cast.
                ReadOnlySpan<ushort> src = new ReadOnlySpan<ushort>(ptr, (int)count);
                Span<Half> dst = new Span<Half>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = (Half)BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
            }
            else if (DType == DType.F16 && targetDtype == DType.BF16)
            {
                // F16 → BF16: cast through F32 then truncate the bottom 16 bits.
                ReadOnlySpan<Half> src = new ReadOnlySpan<Half>(ptr, (int)count);
                Span<ushort> dst = new Span<ushort>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                {
                    uint bits = BitConverter.SingleToUInt32Bits((float)src[i]);
                    dst[i] = (ushort)(bits >> 16);
                }
            }
            else if (DType == DType.F8E4M3 && targetDtype == DType.BF16)
            {
                ReadOnlySpan<byte> src = new ReadOnlySpan<byte>(ptr, (int)count);
                Span<ushort> dst = new Span<ushort>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                {
                    float f = Fp8E4M3ToFloat(src[i]);
                    dst[i] = (ushort)(BitConverter.SingleToUInt32Bits(f) >> 16);
                }
            }
            else
            {
                throw new SharpInference.Core.Exceptions.SharpInferenceException(
                    $"Unsupported dtype conversion: {DType} → {targetDtype}.");
            }

            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    /// <summary>Computes the total byte size for a tensor of the given shape and dtype. Delegates to DType.ComputeByteCount.</summary>
    public static long ComputeByteSize(TensorShape shape, DType dtype) => dtype.ComputeByteCount(shape.ElementCount);

    /// <summary>Fused dequant: FP8-E4M3 → F16, multiplying each value by a per-tensor F32 scalar scale. Used for ComfyUI-style fp8_scaled checkpoints (e.g. flux1-krea-dev_fp8_scaled, flux1-dev-kontext_fp8_scaled) where each linear weight is stored as FP8 plus a companion `scale_weight` scalar so that real_weight = fp8_value * scale_weight. Doing this in one pass avoids allocating an F32 intermediate, halving peak memory during checkpoint load.</summary>
    public unsafe Tensor DequantFp8E4M3ScaledToF16(float scale)
    {
        if (DType != DType.F8E4M3)
            throw new SharpInference.Core.Exceptions.SharpInferenceException(
                $"DequantFp8E4M3ScaledToF16 requires FP8 E4M3 input, got {DType}");

        Tensor result = new Tensor(Shape, DType.F16, Device);
        try
        {
            long count = Shape.ElementCount;
            ReadOnlySpan<byte> src = new ReadOnlySpan<byte>(DataPointer, (int)count);
            Span<Half> dst = new Span<Half>(result.DataPointer, (int)count);
            for (int i = 0; i < (int)count; i++)
                dst[i] = (Half)(Fp8E4M3ToFloat(src[i]) * scale);
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    // ── FP8 E4M3FN Conversion ─────────────────────────────────────────────
    // E4M3FN: sign(1) + exponent(4) + mantissa(3), bias=7, no NaN/Inf, max=448, min_subnormal=2^-9

    /// <summary>Converts an FP8 E4M3FN byte to float.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Fp8E4M3ToFloat(byte fp8)
    {
        uint sign = (uint)(fp8 >> 7) & 1;
        uint exp = (uint)(fp8 >> 3) & 0xF;
        uint mant = (uint)(fp8 & 0x7);

        if (exp == 0 && mant == 0)
            return sign != 0 ? -0.0f : 0.0f;

        float value;
        if (exp == 0)
        {
            // Subnormal: value = (-1)^sign * 2^(1-bias) * (mant / 8)
            value = MathF.Pow(2.0f, 1 - 7) * (mant / 8.0f);
        }
        else
        {
            // Normal: value = (-1)^sign * 2^(exp-bias) * (1 + mant/8)
            value = MathF.Pow(2.0f, (int)exp - 7) * (1.0f + mant / 8.0f);
        }

        return sign != 0 ? -value : value;
    }

    /// <summary>Converts a float to FP8 E4M3FN byte with saturation (clamps to max=448).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte FloatToFp8E4M3(float f)
    {
        uint bits = BitConverter.SingleToUInt32Bits(f);
        uint sign = (bits >> 31) & 1;
        int f32Exp = (int)((bits >> 23) & 0xFF) - 127; // unbiased exponent
        uint f32Mant = bits & 0x7FFFFF; // 23-bit mantissa

        // Handle special cases
        if (float.IsNaN(f))
            return (byte)((sign << 7) | 0x7F); // Max magnitude in E4M3FN (no NaN encoding)

        float absF = MathF.Abs(f);
        if (absF == 0.0f)
            return (byte)(sign << 7);

        // Saturation: clamp to max representable value (448)
        if (absF > 448.0f)
            return (byte)((sign << 7) | 0x7E); // 0_1111_110 = max normal = 448

        // Subnormal range: absF < 2^(1-7) = 2^-6 = 0.015625
        if (f32Exp < -6)
        {
            // Round to nearest subnormal: value = 2^-6 * (mant/8)
            float scaled = absF / MathF.Pow(2.0f, -6);
            uint mant = (uint)MathF.Round(scaled * 8.0f);
            if (mant == 0) return (byte)(sign << 7);
            if (mant > 7) mant = 7;
            return (byte)((sign << 7) | mant);
        }

        // Normal range
        int e4Exp = f32Exp + 7; // Re-bias for E4M3
        if (e4Exp < 1) e4Exp = 1;
        if (e4Exp > 15) e4Exp = 15;

        // Round mantissa from 23 bits to 3 bits
        uint mant3 = (f32Mant + (1u << 19)) >> 20; // round to nearest
        if (mant3 > 7)
        {
            mant3 = 0;
            e4Exp++;
            if (e4Exp > 15) return (byte)((sign << 7) | 0x7E); // overflow to max
        }

        return (byte)((sign << 7) | ((uint)e4Exp << 3) | mant3);
    }

    // ── FP8 E5M2 Conversion ─────────────────────────────────────────────
    // E5M2: sign(1) + exponent(5) + mantissa(2), bias=15, has NaN/Inf, max=57344

    /// <summary>Converts an FP8 E5M2 byte to float.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Fp8E5M2ToFloat(byte fp8)
    {
        // E5M2 maps directly to F16 upper byte (same exponent/mantissa layout as FP16 truncated)
        // FP16: sign(1) + exp(5) + mant(10). E5M2 is the upper 8 bits of FP16.
        ushort f16Bits = (ushort)((uint)fp8 << 8);
        return (float)BitConverter.UInt16BitsToHalf(f16Bits);
    }

    /// <summary>Converts a float to FP8 E5M2 byte with saturation.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte FloatToFp8E5M2(float f)
    {
        // Convert to F16 first, then take upper byte (E5M2 = truncated F16)
        Half h = (Half)f;
        ushort f16Bits = BitConverter.HalfToUInt16Bits(h);
        // Round: check bit 7 (MSB of discarded mantissa bits)
        byte upper = (byte)(f16Bits >> 8);
        if ((f16Bits & 0x80) != 0 && (upper & 0x7F) < 0x7F) // round up if halfway or above, not at max
            upper++;
        return upper;
    }

    /// <summary>Frees owned memory via atomic pointer exchange. If GPU data is cached, frees it without D2H copy. Borrowed tensors are no-ops.</summary>
    public void Dispose()
    {
        nint ptr = Interlocked.Exchange(ref _dataPointer, 0);
        Action? gpuDispose = Interlocked.Exchange(ref _gpuDisposeCallback, null);
        Interlocked.Exchange(ref _gpuSyncCallback, null);
        gpuDispose?.Invoke();
        NativeBuffer? buffer = Interlocked.Exchange(ref _ownedBuffer, null);
        if (ptr != 0 && buffer is not null)
        {
            buffer.Dispose();
        }
        GC.SuppressFinalize(this);
    }

    ~Tensor()
    {
        nint ptr = Interlocked.Exchange(ref _dataPointer, 0);
        Action? gpuDispose = Interlocked.Exchange(ref _gpuDisposeCallback, null);
        Interlocked.Exchange(ref _gpuSyncCallback, null);
        gpuDispose?.Invoke();
        NativeBuffer? buffer = Interlocked.Exchange(ref _ownedBuffer, null);
        if (ptr != 0 && buffer is not null)
        {
            buffer.Dispose();
        }
    }
}
