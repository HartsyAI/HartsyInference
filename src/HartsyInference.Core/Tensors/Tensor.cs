using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Memory;

namespace HartsyInference.Core.Tensors;

/// <summary>Core tensor type holding a pointer to unmanaged memory, shape, dtype, and device.</summary>
public sealed unsafe class Tensor : IDisposable
{
    private NativeBuffer? _ownedBuffer;
    private nint _dataPointer;

    /// <summary>Optional owner object kept alive for the lifetime of this tensor. For tensors that borrow an
    /// external pointer (e.g. an mmap'd weight file), this roots the owner (e.g. the <c>MmapHandle</c>) so its
    /// finalizer can't run — and unmap the memory out from under the borrowed pointer — while the tensor is still
    /// reachable. Without this, a loader whose tensors outlive the loader reference itself (a common pattern when a
    /// helper returns only the tensor dictionary) would dangle the moment a GC collects the loader.</summary>
    private object? _keepAlive;

    /// <summary>For owned tensors: the host buffer is allocated lazily on first CPU access (see <see cref="EnsureHostBuffer"/>).
    /// This byte size is kept so the allocation can happen later. Zero for borrowed tensors.</summary>
    private readonly long _byteSize;

    /// <summary>True when this tensor owns its (lazily allocated) host buffer; false when it borrows an external pointer
    /// (mmap'd weights, pooled buffers, reshapes/views). Drives <see cref="OwnsMemory"/> and the lazy-alloc path.</summary>
    private readonly bool _ownsLazy;

    /// <summary>Set once on Dispose so the lazy host-buffer path can't resurrect freed memory.</summary>
    private bool _disposed;

    /// <summary>Backend-set callback: copies GPU→CPU then frees GPU pointer. Invoked lazily on first CPU data access.</summary>
    internal Action? _gpuSyncCallback;

    /// <summary>Backend-set callback: frees GPU pointer without D2H copy. Invoked on Dispose when GPU data was never synced to CPU.</summary>
    internal Action? _gpuDisposeCallback;

    /// <summary>GPU cleanup callbacks from tensors reclaimed by the finalizer (never explicitly Disposed) — queued
    /// here instead of invoked inline. The finalizer thread has no business making CUDA driver calls (stream
    /// enqueue order across threads is a race even though individual driver calls are thread-safe): a free enqueued
    /// from the finalizer thread can land between two dependent ops the inference thread assumed were adjacent in
    /// the stream, corrupting an in-flight buffer. Drained by the backend on the thread that actually owns the
    /// stream (see <c>CudaContext.EnsureCurrent</c>) so every GPU driver call still comes from one thread.</summary>
    internal static readonly System.Collections.Concurrent.ConcurrentQueue<Action> PendingFinalizerGpuCleanup = new();

    /// <summary>Drains and invokes every GPU cleanup callback queued by tensor finalizers. Call only from the thread
    /// that owns the backend's CUDA context/stream.</summary>
    public static void DrainPendingFinalizerGpuCleanup()
    {
        while (PendingFinalizerGpuCleanup.TryDequeue(out Action? cleanup))
        {
            cleanup();
        }
    }

    /// <summary>Creates a new owned tensor. The host buffer is NOT allocated here; it is allocated (zeroed) lazily on the
    /// first CPU access via <see cref="DataPointer"/>/<see cref="AsSpan{T}"/>/etc. GPU-resident activations (whose data
    /// lives on the device and is freed without ever being read on the host) therefore never pay for a host malloc+memset.</summary>
    public Tensor(TensorShape shape, DType dtype, DeviceKind device = default)
    {
        Shape = shape;
        DType = dtype;
        Device = device;
        _byteSize = dtype.ComputeByteCount(shape.ElementCount);
        _ownsLazy = true;
    }

    /// <summary>Creates a tensor that borrows memory from an external pointer (e.g., mmap'd weights).</summary>
    public Tensor(void* dataPointer, TensorShape shape, DType dtype, DeviceKind device = default)
    {
        _dataPointer = (nint)dataPointer;
        Shape = shape;
        DType = dtype;
        Device = device;
        _byteSize = dtype.ComputeByteCount(shape.ElementCount);
        _ownedBuffer = null;
        _ownsLazy = false;
    }

    /// <summary>Roots <paramref name="owner"/> for this tensor's lifetime (see <see cref="_keepAlive"/>). Used by
    /// borrowing loaders to pin the backing store (e.g. an <c>MmapHandle</c>) against premature GC/finalization.</summary>
    internal void SetKeepAlive(object? owner) => _keepAlive = owner;

    /// <summary>The owner object (if any) kept alive for this tensor's lifetime. See <see cref="SetKeepAlive"/>.</summary>
    internal object? KeepAliveOwner => _keepAlive;

    /// <summary>Shape and strides of this tensor.</summary>
    public TensorShape Shape { get; }

    /// <summary>Data type of the tensor elements.</summary>
    public DType DType { get; }

    /// <summary>Device this tensor resides on.</summary>
    public DeviceKind Device { get; }

    /// <summary>Total number of elements across all dimensions.</summary>
    public long ElementCount => Shape.ElementCount;

    /// <summary>Whether this tensor owns its memory or borrows it from a mmap/view. True even before the lazy host
    /// buffer has been allocated, since ownership is a property of how the tensor was constructed, not of allocation timing.</summary>
    public bool OwnsMemory => _ownsLazy;

    /// <summary>Ensures the owned host buffer exists (allocating it zeroed on first call) and returns its pointer. For
    /// borrowed tensors the external pointer is returned as-is. This is the single lazy-allocation chokepoint: GPU
    /// activations that are never read on the host skip it entirely and so never allocate host memory. The CUDA D2H
    /// sync callback also calls this to obtain a destination buffer at sync time.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void* EnsureHostBuffer()
    {
        nint ptr = _dataPointer;
        if (ptr != 0)
            return (void*)ptr;
        if (_disposed || !_ownsLazy)
            throw new ObjectDisposedException(nameof(Tensor));
        NativeBuffer buffer = new NativeBuffer((nuint)_byteSize);
        _ownedBuffer = buffer;
        ptr = (nint)buffer.Pointer;
        _dataPointer = ptr;
        return (void*)ptr;
    }

    /// <summary>Per-tensor scale for ComfyUI-style <c>fp8_scaled</c> quantization, where the real value of each weight is <c>fp8_byte_decoded * Fp8ScaleFactor</c>. Default 1.0 means no extra scaling. Non-1 values are folded into cuBLAS' <c>alpha</c> at GEMM call sites so scaling happens for free during matmul.</summary>
    public float Fp8ScaleFactor { get; set; } = 1.0f;

    /// <summary>Pointer to the raw tensor data. If GPU data is cached, triggers a lazy sync (D2H copy) first; otherwise
    /// the owned host buffer is allocated (zeroed) on first access.</summary>
    public void* DataPointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            EnsureCpuData();
            return EnsureHostBuffer();
        }
    }

    /// <summary>Returns a span over the tensor data interpreted as <typeparamref name="T"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan<T>() where T : unmanaged
    {
        EnsureCpuData();
        void* ptr = EnsureHostBuffer();
        int count = (int)(DType.ComputeByteCount(Shape.ElementCount) / sizeof(T));
        return new Span<T>(ptr, count);
    }

    /// <summary>Returns a read-only span over the tensor data.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AsReadOnlySpan<T>() where T : unmanaged
    {
        EnsureCpuData();
        void* ptr = EnsureHostBuffer();
        int count = (int)(DType.ComputeByteCount(Shape.ElementCount) / sizeof(T));
        return new ReadOnlySpan<T>(ptr, count);
    }

    /// <summary>Creates a zero-alloc TensorRef view of this tensor for use in kernel implementations.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef AsRef()
    {
        EnsureCpuData();
        nint ptr = (nint)EnsureHostBuffer();
        return new TensorRef(ptr, Shape, DType, Device);
    }

    /// <summary>Creates a view with a different shape but same underlying data. No copy. The view roots this
    /// tensor (see <see cref="_keepAlive"/>): without that, reshaping a temporary (e.g. an <c>EnsureF32</c> cast
    /// held by nothing else) left the view dangling once GC finalized the parent and freed its buffer — a
    /// GC-timing-dependent AccessViolation observed in the Qwen3-TTS vocoder after ~9 min of generation.</summary>
    public Tensor Reshape(TensorShape newShape)
    {
        void* ptr = DataPointer;

        if (newShape.ElementCount != Shape.ElementCount)
            throw new HartsyInferenceException(
                $"Cannot reshape {Shape} ({Shape.ElementCount} elements) to {newShape} ({newShape.ElementCount} elements).");

        Tensor view = new(ptr, newShape, DType, Device);
        view.SetKeepAlive(this);
        return view;
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
            // A byte-identical copy of an fp8_scaled tensor is still scaled — dropping the factor here would
            // silently rescale the weight by 1/scale at the next GEMM.
            copy.Fp8ScaleFactor = Fp8ScaleFactor;
            return copy;
        }

        throw new HartsyInferenceException(
            $"Direct tensor copy from {Device} to {targetDevice} is not supported. Use IBackend.CopyTo for cross-device transfers.");
    }

    /// <summary>Creates a copy cast to the specified dtype. Quantized types require GgufDequantizer.</summary>
    public Tensor CastTo(DType targetDtype)
    {
        void* ptr = DataPointer;

        if (targetDtype == DType)
            return To(Device);

        if (DType.IsQuantized || targetDtype.IsQuantized)
            throw new HartsyInferenceException(
                $"Quantized dtype conversion ({DType} → {targetDtype}) requires a dedicated dequantizer. Use GgufDequantizer instead.");

        Tensor result = new Tensor(Shape, targetDtype, Device);
        try
        {
            long count = Shape.ElementCount;
            // Dequantizing an fp8_scaled tensor must fold the per-tensor scale into the values: the fp8 bytes
            // alone are `real_value / scale`. The result carries factor 1.0 (already applied), so downstream
            // GEMM alpha logic sees an ordinary unscaled tensor. Scale is 1.0 for everything non-fp8_scaled,
            // making the multiply a no-op on plain checkpoints.
            float fp8Scale = Fp8ScaleFactor;

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
                    dst[i] = Fp8E4M3ToFloat(src[i]) * fp8Scale;
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
                    dst[i] = (Half)(Fp8E4M3ToFloat(src[i]) * fp8Scale);
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
                    dst[i] = Fp8E5M2ToFloat(src[i]) * fp8Scale;
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
                // BF16 has F32-range exponent; values |x| > 65504 saturate to ±Inf in F16,
                // matching the standard half cast.
                ReadOnlySpan<ushort> src = new ReadOnlySpan<ushort>(ptr, (int)count);
                Span<Half> dst = new Span<Half>(result.DataPointer, (int)count);
                for (int i = 0; i < (int)count; i++)
                    dst[i] = (Half)BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
            }
            else if (DType == DType.F16 && targetDtype == DType.BF16)
            {
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
                    float f = Fp8E4M3ToFloat(src[i]) * fp8Scale;
                    dst[i] = (ushort)(BitConverter.SingleToUInt32Bits(f) >> 16);
                }
            }
            else
            {
                throw new HartsyInferenceException(
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

    /// <summary>Fused dequant: FP8-E4M3 → F16, multiplying each value by a per-tensor F32 scalar. Used for ComfyUI-style fp8_scaled checkpoints (e.g. flux1-krea-dev_fp8_scaled) where real_weight = fp8_value * scale_weight. Single-pass to avoid an F32 intermediate, halving peak memory during checkpoint load.</summary>
    public unsafe Tensor DequantFp8E4M3ScaledToF16(float scale)
    {
        if (DType != DType.F8E4M3)
            throw new HartsyInferenceException(
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

    /// <summary>Computes the total byte size for a tensor of the given shape and dtype.</summary>
    public static long ComputeByteSize(TensorShape shape, DType dtype) => dtype.ComputeByteCount(shape.ElementCount);

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

    // E4M3FN: sign(1) + exponent(4) + mantissa(3), bias=7, no NaN/Inf, max=448, min_subnormal=2^-9
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte FloatToFp8E4M3(float f)
    {
        uint bits = BitConverter.SingleToUInt32Bits(f);
        uint sign = (bits >> 31) & 1;
        int f32Exp = (int)((bits >> 23) & 0xFF) - 127;
        uint f32Mant = bits & 0x7FFFFF;

        if (float.IsNaN(f))
            return (byte)((sign << 7) | 0x7F); // Max magnitude in E4M3FN (no NaN encoding)

        float absF = MathF.Abs(f);
        if (absF == 0.0f)
            return (byte)(sign << 7);

        // Saturate to max representable value (448)
        if (absF > 448.0f)
            return (byte)((sign << 7) | 0x7E);

        // Subnormal range: absF < 2^(1-7) = 2^-6 = 0.015625
        if (f32Exp < -6)
        {
            float scaled = absF / MathF.Pow(2.0f, -6);
            uint mant = (uint)MathF.Round(scaled * 8.0f);
            if (mant == 0) return (byte)(sign << 7);
            if (mant > 7) mant = 7;
            return (byte)((sign << 7) | mant);
        }

        int e4Exp = f32Exp + 7;
        if (e4Exp < 1) e4Exp = 1;
        if (e4Exp > 15) e4Exp = 15;

        uint mant3 = (f32Mant + (1u << 19)) >> 20; // round to nearest
        if (mant3 > 7)
        {
            mant3 = 0;
            e4Exp++;
            if (e4Exp > 15) return (byte)((sign << 7) | 0x7E);
        }

        return (byte)((sign << 7) | ((uint)e4Exp << 3) | mant3);
    }

    // E5M2: sign(1) + exponent(5) + mantissa(2), bias=15, has NaN/Inf, max=57344.
    // Maps directly onto the upper byte of FP16 (same exponent/mantissa layout).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Fp8E5M2ToFloat(byte fp8)
    {
        ushort f16Bits = (ushort)((uint)fp8 << 8);
        return (float)BitConverter.UInt16BitsToHalf(f16Bits);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte FloatToFp8E5M2(float f)
    {
        Half h = (Half)f;
        ushort f16Bits = BitConverter.HalfToUInt16Bits(h);
        byte upper = (byte)(f16Bits >> 8);
        // Round up if discarded bits' MSB is set and we're not already at max
        if ((f16Bits & 0x80) != 0 && (upper & 0x7F) < 0x7F)
            upper++;
        return upper;
    }

    /// <summary>Frees owned memory via atomic pointer exchange. If GPU data is cached, frees it without D2H copy. Borrowed tensors are no-ops.</summary>
    public void Dispose()
    {
        _disposed = true;
        nint ptr = Interlocked.Exchange(ref _dataPointer, 0);
        Action? gpuDispose = Interlocked.Exchange(ref _gpuDisposeCallback, null);
        Interlocked.Exchange(ref _gpuSyncCallback, null);
        gpuDispose?.Invoke();
        NativeBuffer? buffer = Interlocked.Exchange(ref _ownedBuffer, null);
        if (ptr != 0 && buffer is not null)
        {
            buffer.Dispose();
        }
        _keepAlive = null;
        GC.SuppressFinalize(this);
    }

    ~Tensor()
    {
        _disposed = true;
        nint ptr = Interlocked.Exchange(ref _dataPointer, 0);
        Action? gpuDispose = Interlocked.Exchange(ref _gpuDisposeCallback, null);
        Interlocked.Exchange(ref _gpuSyncCallback, null);
        // Never invoke a CUDA-touching callback from the finalizer thread directly — queue it for the
        // backend's owning thread to drain (see PendingFinalizerGpuCleanup).
        if (gpuDispose is not null)
        {
            PendingFinalizerGpuCleanup.Enqueue(gpuDispose);
        }
        NativeBuffer? buffer = Interlocked.Exchange(ref _ownedBuffer, null);
        if (ptr != 0 && buffer is not null)
        {
            buffer.Dispose();
        }
    }
}
