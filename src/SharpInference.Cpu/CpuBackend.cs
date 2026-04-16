using System.Runtime.InteropServices;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Cpu.Kernels;

namespace SharpInference.Cpu;

/// <summary>CPU-based inference backend that routes all compute operations to SIMD-optimized kernel classes. Supports AVX2, AVX-512, and ARM NEON via <see cref="SimdDispatch"/>.</summary>
public sealed class CpuBackend : IBackend
{
    private int _disposed;

    /// <summary>Gets the device kind for this backend, which is always <see cref="DeviceKind.Cpu"/>.</summary>
    public DeviceKind Device { get; } = DeviceKind.Cpu;

    /// <summary>Gets the capabilities of this CPU backend.</summary>
    public BackendCapabilities Capabilities { get; } = new BackendCapabilities
    {
        SupportsF32 = true,
        SupportsF16 = true,
        SupportsConv2D = true,
        SupportsSdpa = true,
        SupportsFft = true,
        Name = "CPU",
    };

    /// <inheritdoc />
    public void MatMul(Tensor output, Tensor a, Tensor b)
    {
        ThrowIfDisposed();
        MatMulKernels.MatMul(output, a, b);
    }

    /// <inheritdoc />
    public void BatchedMatMul(Tensor output, Tensor a, Tensor b)
    {
        ThrowIfDisposed();
        MatMulKernels.BatchedMatMul(output, a, b);
    }

    /// <inheritdoc />
    public void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW)
    {
        ThrowIfDisposed();
        Conv2DKernels.Conv2D(output, input, weight, bias, strideH, strideW, padH, padW);
    }

    /// <inheritdoc />
    public void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        ThrowIfDisposed();
        NormKernels.GroupNorm(output, input, weight, bias, groups, eps);
    }

    /// <inheritdoc />
    public void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps)
    {
        ThrowIfDisposed();
        NormKernels.LayerNorm(output, input, weight, bias, eps);
    }

    /// <inheritdoc />
    public void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps)
    {
        ThrowIfDisposed();
        NormKernels.RmsNorm(output, input, weight, eps);
    }

    /// <inheritdoc />
    public void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale)
    {
        ThrowIfDisposed();
        AttentionKernels.ScaledDotProductAttention(output, query, key, value, mask, scale);
    }

    /// <inheritdoc />
    public void Gelu(Tensor output, Tensor input)
    {
        ThrowIfDisposed();
        ActivationKernels.Gelu(output, input);
    }

    /// <inheritdoc />
    public void Silu(Tensor output, Tensor input)
    {
        ThrowIfDisposed();
        ActivationKernels.Silu(output, input);
    }

    /// <inheritdoc />
    public void Add(Tensor output, Tensor a, Tensor b)
    {
        ThrowIfDisposed();
        ElementWiseKernels.Add(output, a, b);
    }

    /// <inheritdoc />
    public void Mul(Tensor output, Tensor a, Tensor b)
    {
        ThrowIfDisposed();
        ElementWiseKernels.Mul(output, a, b);
    }

    /// <inheritdoc />
    public void Scale(Tensor output, Tensor input, float scalar)
    {
        ThrowIfDisposed();
        ElementWiseKernels.Scale(output, input, scalar);
    }

    /// <inheritdoc />
    public void Clamp(Tensor output, Tensor input, float min, float max)
    {
        ThrowIfDisposed();
        ElementWiseKernels.Clamp(output, input, min, max);
    }

    /// <inheritdoc />
    public void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim)
    {
        ThrowIfDisposed();
        ElementWiseKernels.Concat(output, inputs, dim);
    }

    /// <inheritdoc />
    public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim)
    {
        ThrowIfDisposed();
        ElementWiseKernels.Split(outputs, input, dim);
    }

    /// <inheritdoc />
    public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
        ThrowIfDisposed();
        UpDownSampleKernels.UpsampleNearest2D(output, input, scaleH, scaleW);
    }

    /// <inheritdoc />
    public void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
        ThrowIfDisposed();
        UpDownSampleKernels.UpsampleBilinear2D(output, input, scaleH, scaleW);
    }

    /// <inheritdoc />
    public unsafe void CopyTo(Tensor destination, Tensor source)
    {
        ThrowIfDisposed();
        long byteSize = Tensor.ComputeByteSize(source.Shape, source.DType);
        NativeMemory.Copy(source.DataPointer, destination.DataPointer, (nuint)byteSize);
    }

    /// <inheritdoc />
    public void Fill(Tensor tensor, float value)
    {
        ThrowIfDisposed();
        Span<float> span = tensor.AsSpan<float>();
        span.Fill(value);
    }

    /// <inheritdoc />
    public void Fft(Tensor output, Tensor input)
    {
        ThrowIfDisposed();
        AudioKernels.Fft(output, input);
    }

    /// <inheritdoc />
    public void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window)
    {
        ThrowIfDisposed();
        AudioKernels.Stft(output, input, fftSize, hopLength, window);
    }

    /// <inheritdoc />
    public void MelFilterbank(Tensor output, Tensor input, Tensor filters)
    {
        ThrowIfDisposed();
        AudioKernels.MelFilterbank(output, input, filters);
    }

    /// <summary>Releases all resources used by this backend.</summary>
    public void Dispose()
    {
        Volatile.Write(ref _disposed, 1);
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }
}
