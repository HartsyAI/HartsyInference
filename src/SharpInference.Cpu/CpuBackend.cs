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
    public unsafe void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        ThrowIfDisposed();
        MatMulKernels.LinearTransB(output, input, weight, bias);
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
    public void Sigmoid(Tensor output, Tensor input)
    {
        ThrowIfDisposed();
        ActivationKernels.Sigmoid(output, input);
    }

    /// <inheritdoc />
    public void Tanh(Tensor output, Tensor input)
    {
        ThrowIfDisposed();
        ActivationKernels.Tanh(output, input);
    }

    /// <inheritdoc />
    public void Elu(Tensor output, Tensor input, float alpha)
    {
        ThrowIfDisposed();
        ActivationKernels.Elu(output, input, alpha);
    }

    /// <inheritdoc />
    public void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta)
    {
        ThrowIfDisposed();
        ActivationKernels.Snake(output, input, alpha, beta);
    }

    /// <inheritdoc />
    public void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups)
    {
        ThrowIfDisposed();
        Conv1dKernels.Conv1d(output, input, weight, bias, stride, padLeft, padRight, dilation, groups);
    }

    /// <inheritdoc />
    public void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation)
    {
        ThrowIfDisposed();
        Conv1dKernels.ConvTranspose1d(output, input, weight, bias, stride, padLeft, padRight, dilation);
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
    public unsafe void Transpose2D(Tensor output, Tensor input, int d1, int d2)
    {
        ThrowIfDisposed();
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int batch = (int)(input.ElementCount / (d1 * d2));

        for (int b = 0; b < batch; b++)
        {
            float* batchIn = inPtr + b * d1 * d2;
            float* batchOut = outPtr + b * d2 * d1;
            for (int i = 0; i < d1; i++)
            {
                for (int j = 0; j < d2; j++)
                {
                    batchOut[j * d1 + i] = batchIn[i * d2 + j];
                }
            }
        }
    }

    /// <inheritdoc />
    public unsafe void Permute0213(Tensor output, Tensor input, int s, int h, int d)
    {
        ThrowIfDisposed();
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int batch = (int)(input.ElementCount / (s * h * d));

        for (int b = 0; b < batch; b++)
        {
            for (int si = 0; si < s; si++)
            {
                for (int hi = 0; hi < h; hi++)
                {
                    int inOffset = ((b * s + si) * h + hi) * d;
                    int outOffset = ((b * h + hi) * s + si) * d;
                    for (int di = 0; di < d; di++)
                    {
                        outPtr[outOffset + di] = inPtr[inOffset + di];
                    }
                }
            }
        }
    }

    /// <inheritdoc />
    public unsafe void GeGlu(Tensor output, Tensor input)
    {
        ThrowIfDisposed();
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int outputElements = (int)output.ElementCount;
        int innerDim = (int)output.Shape[output.Shape.Rank - 1];

        for (int i = 0; i < outputElements; i++)
        {
            int outerIdx = i / innerDim;
            int d = i % innerDim;
            int inputIdx = outerIdx * (innerDim * 2) + d;
            float x = inPtr[inputIdx];
            float gate = inPtr[inputIdx + innerDim];
            float geluGate = gate * 0.5f * (1.0f + MathF.Tanh(0.7978845608f * (gate + 0.044715f * gate * gate * gate)));
            outPtr[i] = x * geluGate;
        }
    }

    /// <inheritdoc />
    public unsafe void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial)
    {
        ThrowIfDisposed();
        float* hPtr = (float*)hidden.DataPointer;
        float* bPtr = (float*)bias.DataPointer;
        int total = (int)hidden.ElementCount;
        int batch = total / (channels * spatial);

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float bVal = bPtr[b * channels + c];
                int offset = (b * channels + c) * spatial;
                for (int s = 0; s < spatial; s++)
                {
                    hPtr[offset + s] += bVal;
                }
            }
        }
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
