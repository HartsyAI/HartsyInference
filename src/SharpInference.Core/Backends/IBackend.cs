using SharpInference.Core.Tensors;

namespace SharpInference.Core.Backends;

/// <summary>Backend interface that all model code programs against. Implementations provide the actual compute (CPU via SIMD, CUDA via PTX/cuBLAS). All operations are eager — they execute immediately and return when complete.</summary>
public interface IBackend : IDisposable
{
    /// <summary>The device this backend targets.</summary>
    DeviceKind Device { get; }

    /// <summary>Capabilities of this backend.</summary>
    BackendCapabilities Capabilities { get; }

    // ── Linear Algebra ──────────────────────────────────────────────────

    /// <summary>Matrix multiply: output = a @ b</summary>
    void MatMul(Tensor output, Tensor a, Tensor b);

    /// <summary>Batched matrix multiply: output[i] = a[i] @ b[i]</summary>
    void BatchedMatMul(Tensor output, Tensor a, Tensor b);

    /// <summary>Linear layer: output = input × weight^T + bias. Input [M, K], weight [N, K], bias [N] (optional), output [M, N]. Also works with leading batch dims: input [B, S, K] → output [B, S, N].</summary>
    void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias);

    // ── Convolution ─────────────────────────────────────────────────────

    /// <summary>2D convolution: output = conv2d(input, weight, bias, stride, padding)</summary>
    void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW);

    // ── Normalization ───────────────────────────────────────────────────

    /// <summary>Group normalization.</summary>
    void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps);

    /// <summary>Layer normalization.</summary>
    void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps);

    /// <summary>RMS normalization.</summary>
    void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps);

    // ── Attention ───────────────────────────────────────────────────────

    /// <summary>Scaled dot-product attention: output = softmax(Q @ K^T / sqrt(d)) @ V</summary>
    void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale);

    // ── Activations ─────────────────────────────────────────────────────

    /// <summary>GELU activation (exact).</summary>
    void Gelu(Tensor output, Tensor input);

    /// <summary>SiLU activation (x * sigmoid(x)).</summary>
    void Silu(Tensor output, Tensor input);

    // ── Element-wise ────────────────────────────────────────────────────

    /// <summary>Element-wise addition: output = a + b</summary>
    void Add(Tensor output, Tensor a, Tensor b);

    /// <summary>Element-wise multiplication: output = a * b</summary>
    void Mul(Tensor output, Tensor a, Tensor b);

    /// <summary>Scalar multiplication: output = input * scalar</summary>
    void Scale(Tensor output, Tensor input, float scalar);

    /// <summary>Element-wise clamp: output = clamp(input, min, max)</summary>
    void Clamp(Tensor output, Tensor input, float min, float max);

    // ── Transpose / Permute ─────────────────────────────────────────────

    /// <summary>Batched 2D transpose: [B, D1, D2] → [B, D2, D1].</summary>
    void Transpose2D(Tensor output, Tensor input, int d1, int d2);

    /// <summary>4D permute swapping dims 1 and 2: [B, S, H, D] → [B, H, S, D].</summary>
    void Permute0213(Tensor output, Tensor input, int s, int h, int d);

    /// <summary>GEGLU activation: splits input in half along last dim, applies GELU gate. Output has half the elements of input.</summary>
    void GeGlu(Tensor output, Tensor input);

    /// <summary>Broadcast add: hidden [B, C, ...spatial] += bias [B, C] in-place.</summary>
    void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial);

    // ── Shape Operations ────────────────────────────────────────────────

    /// <summary>Concatenate tensors along the specified dimension.</summary>
    void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim);

    /// <summary>Split a tensor into chunks along the specified dimension.</summary>
    void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim);

    // ── Sampling ────────────────────────────────────────────────────────

    /// <summary>Nearest-neighbor 2D upsample by the given scale factor.</summary>
    void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW);

    /// <summary>Bilinear 2D upsample by the given scale factor.</summary>
    void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW);

    // ── Data Movement ───────────────────────────────────────────────────

    /// <summary>Copy tensor data to a different device.</summary>
    void CopyTo(Tensor destination, Tensor source);

    /// <summary>Fill a tensor with a constant value.</summary>
    void Fill(Tensor tensor, float value);

    // ── Audio (optional — backends may throw NotSupportedException) ──

    /// <summary>Radix-2 FFT for audio processing.</summary>
    void Fft(Tensor output, Tensor input);

    /// <summary>Short-time Fourier transform.</summary>
    void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window);

    /// <summary>Apply mel filterbank to FFT magnitude spectrogram.</summary>
    void MelFilterbank(Tensor output, Tensor input, Tensor filters);

    // ── Fused Operations ────────────────────────────────────────────────

    /// <summary>Fused GroupNorm + SiLU: normalize, apply affine, then SiLU in one pass. Eliminates intermediate allocation. Default falls back to separate GroupNorm + Silu.</summary>
    void GroupNormSilu(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        GroupNorm(output, input, weight, bias, groups, eps);
        Silu(output, output);
    }

    // ── Dtype Casting ────────────────────────────────────────────────────

    /// <summary>Cast tensor from FP32 to FP16. Default: CPU scalar loop.</summary>
    unsafe void CastToF16(Tensor output, Tensor input)
    {
        float* src = (float*)input.DataPointer;
        Half* dst = (Half*)output.DataPointer;
        int count = (int)input.Shape.ElementCount;
        for (int i = 0; i < count; i++)
            dst[i] = (Half)src[i];
    }

    /// <summary>Cast tensor from FP16 to FP32. Default: CPU scalar loop.</summary>
    unsafe void CastToF32(Tensor output, Tensor input)
    {
        Half* src = (Half*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        int count = (int)input.Shape.ElementCount;
        for (int i = 0; i < count; i++)
            dst[i] = (float)src[i];
    }

    // ── Synchronization ──────────────────────────────────────────────────

    /// <summary>Waits for all pending GPU work to complete. No-op on CPU backends. Call at pipeline phase boundaries to ensure deferred memory frees are processed before large allocations.</summary>
    void Sync() { }

    /// <summary>Frees specific weight tensors from accelerator memory. No-op on CPU backends. Call between pipeline phases to reclaim VRAM (e.g., free UNet weights before VAE decode).</summary>
    void FreeWeights(IEnumerable<Tensor> weights) { }
}
