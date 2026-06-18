using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

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

    /// <summary>Adaptive Instance Normalization 1D. Input <c>[B, C, T]</c> is normalized
    /// per-(batch, channel) across the <c>T</c> axis, then affinely scaled by
    /// <c>(1 + gamma[c])</c> and shifted by <c>beta[c]</c>. <paramref name="gamma"/> and
    /// <paramref name="beta"/> are <c>[B, C]</c> (or <c>[C]</c>, broadcast across batch);
    /// they typically come from a Linear projection of a style / speaker embedding —
    /// AdaIN1d is the core style-conditioning primitive in Kokoro and StyleTTS 2.</summary>
    void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps);

    // ── Attention ───────────────────────────────────────────────────────

    /// <summary>Scaled dot-product attention: output = softmax(Q @ K^T / sqrt(d)) @ V</summary>
    void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale);

    // ── Activations ─────────────────────────────────────────────────────

    /// <summary>GELU activation (exact).</summary>
    void Gelu(Tensor output, Tensor input);

    /// <summary>SiLU activation (x * sigmoid(x)).</summary>
    void Silu(Tensor output, Tensor input);

    /// <summary>Sigmoid activation (1 / (1 + exp(-x))). Used by LSTM gating.</summary>
    void Sigmoid(Tensor output, Tensor input);

    /// <summary>Hyperbolic tangent activation. Used by LSTM cell update / output and
    /// several vocoders (Mish, snake-bias).</summary>
    void Tanh(Tensor output, Tensor input);

    /// <summary>ELU activation: <c>x if x &gt;= 0 else alpha * (exp(x) - 1)</c>. Used by
    /// the SEANet residual blocks in EnCodec and DAC — alpha is typically 1.0.</summary>
    void Elu(Tensor output, Tensor input, float alpha);

    /// <summary>Leaky ReLU: <c>x if x &gt;= 0 else slope * x</c>. Kokoro / StyleTTS 2's
    /// text encoder + decoder use slope=0.2; HiFi-GAN MRF blocks use the same.</summary>
    void LeakyRelu(Tensor output, Tensor input, float slope);

    /// <summary>Snake activation: <c>x + (sin(alpha * x))^2 / divisor</c>, where
    /// <c>divisor = alpha</c> when <paramref name="beta"/> is null (vanilla snake from
    /// the Stable Audio Oobleck VAE), and <c>divisor = beta + 1e-8</c> when supplied
    /// (snake-beta variant from BigVGAN). <paramref name="alpha"/> and
    /// <paramref name="beta"/> are per-channel learnable params of shape <c>[1, C, 1]</c>
    /// (broadcast across batch and time for <c>[B, C, T]</c> input).</summary>
    void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta);

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

    // ── Convolution ─────────────────────────────────────────────────────

    /// <summary>1D convolution with explicit asymmetric padding. Inputs and outputs are
    /// channels-first: <paramref name="input"/> <c>[B, C_in, T_in]</c>,
    /// <paramref name="output"/> <c>[B, C_out, T_out]</c> (pre-allocated by the caller).
    /// <paramref name="weight"/> follows PyTorch convention <c>[C_out, C_in / groups, K]</c>.
    /// Pass <paramref name="padLeft"/>/<paramref name="padRight"/> separately so the same
    /// op covers both causal (left-only) and symmetric padding; pass
    /// <paramref name="groups"/> equal to channels for depthwise mode.</summary>
    void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups);

    /// <summary>1D transposed convolution. Input/output channels-first;
    /// <paramref name="weight"/> follows PyTorch convention <c>[C_in, C_out / groups, K]</c>.
    /// Output length is <c>(T_in - 1) * stride + dilation * (K - 1) + 1 - padLeft - padRight</c>.
    /// For VibeVoice / EnCodec causal decoders pass <c>padLeft = 0</c>,
    /// <c>padRight = K - stride</c> to remove all trailing pad (matches
    /// <c>trim_right_ratio = 1.0</c>). Pass <paramref name="groups"/> equal to channels for
    /// depthwise mode (e.g. BigVGAN anti-aliased upsampling with a shared lowpass filter).</summary>
    void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups);

    // ── Sampling ────────────────────────────────────────────────────────

    /// <summary>Nearest-neighbor 2D upsample by the given scale factor.</summary>
    void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW);

    /// <summary>Bilinear 2D upsample by the given scale factor.</summary>
    void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW);

    /// <summary>2D transposed convolution. Used by YOLO seg's Proto module for upsampling mask
    /// prototypes (k=2, s=2, p=0 doubles spatial dims). Weight shape is <c>[C_in, C_out, kH, kW]</c>
    /// — PyTorch convention, note input channels come first (opposite of standard Conv2d).
    /// Default implementation is a CPU scatter-add loop over F32 NCHW tensors; backends should
    /// override for performance.</summary>
    unsafe void ConvTranspose2d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideH, int strideW, int padH, int padW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32)
            throw new NotSupportedException($"ConvTranspose2d default fallback only supports F32 — got input={input.DType}, output={output.DType}, weight={weight.DType}.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4 || weight.Shape.Rank != 4)
            throw new ArgumentException($"ConvTranspose2d requires 4D tensors; got input {input.Shape}, output {output.Shape}, weight {weight.Shape}.");

        int n = (int)input.Shape[0];
        int cIn = (int)input.Shape[1];
        int iH = (int)input.Shape[2];
        int iW = (int)input.Shape[3];
        int cOut = (int)output.Shape[1];
        int oH = (int)output.Shape[2];
        int oW = (int)output.Shape[3];
        int kH = (int)weight.Shape[2];
        int kW = (int)weight.Shape[3];
        if (weight.Shape[0] != cIn || weight.Shape[1] != cOut)
            throw new ArgumentException($"ConvTranspose2d weight shape [{weight.Shape[0]}, {weight.Shape[1]}, ...] must equal [C_in={cIn}, C_out={cOut}, ...].");

        float* srcBase = (float*)input.DataPointer;
        float* dstBase = (float*)output.DataPointer;
        float* wBase = (float*)weight.DataPointer;
        float* bBase = bias is null ? null : (float*)bias.DataPointer;

        // Initialize output to bias (so the scatter-add can accumulate on top).
        for (int b = 0; b < n; b++)
        {
            for (int co = 0; co < cOut; co++)
            {
                float biasVal = bBase is null ? 0f : bBase[co];
                float* plane = dstBase + ((long)b * cOut + co) * oH * oW;
                for (long i = 0; i < (long)oH * oW; i++) plane[i] = biasVal;
            }
        }

        // Scatter-add: each input pixel (ci, yi, xi) contributes weight[ci, co, ky, kx] * value
        // to every output position (yi*sH+ky-pH, xi*sW+kx-pW) for each (co, ky, kx).
        for (int b = 0; b < n; b++)
        {
            for (int ci = 0; ci < cIn; ci++)
            {
                float* srcPlane = srcBase + ((long)b * cIn + ci) * iH * iW;
                for (int co = 0; co < cOut; co++)
                {
                    float* dstPlane = dstBase + ((long)b * cOut + co) * oH * oW;
                    long wOff = ((long)ci * cOut + co) * kH * kW;
                    for (int yi = 0; yi < iH; yi++)
                    {
                        for (int xi = 0; xi < iW; xi++)
                        {
                            float v = srcPlane[yi * iW + xi];
                            if (v == 0f) continue;
                            for (int ky = 0; ky < kH; ky++)
                            {
                                int yo = yi * strideH + ky - padH;
                                if (yo < 0 || yo >= oH) continue;
                                for (int kx = 0; kx < kW; kx++)
                                {
                                    int xo = xi * strideW + kx - padW;
                                    if (xo < 0 || xo >= oW) continue;
                                    dstPlane[yo * oW + xo] += wBase[wOff + ky * kW + kx] * v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>Depthwise 2D convolution — each output channel sees exactly one input channel
    /// (groups = C). Used by YOLO11's class branch and the C2PSA positional encoding. Weight
    /// shape is <c>[C, 1, kH, kW]</c> and bias <c>[C]</c>. Default implementation is a CPU loop
    /// over F32 NCHW tensors; backends should override for performance once a depthwise
    /// kernel is worth shipping.</summary>
    unsafe void Conv2dDepthwise(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideH, int strideW, int padH, int padW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32)
            throw new NotSupportedException($"Conv2dDepthwise default fallback only supports F32 — got input={input.DType}, output={output.DType}, weight={weight.DType}.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new ArgumentException($"Conv2dDepthwise requires [N, C, H, W] tensors; got input {input.Shape} / output {output.Shape}.");
        if (weight.Shape.Rank != 4 || weight.Shape[1] != 1)
            throw new ArgumentException($"Conv2dDepthwise weight must be [C, 1, kH, kW]; got {weight.Shape}.");
        if (input.Shape[1] != weight.Shape[0] || output.Shape[1] != weight.Shape[0])
            throw new ArgumentException("Conv2dDepthwise requires input/output channel count to equal weight channel count.");

        int n = (int)input.Shape[0];
        int c = (int)input.Shape[1];
        int iH = (int)input.Shape[2];
        int iW = (int)input.Shape[3];
        int oH = (int)output.Shape[2];
        int oW = (int)output.Shape[3];
        int kH = (int)weight.Shape[2];
        int kW = (int)weight.Shape[3];

        float* srcBase = (float*)input.DataPointer;
        float* dstBase = (float*)output.DataPointer;
        float* wBase = (float*)weight.DataPointer;
        float* bBase = bias is null ? null : (float*)bias.DataPointer;

        for (int b = 0; b < n; b++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                float* srcPlane = srcBase + ((long)b * c + ch) * iH * iW;
                float* dstPlane = dstBase + ((long)b * c + ch) * oH * oW;
                float* kernel = wBase + (long)ch * kH * kW;
                float biasVal = bBase is null ? 0f : bBase[ch];

                for (int oy = 0; oy < oH; oy++)
                {
                    int iy0 = oy * strideH - padH;
                    for (int ox = 0; ox < oW; ox++)
                    {
                        int ix0 = ox * strideW - padW;
                        float v = biasVal;
                        for (int ky = 0; ky < kH; ky++)
                        {
                            int iy = iy0 + ky;
                            if (iy < 0 || iy >= iH) continue;
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int ix = ix0 + kx;
                                if (ix < 0 || ix >= iW) continue;
                                v += kernel[ky * kW + kx] * srcPlane[iy * iW + ix];
                            }
                        }
                        dstPlane[oy * oW + ox] = v;
                    }
                }
            }
        }
    }

    /// <summary>2D max-pooling with explicit kernel, stride, and zero-padding. Used by YOLO's
    /// SPPF block (k=5, s=1, p=2 — preserves spatial dims). Default implementation is a CPU loop
    /// over F32 NCHW tensors; backends should override for performance.</summary>
    unsafe void MaxPool2D(Tensor output, Tensor input, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException($"MaxPool2D default fallback only supports F32 — got input={input.DType}, output={output.DType}. Override in the backend if you need other dtypes.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new ArgumentException($"MaxPool2D requires [N, C, H, W] tensors; got input {input.Shape} / output {output.Shape}.");

        int n = (int)input.Shape[0];
        int c = (int)input.Shape[1];
        int iH = (int)input.Shape[2];
        int iW = (int)input.Shape[3];
        int oH = (int)output.Shape[2];
        int oW = (int)output.Shape[3];

        float* srcBase = (float*)input.DataPointer;
        float* dstBase = (float*)output.DataPointer;

        // NCHW: outer indices [n, c] address a contiguous H*W plane.
        for (int b = 0; b < n; b++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                long planeOffset = ((long)b * c + ch) * iH * iW;
                long outPlaneOffset = ((long)b * c + ch) * oH * oW;
                float* srcPlane = srcBase + planeOffset;
                float* dstPlane = dstBase + outPlaneOffset;

                for (int oy = 0; oy < oH; oy++)
                {
                    int iy0 = oy * strideH - padH;
                    for (int ox = 0; ox < oW; ox++)
                    {
                        int ix0 = ox * strideW - padW;
                        float maxVal = float.NegativeInfinity;
                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            int iy = iy0 + ky;
                            if (iy < 0 || iy >= iH) continue;
                            for (int kx = 0; kx < kernelW; kx++)
                            {
                                int ix = ix0 + kx;
                                if (ix < 0 || ix >= iW) continue;
                                float v = srcPlane[iy * iW + ix];
                                if (v > maxVal) maxVal = v;
                            }
                        }
                        // If the entire receptive field was out-of-bounds (impossible for k=5,s=1,p=2
                        // but defensible for arbitrary configs), fall back to 0 instead of -inf to
                        // avoid poisoning downstream layers. Won't trigger in any YOLO config.
                        dstPlane[oy * oW + ox] = float.IsNegativeInfinity(maxVal) ? 0f : maxVal;
                    }
                }
            }
        }
    }

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

    /// <summary>Cast tensor from FP32 to BF16. Default: CPU fallback via Tensor.CastTo.</summary>
    void CastToBf16(Tensor output, Tensor input)
    {
        Tensor casted = input.CastTo(DType.BF16);
        try
        {
            unsafe
            {
                Buffer.MemoryCopy((void*)casted.DataPointer, (void*)output.DataPointer,
                    output.ElementCount * 2, casted.ElementCount * 2);
            }
        }
        finally { casted.Dispose(); }
    }

    /// <summary>Cast tensor from FP8 E4M3 to FP16. Default: CPU via F32 intermediate.</summary>
    void CastF8E4M3ToF16(Tensor output, Tensor input)
    {
        Tensor f32 = input.CastTo(DType.F32);
        CastToF16(output, f32);
        f32.Dispose();
    }

    /// <summary>Cast tensor from FP16 to FP8 E4M3. Default: CPU via Tensor.CastTo.</summary>
    void CastF16ToF8E4M3(Tensor output, Tensor input)
    {
        Tensor f8 = input.CastTo(DType.F8E4M3);
        unsafe
        {
            long byteCount = output.Shape.ElementCount; // 1 byte per F8 element
            Buffer.MemoryCopy(f8.DataPointer, output.DataPointer, byteCount, byteCount);
        }
        f8.Dispose();
    }

    // ── Synchronization ──────────────────────────────────────────────────

    /// <summary>Waits for all pending GPU work to complete. No-op on CPU backends. Call at pipeline phase boundaries to ensure deferred memory frees are processed before large allocations.</summary>
    void Sync() { }

    /// <summary>Frees specific weight tensors from accelerator memory. No-op on CPU backends. Call between pipeline phases to reclaim VRAM (e.g., free UNet weights before VAE decode).</summary>
    void FreeWeights(IEnumerable<Tensor> weights) { }

    /// <summary>Pre-uploads weights into the backend's weight cache so subsequent ops hit cached device memory instead of re-uploading per call. No-op on backends without a weight cache; pair with <see cref="FreeWeights"/> at pipeline phase boundaries.</summary>
    void PreloadWeights(IEnumerable<Tensor> weights) { }

    /// <summary>Streaming cache for backends that overlap weight uploads with compute on a side stream. <c>null</c> on backends without that capability — consumers should fall back to <see cref="PreloadWeights"/> + <see cref="FreeWeights"/>.</summary>
    IStreamingWeightCache? StreamingCache => null;
}
