using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Convolution, transposed convolution, depthwise and 3D forms, pooling, and frame-major Conv3D with its padding helpers.
public partial interface IBackend
{
    /// <summary>2D convolution: output = conv2d(input, weight, bias, stride, padding)</summary>
    void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW);

    /// <summary>3-D convolution over the frame-major padded buffer <see cref="BuildPaddedFrames"/> writes:
    /// <paramref name="paddedFrames"/> <c>[T, cIn, H, W]</c> (already padded in time), <paramref name="weight"/>
    /// <c>[cOut, cIn, kt, kh, kw]</c>, <paramref name="output"/> <c>[1, cOut, (T − kt)/strideT + 1, H', W']</c> with
    /// symmetric spatial padding and an optional per-channel <paramref name="bias"/>. One kernel with every tap in its
    /// reduction, instead of a 2-D pass per tap over all T frames. Returns false when the backend does not serve the
    /// call (the default), and the caller runs its per-tap 2-D decomposition instead.</summary>
    bool TryConv3DFrameMajor(Tensor output, Tensor paddedFrames, Tensor weight, Tensor? bias,
        int strideT, int strideH, int strideW, int padH, int padW) => false;

    /// <summary>Builds the padded input for BATCHED CausalConv3d: transpose + temporal pad + cache prepend + spatial
    /// pad. <paramref name="reflectSpatial"/> mirrors the H/W borders (<c>F.pad(mode="reflect")</c>, the LTX-2 VAE's
    /// default) instead of edge-clamping them.</summary>
    unsafe void BuildPaddedFrames(Tensor padded, Tensor input, Tensor? cache, int zeroPad,
        bool replicateFirst = false, int padH = 0, int padW = 0, bool reflectSpatial = false)
    {
        int paddedT = (int)padded.Shape[0], cIn = (int)padded.Shape[1];
        int h = (int)input.Shape[3], w = (int)input.Shape[4];
        int hp = h + 2 * padH, wp = w + 2 * padW;
        long hw = (long)h * w;
        int tin = (int)input.Shape[2];
        int cacheLen = cache is null ? 0 : (int)cache.Shape[2];
        float* pp = (float*)padded.DataPointer, ip = (float*)input.DataPointer;
        float* cp = cache is null ? null : (float*)cache.DataPointer;
        for (int t = 0; t < paddedT; t++)
            for (int ci = 0; ci < cIn; ci++)
            {
                long dst = ((long)t * cIn + ci) * hp * wp;
                if (t < zeroPad && !replicateFirst)
                {
                    for (long i = 0; i < (long)hp * wp; i++) pp[dst + i] = 0f;
                    continue;
                }
                int az = t < zeroPad ? 0 : t - zeroPad;   // replicate-first: pad frames clamp to the first content frame
                float* src;
                if (az < cacheLen) src = cp! + ((long)ci * cacheLen + az) * hw;
                else
                {
                    int ti = az - cacheLen;
                    if (ti >= tin) ti = tin - 1;          // trailing clamp (non-causal replicate)
                    src = ip + ((long)ci * tin + ti) * hw;
                }
                for (int y = 0; y < hp; y++)
                {
                    int sy = PadIndex(y - padH, h, reflectSpatial);
                    for (int x = 0; x < wp; x++)
                    {
                        int sx = PadIndex(x - padW, w, reflectSpatial);
                        pp[dst + (long)y * wp + x] = src[(long)sy * w + sx];
                    }
                }
            }
    }

    /// <summary>Maps an out-of-range spatial index to its source index — mirrored (<c>-1 → 1</c>, <c>len → len-2</c>,
    /// no edge repeat) when <paramref name="reflect"/>, else edge-clamped. The trailing clamp only bites when the pad
    /// exceeds <c>len-1</c>, where reflection has no defined answer.</summary>
    private static int PadIndex(int i, int len, bool reflect)
    {
        if (reflect)
        {
            if (i < 0) i = -i;
            else if (i >= len) i = 2 * (len - 1) - i;
        }
        return Math.Clamp(i, 0, len - 1);
    }

    /// <summary>1D convolution with asymmetric padding, channels-first; separate padLeft/padRight cover causal and symmetric padding.</summary>
    /// <param name="weight">PyTorch convention <c>[C_out, C_in / groups, K]</c>.</param>
    /// <param name="groups">Equal to channels for depthwise mode.</param>
    void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups);

    /// <summary>1D transposed convolution, channels-first; output length = <c>(T_in-1)*stride + dilation*(K-1) + 1 - pads</c>.</summary>
    /// <param name="weight">PyTorch convention <c>[C_in, C_out / groups, K]</c>.</param>
    /// <param name="padRight">VibeVoice/EnCodec causal decoders pass padLeft=0, padRight=K-stride to remove all trailing pad.</param>
    /// <param name="groups">Equal to channels for depthwise mode (e.g. BigVGAN anti-aliased upsampling).</param>
    void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups);

    /// <summary>2D transposed convolution (YOLO seg's Proto mask upsample); weight is <c>[C_in,C_out,kH,kW]</c> — input channels first.</summary>
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

    /// <summary>Dense 3D convolution (gather form) over <c>[N,C,D,H,W]</c>; used by TRELLIS's sparse-structure VAE decoder.</summary>
    unsafe void Conv3d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32)
            throw new NotSupportedException($"Conv3d default fallback only supports F32 — got input={input.DType}, output={output.DType}, weight={weight.DType}.");
        if (input.Shape.Rank != 5 || output.Shape.Rank != 5 || weight.Shape.Rank != 5)
            throw new ArgumentException($"Conv3d requires 5D [N,C,D,H,W] tensors; got input {input.Shape}, output {output.Shape}, weight {weight.Shape}.");

        int n = (int)input.Shape[0], cIn = (int)input.Shape[1];
        int iD = (int)input.Shape[2], iH = (int)input.Shape[3], iW = (int)input.Shape[4];
        int cOut = (int)output.Shape[1];
        int oD = (int)output.Shape[2], oH = (int)output.Shape[3], oW = (int)output.Shape[4];
        int kD = (int)weight.Shape[2], kH = (int)weight.Shape[3], kW = (int)weight.Shape[4];
        if (weight.Shape[0] != cOut || weight.Shape[1] != cIn)
            throw new ArgumentException($"Conv3d weight [{weight.Shape[0]},{weight.Shape[1]},...] must equal [C_out={cOut}, C_in={cIn}, ...].");

        float* srcBase = (float*)input.DataPointer, dstBase = (float*)output.DataPointer;
        float* wBase = (float*)weight.DataPointer, bBase = bias is null ? null : (float*)bias.DataPointer;

        for (int b = 0; b < n; b++)
            for (int co = 0; co < cOut; co++)
            {
                float biasVal = bBase is null ? 0f : bBase[co];
                float* dst = dstBase + (((long)b * cOut + co) * oD) * oH * oW;
                for (int od = 0; od < oD; od++)
                    for (int oh = 0; oh < oH; oh++)
                        for (int ow = 0; ow < oW; ow++)
                        {
                            float acc = biasVal;
                            int id0 = od * strideD - padD, ih0 = oh * strideH - padH, iw0 = ow * strideW - padW;
                            for (int ci = 0; ci < cIn; ci++)
                            {
                                float* src = srcBase + (((long)b * cIn + ci) * iD) * iH * iW;
                                long wOff = (((long)co * cIn + ci) * kD) * kH * kW;
                                for (int kd = 0; kd < kD; kd++)
                                {
                                    int id = id0 + kd; if (id < 0 || id >= iD) continue;
                                    for (int kh = 0; kh < kH; kh++)
                                    {
                                        int ih = ih0 + kh; if (ih < 0 || ih >= iH) continue;
                                        for (int kw = 0; kw < kW; kw++)
                                        {
                                            int iw = iw0 + kw; if (iw < 0 || iw >= iW) continue;
                                            acc += src[(id * iH + ih) * iW + iw] * wBase[wOff + (kd * kH + kh) * kW + kw];
                                        }
                                    }
                                }
                            }
                            dst[(od * oH + oh) * oW + ow] = acc;
                        }
            }
    }

    /// <summary>Depthwise 2D convolution (groups=C), weight <c>[C,1,kH,kW]</c> — used by YOLO11's class branch and C2PSA encoding.</summary>
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

    /// <summary>2D max-pooling with explicit kernel/stride/zero-padding — used by YOLO's SPPF block (k=5,s=1,p=2 preserves dims).</summary>
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
                        bool hasValue = false;
                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            int iy = iy0 + ky;
                            if (iy < 0 || iy >= iH) continue;
                            for (int kx = 0; kx < kernelW; kx++)
                            {
                                int ix = ix0 + kx;
                                if (ix < 0 || ix >= iW) continue;
                                hasValue = true;
                                float v = srcPlane[iy * iW + ix];
                                if (v > maxVal) maxVal = v;
                            }
                        }
                        // An empty padded window emits zero. A valid -infinity input remains -infinity.
                        dstPlane[oy * oW + ox] = hasValue ? maxVal : 0f;
                    }
                }
            }
        }
    }
}
