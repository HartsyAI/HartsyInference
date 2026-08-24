using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae.QwenImage;

/// <summary>A WAN-family <c>ResidualBlock</c> collapsed to image-mode (<c>T = 1</c>) 2D convs. The
/// upstream PyTorch module is:
/// <code>
/// nn.Sequential(
///     RMS_norm(in_dim),    # .0 — gamma [in_dim, 1, 1, 1]
///     nn.SiLU(),           # .1
///     CausalConv3d(in_dim, out_dim, 3, padding=1),   # .2 — weight [out, in, 3, 3, 3]
///     RMS_norm(out_dim),   # .3 — gamma [out_dim, 1, 1, 1]
///     nn.SiLU(),           # .4
///     nn.Dropout(0),       # .5 (no params, no-op at eval)
///     CausalConv3d(out_dim, out_dim, 3, padding=1))  # .6 — weight [out, out, 3, 3, 3]
///
/// shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
/// </code>
/// The block's forward is <c>residual(x) + shortcut(x)</c>. We collapse every Conv3D to a 2D conv
/// per <see cref="QwenImageVaeOps.SliceConv3dToConv2d"/>. Both conv kernels are 3×3×3 → 3×3
/// (last temporal slot); the shortcut conv (if present) is 1×1×1 → 1×1 (only slot).</summary>
public sealed class QwenImageResidualBlock
{
    private readonly int _inDim;
    private readonly int _outDim;
    private readonly bool _hasShortcut;

    private Tensor? _gamma0;       // RmsNorm gamma for input  (in_dim)
    private Tensor? _conv2Weight;  // Conv3D→Conv2D weight [out, in, 3, 3]
    private Tensor? _conv2Bias;    // [out]
    private Tensor? _gamma3;       // RmsNorm gamma for mid    (out_dim)
    private Tensor? _conv6Weight;  // [out, out, 3, 3]
    private Tensor? _conv6Bias;    // [out]
    private Tensor? _shortcutWeight;  // [out, in, 1, 1] if in != out
    private Tensor? _shortcutBias;    // [out]

    public QwenImageResidualBlock(int inDim, int outDim)
    {
        _inDim = inDim;
        _outDim = outDim;
        _hasShortcut = inDim != outDim;
    }

    /// <summary>Loads block weights. <paramref name="prefix"/> is e.g.
    /// <c>decoder.upsamples.4</c> — and the keys look like
    /// <c>{prefix}.residual.0.gamma</c>, <c>{prefix}.residual.2.weight</c>,
    /// <c>{prefix}.shortcut.weight</c>, etc.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _gamma0 = QwenImageVaeOps.FlattenGamma(weights[$"{prefix}.residual.0.gamma"]);

        Tensor c2 = weights[$"{prefix}.residual.2.weight"];
        _conv2Weight = QwenImageVaeOps.SliceConv3dToConv2d(c2, temporalSlot: -1);  // kT=3 → take last slot
        _conv2Bias = weights[$"{prefix}.residual.2.bias"].DType == DType.F32 ? weights[$"{prefix}.residual.2.bias"]
            : weights[$"{prefix}.residual.2.bias"].CastTo(DType.F32);

        _gamma3 = QwenImageVaeOps.FlattenGamma(weights[$"{prefix}.residual.3.gamma"]);

        Tensor c6 = weights[$"{prefix}.residual.6.weight"];
        _conv6Weight = QwenImageVaeOps.SliceConv3dToConv2d(c6, temporalSlot: -1);
        _conv6Bias = weights[$"{prefix}.residual.6.bias"].DType == DType.F32 ? weights[$"{prefix}.residual.6.bias"]
            : weights[$"{prefix}.residual.6.bias"].CastTo(DType.F32);

        if (_hasShortcut)
        {
            Tensor sc = weights[$"{prefix}.shortcut.weight"];
            // Shortcut is CausalConv3d(in, out, 1) — kT = kH = kW = 1.
            _shortcutWeight = QwenImageVaeOps.SliceConv3dToConv2d(sc, temporalSlot: 0);
            _shortcutBias = weights[$"{prefix}.shortcut.bias"].DType == DType.F32 ? weights[$"{prefix}.shortcut.bias"]
                : weights[$"{prefix}.shortcut.bias"].CastTo(DType.F32);
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_gamma0 is not null) yield return _gamma0;
        if (_conv2Weight is not null) yield return _conv2Weight;
        if (_conv2Bias is not null) yield return _conv2Bias;
        if (_gamma3 is not null) yield return _gamma3;
        if (_conv6Weight is not null) yield return _conv6Weight;
        if (_conv6Bias is not null) yield return _conv6Bias;
        if (_shortcutWeight is not null) yield return _shortcutWeight;
        if (_shortcutBias is not null) yield return _shortcutBias;
    }

    /// <summary>Forward pass. <paramref name="x"/> is <c>[B, in_dim, H, W]</c> F32; output is
    /// <c>[B, out_dim, H, W]</c> F32.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        int batch = (int)x.Shape[0];
        int height = (int)x.Shape[2];
        int width = (int)x.Shape[3];

        // Residual path: RMSNorm → SiLU → Conv2d → RMSNorm → SiLU → Conv2d.
        // GPU-residency port: the norms and the residual add run as backend ops (WanRmsNormChannel — the
        // same WAN RMS_norm the video VAE uses — and device Add). The previous host loops each forced a
        // full D2H drain of an up-to-[1,96,1024,1024] conv output + CPU triple-loop + H2D re-upload; over
        // ~17 residual blocks that was ~20 s of the 27 s Krea2 gen (the VAE, not the DiT, was the gap).
        Tensor h = new Tensor(x.Shape, DType.F32);
        backend.WanRmsNormChannel(h, x, _gamma0!, 1e-12f);

        Tensor hSilu = new Tensor(h.Shape, DType.F32);
        backend.Silu(hSilu, h);
        h.Dispose();

        TensorShape midShape = new TensorShape(batch, _outDim, height, width);
        Tensor hMid = new Tensor(midShape, DType.F32);
        backend.Conv2D(hMid, hSilu, _conv2Weight!, _conv2Bias, strideH: 1, strideW: 1, padH: 1, padW: 1);
        hSilu.Dispose();

        Tensor hNorm = new Tensor(midShape, DType.F32);
        backend.WanRmsNormChannel(hNorm, hMid, _gamma3!, 1e-12f);
        hMid.Dispose();

        Tensor hSilu2 = new Tensor(midShape, DType.F32);
        backend.Silu(hSilu2, hNorm);
        hNorm.Dispose();

        Tensor hOut = new Tensor(midShape, DType.F32);
        backend.Conv2D(hOut, hSilu2, _conv6Weight!, _conv6Bias, strideH: 1, strideW: 1, padH: 1, padW: 1);
        hSilu2.Dispose();

        // Shortcut path: identity (in == out) or 1×1 conv (in != out).
        Tensor shortcut;
        if (_hasShortcut)
        {
            shortcut = new Tensor(midShape, DType.F32);
            backend.Conv2D(shortcut, x, _shortcutWeight!, _shortcutBias, strideH: 1, strideW: 1, padH: 0, padW: 0);
        }
        else
        {
            shortcut = x;  // alias — caller owns x; we don't free it
        }

        // Device residual add into a fresh tensor (x may be the caller's tensor when shortcut aliases it).
        Tensor sum = new Tensor(midShape, DType.F32);
        backend.Add(sum, hOut, shortcut);
        hOut.Dispose();
        if (_hasShortcut) shortcut.Dispose();
        return sum;
    }
}

/// <summary>WAN-family <c>AttentionBlock</c> for the VAE bottleneck. Single-head global attention
/// over spatial positions, with channels as the feature dim. Layout (image mode, T=1):
/// <code>
/// norm   = RMS_norm(dim, images=True)   # gamma [dim, 1, 1]
/// to_qkv = Conv2d(dim, 3*dim, 1)        # weight [3*dim, dim, 1, 1], bias [3*dim]
/// proj   = Conv2d(dim, dim, 1)          # weight [dim, dim, 1, 1], bias [dim]
///
/// def forward(x):
///     identity = x
///     x = norm(x)
///     qkv = to_qkv(x).reshape(B, 1, dim*3, H*W).permute(0, 1, 3, 2)
///     # split into q, k, v each [B, 1, H*W, dim]
///     out = SDPA(q, k, v)   # [B, 1, H*W, dim]
///     out = out.permute(...).reshape(B, dim, H, W)
///     out = proj(out)
///     return identity + out
/// </code>
/// Both <c>to_qkv</c> and <c>proj</c> are <c>Conv2d</c> 1×1 (the checkpoint stores them with 4-D
/// weight shape <c>[out, in, 1, 1]</c> directly — these are NOT 5-D conv weights despite the
/// surrounding 3D residual blocks).</summary>
public sealed class QwenImageAttentionBlock
{
    private readonly int _dim;

    private Tensor? _normGamma;     // [dim] (flattened from [dim, 1, 1])
    private Tensor? _qkvWeight;     // [3*dim, dim, 1, 1]
    private Tensor? _qkvBias;       // [3*dim]
    private Tensor? _projWeight;    // [dim, dim, 1, 1]
    private Tensor? _projBias;      // [dim]

    public QwenImageAttentionBlock(int dim) { _dim = dim; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _normGamma = QwenImageVaeOps.FlattenGamma(weights[$"{prefix}.norm.gamma"]);

        // The to_qkv/proj weights are STORED as 4-D [out, in, 1, 1] for 2D Conv1×1 — verified by
        // the actual file (e.g. decoder.middle.1.to_qkv.weight has shape [1152, 384, 1, 1]).
        Tensor qkv = weights[$"{prefix}.to_qkv.weight"];
        _qkvWeight = qkv.DType == DType.F32 ? qkv : qkv.CastTo(DType.F32);

        Tensor qkvB = weights[$"{prefix}.to_qkv.bias"];
        _qkvBias = qkvB.DType == DType.F32 ? qkvB : qkvB.CastTo(DType.F32);

        Tensor pj = weights[$"{prefix}.proj.weight"];
        _projWeight = pj.DType == DType.F32 ? pj : pj.CastTo(DType.F32);

        Tensor pjB = weights[$"{prefix}.proj.bias"];
        _projBias = pjB.DType == DType.F32 ? pjB : pjB.CastTo(DType.F32);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_normGamma is not null) yield return _normGamma;
        if (_qkvWeight is not null) yield return _qkvWeight;
        if (_qkvBias is not null) yield return _qkvBias;
        if (_projWeight is not null) yield return _projWeight;
        if (_projBias is not null) yield return _projBias;
    }

    /// <summary>Forward pass: <c>[B, dim, H, W] → [B, dim, H, W]</c> (identity + attention residual).</summary>
    public unsafe Tensor Forward(IBackend backend, Tensor x)
    {
        int batch = (int)x.Shape[0];
        int height = (int)x.Shape[2];
        int width = (int)x.Shape[3];
        long seqLen = (long)height * width;

        // 1. RMSNorm across channel dim (device — see QwenImageResidualBlock for the port rationale).
        Tensor normed = new Tensor(x.Shape, DType.F32);
        backend.WanRmsNormChannel(normed, x, _normGamma!, 1e-12f);

        // 2. to_qkv: Conv2D 1×1, producing [B, 3*dim, H, W].
        TensorShape qkvShape = new TensorShape(batch, 3 * _dim, height, width);
        Tensor qkv = new Tensor(qkvShape, DType.F32);
        backend.Conv2D(qkv, normed, _qkvWeight!, _qkvBias, strideH: 1, strideW: 1, padH: 0, padW: 0);
        normed.Dispose();

        // 3. Split qkv into q/k/v [B, 1, seqLen, dim] for SDPA — device for B=1 (the pipeline case):
        //    each channel block [dim, seqLen] is a contiguous slab of the flat qkv (SliceRows), and the
        //    channel-major → sequence-major flip is a 2-D transpose. The old host loops D2H-drained the
        //    ~75 MB qkv + re-uploaded q/k/v every call.
        TensorShape qkvHeadShape = new TensorShape(batch, 1, seqLen, _dim);
        Tensor q, k, v;
        if (batch == 1)
        {
            q = SliceBlockTransposed(backend, qkv, 0, seqLen, qkvHeadShape);
            k = SliceBlockTransposed(backend, qkv, 1, seqLen, qkvHeadShape);
            v = SliceBlockTransposed(backend, qkv, 2, seqLen, qkvHeadShape);
        }
        else
        {
            (q, k, v) = SplitQkvHost(qkv, batch, seqLen, qkvHeadShape);
        }
        qkv.Dispose();

        // 4. SDPA: scale = 1 / sqrt(dim). Output [B, 1, seqLen, dim].
        float scale = 1.0f / MathF.Sqrt(_dim);
        Tensor attnOut = new Tensor(qkvHeadShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, q, k, v, mask: null, scale);
        q.Dispose();
        k.Dispose();
        v.Dispose();

        // 5. Reshape [B, 1, seqLen, dim] back to [B, dim, H, W] — the inverse transpose, on-device for B=1.
        Tensor reshaped = new Tensor(x.Shape, DType.F32);
        if (batch == 1)
        {
            backend.Transpose2D(reshaped, attnOut, (int)seqLen, _dim);
        }
        else
        {
            ReshapeAttnHost(attnOut, reshaped, batch, seqLen);
        }
        attnOut.Dispose();

        // 6. proj: Conv2D 1×1.
        Tensor projOut = new Tensor(x.Shape, DType.F32);
        backend.Conv2D(projOut, reshaped, _projWeight!, _projBias, strideH: 1, strideW: 1, padH: 0, padW: 0);
        reshaped.Dispose();

        // 7. Residual add (device).
        Tensor outp = new Tensor(x.Shape, DType.F32);
        backend.Add(outp, projOut, x);
        projOut.Dispose();
        return outp;
    }

    /// <summary>Device path (B=1): extracts channel block <paramref name="block"/> of the flat
    /// <c>[1, 3·dim, S]</c> qkv (a contiguous slab → SliceRows) and transposes <c>[dim, S] → [S, dim]</c>.</summary>
    private Tensor SliceBlockTransposed(IBackend backend, Tensor qkv, int block, long seqLen, TensorShape headShape)
    {
        Tensor chanMajor = new Tensor(new TensorShape(1, _dim, seqLen), DType.F32);
        backend.SliceRows(chanMajor, qkv, block * _dim);   // elemOffset = block·dim·seqLen
        Tensor seqMajor = new Tensor(headShape, DType.F32);
        backend.Transpose2D(seqMajor, chanMajor, _dim, (int)seqLen);
        chanMajor.Dispose();
        return seqMajor;
    }

    /// <summary>Host fallback for B&gt;1 (unused by the image pipelines).</summary>
    private unsafe (Tensor q, Tensor k, Tensor v) SplitQkvHost(Tensor qkv, int batch, long seqLen, TensorShape headShape)
    {
        Tensor q = new Tensor(headShape, DType.F32);
        Tensor k = new Tensor(headShape, DType.F32);
        Tensor v = new Tensor(headShape, DType.F32);
        float* qkvPtr = (float*)qkv.DataPointer;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;
        float* vPtr = (float*)v.DataPointer;
        long stride_b = 3L * _dim * seqLen;
        for (int b = 0; b < batch; b++)
        {
            long bOff = b * stride_b;
            for (long s = 0; s < seqLen; s++)
            {
                for (int c = 0; c < _dim; c++)
                {
                    qPtr[(long)b * seqLen * _dim + s * _dim + c] = qkvPtr[bOff + ((long)0 * _dim + c) * seqLen + s];
                    kPtr[(long)b * seqLen * _dim + s * _dim + c] = qkvPtr[bOff + ((long)1 * _dim + c) * seqLen + s];
                    vPtr[(long)b * seqLen * _dim + s * _dim + c] = qkvPtr[bOff + ((long)2 * _dim + c) * seqLen + s];
                }
            }
        }
        return (q, k, v);
    }

    /// <summary>Host fallback for B&gt;1: [B, 1, S, dim] → [B, dim, H·W].</summary>
    private unsafe void ReshapeAttnHost(Tensor attnOut, Tensor reshaped, int batch, long seqLen)
    {
        float* attnPtr = (float*)attnOut.DataPointer;
        float* reshPtr = (float*)reshaped.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < _dim; c++)
            {
                long inBase = (long)b * seqLen * _dim;
                long outBase = ((long)b * _dim + c) * seqLen;
                for (long s = 0; s < seqLen; s++)
                {
                    reshPtr[outBase + s] = attnPtr[inBase + s * _dim + c];
                }
            }
        }
    }
}

/// <summary>WAN-family <c>Resample</c> in <c>upsample2d</c> or <c>upsample3d</c> mode. Spatially
/// upsamples 2×2 (nearest), then applies a 2D conv that REDUCES channels by half. For
/// <c>upsample3d</c> mode there's also a <c>time_conv</c> (3, 1, 1) 3D conv that doubles the
/// temporal dim — which we skip in image mode (T=1).
///
/// <para>Stored as: <c>resample.1.weight</c> (out=dim/2, in=dim, kH=3, kW=3) +
/// <c>resample.1.bias</c> (dim/2). The <c>.0</c> slot is the nn.Upsample (no params).</para></summary>
public sealed class QwenImageResample
{
    private readonly int _inDim;
    private readonly int _outDim;

    private Tensor? _convWeight;  // [outDim, inDim, 3, 3]
    private Tensor? _convBias;    // [outDim]

    public QwenImageResample(int inDim)
    {
        _inDim = inDim;
        _outDim = inDim / 2;  // WAN convention
    }

    public int OutDim => _outDim;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // The 2D resample.1.weight is already 4-D [out, in, kH, kW] — no temporal axis.
        Tensor w = weights[$"{prefix}.resample.1.weight"];
        _convWeight = w.DType == DType.F32 ? w : w.CastTo(DType.F32);

        Tensor b = weights[$"{prefix}.resample.1.bias"];
        _convBias = b.DType == DType.F32 ? b : b.CastTo(DType.F32);

        // NOTE: time_conv (when present) is intentionally skipped — it's the temporal-upsample path
        // for video mode. Image mode runs with T=1 throughout, so the temporal-expansion output
        // would just produce duplicate frames we'd discard.
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convWeight is not null) yield return _convWeight;
        if (_convBias is not null) yield return _convBias;
    }

    /// <summary>Forward: <c>[B, in, H, W] → [B, out, 2H, 2W]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        int batch = (int)x.Shape[0];
        int height = (int)x.Shape[2];
        int width = (int)x.Shape[3];

        TensorShape upShape = new TensorShape(batch, _inDim, height * 2, width * 2);
        Tensor upsampled = new Tensor(upShape, DType.F32);
        backend.UpsampleNearest2D(upsampled, x, scaleH: 2, scaleW: 2);

        TensorShape outShape = new TensorShape(batch, _outDim, height * 2, width * 2);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Conv2D(output, upsampled, _convWeight!, _convBias, strideH: 1, strideW: 1, padH: 1, padW: 1);
        upsampled.Dispose();
        return output;
    }
}
