using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae.QwenImage;

/// <summary>Decoder for the Qwen-Image / WAN 2.1 family 3D causal autoencoder, restricted to image
/// mode (<c>T = 1</c>). Decodes a 16-channel latent <c>[B, 16, H/8, W/8]</c> to an RGB image
/// <c>[B, 3, H, W]</c>. The encoder side is intentionally not implemented — Anima is t2i so the only
/// VAE direction we need at inference is decode.
///
/// <para><b>Structure</b> (verified by probing <c>qwen_image_vae.safetensors</c>):</para>
/// <list type="number">
///   <item><c>conv2</c> (top-level 1×1×1 conv, <c>[16, 16, 1, 1, 1]</c>) — post-quant identity-ish projection.</item>
///   <item><c>decoder.conv1</c> (3D causal conv, <c>[384, 16, 3, 3, 3]</c>) — latent → 384-ch features.</item>
///   <item><c>decoder.middle.0</c> ResidualBlock (384→384).</item>
///   <item><c>decoder.middle.1</c> AttentionBlock (384, single-head, global spatial).</item>
///   <item><c>decoder.middle.2</c> ResidualBlock (384→384).</item>
///   <item>15 upsample stages — fixed schedule:
///     <c>res(384→384), res(384→384), res(384→384), resample-2D(384→192),
///        res(192→384,shortcut), res(384→384), res(384→384), resample-2D(384→192),
///        res(192→192), res(192→192), res(192→192), resample-2D(192→96),
///        res(96→96), res(96→96), res(96→96)</c>.</item>
///   <item><c>decoder.head</c>: RMSNorm(96) → SiLU → 3D causal conv <c>[3, 96, 3, 3, 3]</c> → RGB.</item>
/// </list>
///
/// <para>All 3D causal convs collapse to 2D for <c>T = 1</c> via
/// <see cref="QwenImageVaeOps.SliceConv3dToConv2d"/> (last temporal slot for kT=3, slot 0 for kT=1).
/// The <c>time_conv</c> branches inside the spatial-resamples are skipped — they're for video temporal
/// upsampling, irrelevant when <c>T = 1</c>.</para></summary>
public sealed class QwenImageVaeDecoder : IDisposable
{
    /// <summary>Per-channel widths through the decoder, derived from the file's own shapes. Useful
    /// for sanity-checking the loaded checkpoint against expectations.</summary>
    public const int LatentChannels = 16;
    public const int MidChannels = 384;
    public const int FinalChannels = 96;

    /// <summary>Fixed upsample-stage schedule (matches <c>qwen_image_vae.safetensors</c> exactly).</summary>
    private static readonly UpsampleStage[] _stages =
    [
        // Three residuals at 384 ch (no spatial change).
        new(StageKind.Residual, 384, 384),
        new(StageKind.Residual, 384, 384),
        new(StageKind.Residual, 384, 384),
        // First spatial 2× upsample, channels reduce 384 → 192.
        new(StageKind.Resample, 384, 192),
        // Channel re-expand residual (with shortcut) — WAN architectural choice.
        new(StageKind.Residual, 192, 384),
        new(StageKind.Residual, 384, 384),
        new(StageKind.Residual, 384, 384),
        // Second spatial 2× upsample.
        new(StageKind.Resample, 384, 192),
        // Three residuals at 192.
        new(StageKind.Residual, 192, 192),
        new(StageKind.Residual, 192, 192),
        new(StageKind.Residual, 192, 192),
        // Third spatial 2× upsample (no time_conv on this last one — checkpoint confirms).
        new(StageKind.Resample, 192, 96),
        // Final residuals at 96.
        new(StageKind.Residual, 96, 96),
        new(StageKind.Residual, 96, 96),
        new(StageKind.Residual, 96, 96),
    ];

    private readonly VaeConfig _config;

    // Top-level conv2: post-quant 1×1×1 (collapsed to 1×1 2D).
    private Tensor? _conv2Weight;
    private Tensor? _conv2Bias;

    // decoder.conv1: 3D causal conv [384, 16, 3, 3, 3] (collapsed to [384, 16, 3, 3]).
    private Tensor? _convInWeight;
    private Tensor? _convInBias;

    // Middle (ResNet → Attention → ResNet) all at 384.
    private readonly QwenImageResidualBlock _midRes0 = new(MidChannels, MidChannels);
    private readonly QwenImageAttentionBlock _midAttn = new(MidChannels);
    private readonly QwenImageResidualBlock _midRes1 = new(MidChannels, MidChannels);

    // 15 upsample stages — built per the fixed schedule above.
    private readonly object[] _upStages;

    // Output head: RMSNorm(96) → SiLU → Conv3D-collapsed-to-2D (3-ch RGB output).
    private Tensor? _headGamma;
    private Tensor? _headConvWeight;  // [3, 96, 3, 3] (from [3, 96, 3, 3, 3])
    private Tensor? _headConvBias;

    private int _disposed;

    public QwenImageVaeDecoder(VaeConfig config)
    {
        _config = config;
        _upStages = new object[_stages.Length];
        for (int i = 0; i < _stages.Length; i++)
        {
            UpsampleStage s = _stages[i];
            _upStages[i] = s.Kind == StageKind.Residual
                ? new QwenImageResidualBlock(s.InCh, s.OutCh)
                : new QwenImageResample(s.InCh);
        }
    }

    public VaeConfig Config => _config;

    /// <summary>Loads decoder weights. <paramref name="weights"/> is the raw safetensors dict for the
    /// Qwen-Image VAE file (no key normalization required — we use the upstream key names directly).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // ── Top-level conv2 (post-quant) ──
        Tensor c2W = weights["conv2.weight"];
        _conv2Weight = QwenImageVaeOps.SliceConv3dToConv2d(c2W, temporalSlot: 0);  // kT = 1 → slot 0
        Tensor c2B = weights["conv2.bias"];
        _conv2Bias = c2B.DType == DType.F32 ? c2B : c2B.CastTo(DType.F32);

        // ── decoder.conv1: latent → 384 ch ──
        Tensor cinW = weights["decoder.conv1.weight"];
        _convInWeight = QwenImageVaeOps.SliceConv3dToConv2d(cinW, temporalSlot: -1);  // kT = 3 → last slot
        Tensor cinB = weights["decoder.conv1.bias"];
        _convInBias = cinB.DType == DType.F32 ? cinB : cinB.CastTo(DType.F32);

        // ── Middle blocks ──
        _midRes0.LoadWeights(weights, "decoder.middle.0");
        _midAttn.LoadWeights(weights, "decoder.middle.1");
        _midRes1.LoadWeights(weights, "decoder.middle.2");

        // ── 15 upsample stages ──
        for (int i = 0; i < _upStages.Length; i++)
        {
            string prefix = $"decoder.upsamples.{i}";
            switch (_upStages[i])
            {
                case QwenImageResidualBlock rb: rb.LoadWeights(weights, prefix); break;
                case QwenImageResample rs: rs.LoadWeights(weights, prefix); break;
            }
        }

        // ── Output head ──
        _headGamma = QwenImageVaeOps.FlattenGamma(weights["decoder.head.0.gamma"]);
        Tensor headW = weights["decoder.head.2.weight"];
        _headConvWeight = QwenImageVaeOps.SliceConv3dToConv2d(headW, temporalSlot: -1);
        Tensor headB = weights["decoder.head.2.bias"];
        _headConvBias = headB.DType == DType.F32 ? headB : headB.CastTo(DType.F32);
    }

    /// <summary>Enumerates every loaded weight tensor (for GPU preload accounting).</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_conv2Weight is not null) yield return _conv2Weight;
        if (_conv2Bias is not null) yield return _conv2Bias;
        if (_convInWeight is not null) yield return _convInWeight;
        if (_convInBias is not null) yield return _convInBias;
        foreach (Tensor w in _midRes0.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midAttn.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midRes1.EnumerateWeights()) yield return w;
        for (int i = 0; i < _upStages.Length; i++)
        {
            switch (_upStages[i])
            {
                case QwenImageResidualBlock rb:
                    foreach (Tensor w in rb.EnumerateWeights()) yield return w;
                    break;
                case QwenImageResample rs:
                    foreach (Tensor w in rs.EnumerateWeights()) yield return w;
                    break;
            }
        }
        if (_headGamma is not null) yield return _headGamma;
        if (_headConvWeight is not null) yield return _headConvWeight;
        if (_headConvBias is not null) yield return _headConvBias;
    }

    /// <summary>Decodes a latent <c>[B, 16, H, W]</c> to an RGB image <c>[B, 3, H*8, W*8]</c> in F32.
    /// Values are in roughly <c>[-1, 1]</c>; downstream <c>ImagePostProcessor.TensorToRgbBytes</c>
    /// rescales to <c>[0, 255]</c>.</summary>
    public unsafe Tensor Decode(IBackend backend, Tensor latent)
    {
        if (latent.Shape.Rank != 4)
            throw new ArgumentException($"Latent must be 4-D [B, C, H, W], got {latent.Shape}.", nameof(latent));
        if ((int)latent.Shape[1] != LatentChannels)
            throw new ArgumentException(
                $"Latent must have {LatentChannels} channels, got {latent.Shape[1]}.", nameof(latent));

        int batch = (int)latent.Shape[0];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];

        // ── 1. Undo scaling: z = latent / scale + shift ──
        Tensor z = UndoScaling(backend, latent);

        // ── 2. Top-level conv2 (post-quant 1×1) ──
        Tensor zPost = new Tensor(z.Shape, DType.F32);
        backend.Conv2D(zPost, z, _conv2Weight!, _conv2Bias, strideH: 1, strideW: 1, padH: 0, padW: 0);
        z.Dispose();
        QwenImageVaeDebugDump.Dump("post_quant_conv", zPost);

        // ── 3. decoder.conv1: 16 → 384 (3×3, pad=1) ──
        TensorShape midShape = new TensorShape(batch, MidChannels, height, width);
        Tensor hidden = new Tensor(midShape, DType.F32);
        backend.Conv2D(hidden, zPost, _convInWeight!, _convInBias, strideH: 1, strideW: 1, padH: 1, padW: 1);
        zPost.Dispose();
        QwenImageVaeDebugDump.Dump("decoder.conv_in", hidden);

        // ── 4. Middle: ResNet → Attention → ResNet ──
        Tensor afterMid0 = _midRes0.Forward(backend, hidden);
        hidden.Dispose();
        Tensor afterAttn = _midAttn.Forward(backend, afterMid0);
        afterMid0.Dispose();
        Tensor afterMid1 = _midRes1.Forward(backend, afterAttn);
        afterAttn.Dispose();
        hidden = afterMid1;
        QwenImageVaeDebugDump.Dump("decoder.mid_block", hidden);

        // ── 5. 15 upsample stages — grouped to match diffusers up_blocks {0..3} boundaries:
        //         raw idx 0..3   → up_blocks.0 (3 res + 1 upsampler) — dump after idx 3
        //         raw idx 4..7   → up_blocks.1 (3 res + 1 upsampler) — dump after idx 7
        //         raw idx 8..11  → up_blocks.2 (3 res + 1 upsampler) — dump after idx 11
        //         raw idx 12..14 → up_blocks.3 (3 res, no upsampler) — dump after idx 14
        for (int i = 0; i < _upStages.Length; i++)
        {
            Tensor next;
            switch (_upStages[i])
            {
                case QwenImageResidualBlock rb:
                    next = rb.Forward(backend, hidden);
                    break;
                case QwenImageResample rs:
                    next = rs.Forward(backend, hidden);
                    break;
                default:
                    throw new InvalidOperationException($"Unknown stage type at index {i}.");
            }
            hidden.Dispose();
            hidden = next;
            if (i == 3) QwenImageVaeDebugDump.Dump("decoder.up_blocks.0", hidden);
            else if (i == 7) QwenImageVaeDebugDump.Dump("decoder.up_blocks.1", hidden);
            else if (i == 11) QwenImageVaeDebugDump.Dump("decoder.up_blocks.2", hidden);
            else if (i == 14) QwenImageVaeDebugDump.Dump("decoder.up_blocks.3", hidden);
        }

        // ── 6. Head: RMSNorm(96) → SiLU → 3×3 Conv2D to RGB ──
        // Device norm (WanRmsNormChannel) — the host loop here drained the full-res [1,96,H,W] tensor.
        Tensor headNorm = new Tensor(hidden.Shape, DType.F32);
        backend.WanRmsNormChannel(headNorm, hidden, _headGamma!, 1e-12f);
        hidden.Dispose();

        Tensor headSilu = new Tensor(headNorm.Shape, DType.F32);
        backend.Silu(headSilu, headNorm);
        headNorm.Dispose();

        int finalH = (int)headSilu.Shape[2];
        int finalW = (int)headSilu.Shape[3];
        TensorShape rgbShape = new TensorShape(batch, 3, finalH, finalW);
        Tensor rgb = new Tensor(rgbShape, DType.F32);
        backend.Conv2D(rgb, headSilu, _headConvWeight!, _headConvBias, strideH: 1, strideW: 1, padH: 1, padW: 1);
        headSilu.Dispose();
        QwenImageVaeDebugDump.Dump("decoder.conv_out", rgb);

        // ── 7. Clamp to [-1, 1] — matches diffusers' AutoencoderKLQwenImage._decode:
        //   `out = torch.clamp(out, min=-1.0, max=1.0)` applied AFTER decoder.forward.
        //   Without this, the C# output exceeds [-1, 1] (saw range [-1.33, 1.21]) and
        //   image rescaling overflows the [0, 255] byte range. Device clamp; the caller's
        //   TensorToRgbBytes performs the single terminal D2H.
        Tensor clamped = new Tensor(rgb.Shape, DType.F32);
        backend.Clamp(clamped, rgb, -1.0f, 1.0f);
        rgb.Dispose();
        QwenImageVaeDebugDump.DumpOutput(clamped);

        return clamped;
    }

    /// <summary>Rescales the latent BEFORE the VAE decoder runs. Two paths:
    /// <list type="bullet">
    ///   <item><b>Per-channel</b> (Qwen-Image/Wan family): <c>output[b, c, h, w] = latent[b, c, h, w] * std[c] + mean[c]</c>.
    ///        Matches the diffusers Qwen-Image pipeline's pre-decode step (per
    ///        <c>diffusers/pipelines/qwenimage/pipeline_qwenimage.py</c> line 755-763:
    ///        <c>latents = latents / (1/std) + mean</c>, where <c>std</c> and <c>mean</c> are 16-vectors).</item>
    ///   <item><b>Scalar</b> (SD/Flux family, fallback): <c>output = latent / scaling_factor + shift_factor</c>.</item>
    /// </list>
    /// The per-channel path runs when <see cref="VaeConfig.LatentsMean"/> and <see cref="VaeConfig.LatentsStd"/>
    /// are both set on the config.</summary>
    private unsafe Tensor UndoScaling(IBackend backend, Tensor latent)
    {
        Tensor latentF32 = latent.DType == DType.F32 ? latent : latent.CastTo(DType.F32);

        if (_config.LatentsMean is float[] meanArr && _config.LatentsStd is float[] stdArr)
        {
            if (meanArr.Length != _config.LatentChannels || stdArr.Length != _config.LatentChannels)
                throw new InvalidOperationException(
                    $"VaeConfig.LatentsMean/Std length must equal LatentChannels ({_config.LatentChannels}); got mean={meanArr.Length}, std={stdArr.Length}.");

            int batch = (int)latentF32.Shape[0];
            int channels = (int)latentF32.Shape[1];
            int height = (int)latentF32.Shape[2];
            int width = (int)latentF32.Shape[3];
            long spatial = (long)height * width;

            Tensor result = new Tensor(latentF32.Shape, DType.F32);
            float* inPtr = (float*)latentF32.DataPointer;
            float* outPtr = (float*)result.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float scale = stdArr[c];
                    float shift = meanArr[c];
                    long bcOff = ((long)b * channels + c) * spatial;
                    for (long s = 0; s < spatial; s++)
                    {
                        outPtr[bcOff + s] = inPtr[bcOff + s] * scale + shift;
                    }
                }
            }
            if (!ReferenceEquals(latentF32, latent)) latentF32.Dispose();
            return result;
        }

        // Scalar fallback (SD/Flux convention): latent / scaling_factor + shift_factor
        Tensor scaled = new Tensor(latentF32.Shape, DType.F32);
        backend.Scale(scaled, latentF32, 1.0f / _config.ScalingFactor);
        if (!ReferenceEquals(latentF32, latent)) latentF32.Dispose();

        if (_config.ShiftFactor.HasValue)
        {
            Tensor shift = new Tensor(scaled.Shape, DType.F32);
            backend.Fill(shift, _config.ShiftFactor.Value);
            Tensor shifted = new Tensor(scaled.Shape, DType.F32);
            backend.Add(shifted, scaled, shift);
            scaled.Dispose();
            shift.Dispose();
            return shifted;
        }
        return scaled;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _conv2Weight = null;
            _conv2Bias = null;
            _convInWeight = null;
            _convInBias = null;
            _headGamma = null;
            _headConvWeight = null;
            _headConvBias = null;
        }
        GC.SuppressFinalize(this);
    }

    private enum StageKind { Residual, Resample }
    private readonly record struct UpsampleStage(StageKind Kind, int InCh, int OutCh);
}
