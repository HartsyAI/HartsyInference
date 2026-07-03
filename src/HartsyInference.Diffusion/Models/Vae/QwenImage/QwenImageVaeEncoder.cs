using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae.QwenImage;

/// <summary>Encoder for the Qwen-Image / WAN 2.1 family 3D causal autoencoder, image mode (<c>T = 1</c>).
/// The mirror of <see cref="QwenImageVaeDecoder"/>: encodes an RGB image <c>[B, 3, H, W]</c> to a scaled
/// 16-channel latent <c>[B, 16, H/8, W/8]</c> for the Anima / Qwen-Image img2img path. Reuses the same
/// <see cref="QwenImageResidualBlock"/> / <see cref="QwenImageAttentionBlock"/> / <see cref="QwenImageVaeOps"/>
/// primitives the decoder uses (only the strided <see cref="QwenImageDownsample"/> is new).
///
/// <para>Encodes deterministically — returns the posterior <b>mean</b> (no Gaussian sampling); the
/// downstream flow-match <c>AddNoise</c> supplies the stochasticity, same convention as
/// <see cref="VaeEncoder"/>. The scaled latent is <c>(mean − latents_mean) / latents_std</c> per channel
/// (the exact inverse of the decoder's <c>UndoScaling</c>).</para>
///
/// <para>The <c>encoder.downsamples.*</c> stage schedule is probed from the real
/// <c>qwen_image_vae.safetensors</c> (see <see cref="_stages"/>); block topology + ops are shared with the
/// validated decoder.</para></summary>
public sealed class QwenImageVaeEncoder : IDisposable
{
    public const int LatentChannels = 16;
    public const int MidChannels = 384;
    public const int FirstChannels = 96;

    /// <summary>Probed from the real <c>qwen_image_vae.safetensors</c> header (2026-07-02): channel widening
    /// happens in shortcut RESIDUAL blocks (3: 96→192, 6: 192→384); the strided downsamples keep channels.
    /// Stages 5/8 also carry a <c>time_conv</c> (temporal video path) that image mode skips, same as the decoder.</summary>
    private static readonly EncodeStage[] _stages =
    [
        new(StageKind.Residual, 96, 96),
        new(StageKind.Residual, 96, 96),
        new(StageKind.Downsample, 96, 96),       // spatial /2
        new(StageKind.Residual, 96, 192),        // shortcut residual widens
        new(StageKind.Residual, 192, 192),
        new(StageKind.Downsample, 192, 192),     // spatial /2 (+ skipped time_conv)
        new(StageKind.Residual, 192, 384),       // shortcut residual widens
        new(StageKind.Residual, 384, 384),
        new(StageKind.Downsample, 384, 384),     // spatial /2 (+ skipped time_conv)
        new(StageKind.Residual, 384, 384),
        new(StageKind.Residual, 384, 384),
    ];

    private readonly VaeConfig _config;

    // encoder.conv1: 3D causal conv [96, 3, 3, 3, 3] (collapsed to [96, 3, 3, 3]).
    private Tensor? _convInWeight;
    private Tensor? _convInBias;

    private readonly object[] _downStages;

    // Middle (ResNet → Attention → ResNet) at the deepest channel width.
    private readonly QwenImageResidualBlock _midRes0 = new(MidChannels, MidChannels);
    private readonly QwenImageAttentionBlock _midAttn = new(MidChannels);
    private readonly QwenImageResidualBlock _midRes1 = new(MidChannels, MidChannels);

    // Head: RMSNorm(384) → SiLU → 3D causal conv → 32-ch moments (mean ⊕ logvar).
    private Tensor? _headGamma;
    private Tensor? _headConvWeight;   // [32, 384, 3, 3]
    private Tensor? _headConvBias;

    // Top-level conv1 (quant): 1×1×1 over the 32-ch moments.
    private Tensor? _quantConvWeight;  // [32, 32, 1, 1]
    private Tensor? _quantConvBias;

    private int _disposed;

    public QwenImageVaeEncoder(VaeConfig config)
    {
        _config = config;
        _downStages = new object[_stages.Length];
        for (int i = 0; i < _stages.Length; i++)
        {
            EncodeStage s = _stages[i];
            _downStages[i] = s.Kind == StageKind.Residual
                ? new QwenImageResidualBlock(s.InCh, s.OutCh)
                : new QwenImageDownsample(s.InCh, s.OutCh);
        }
    }

    public VaeConfig Config => _config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        Tensor cinW = weights["encoder.conv1.weight"];
        _convInWeight = QwenImageVaeOps.SliceConv3dToConv2d(cinW, temporalSlot: -1);
        _convInBias = AsF32(weights["encoder.conv1.bias"]);

        for (int i = 0; i < _downStages.Length; i++)
        {
            string prefix = $"encoder.downsamples.{i}";
            switch (_downStages[i])
            {
                case QwenImageResidualBlock rb: rb.LoadWeights(weights, prefix); break;
                case QwenImageDownsample ds: ds.LoadWeights(weights, prefix); break;
            }
        }

        _midRes0.LoadWeights(weights, "encoder.middle.0");
        _midAttn.LoadWeights(weights, "encoder.middle.1");
        _midRes1.LoadWeights(weights, "encoder.middle.2");

        _headGamma = QwenImageVaeOps.FlattenGamma(weights["encoder.head.0.gamma"]);
        _headConvWeight = QwenImageVaeOps.SliceConv3dToConv2d(weights["encoder.head.2.weight"], temporalSlot: -1);
        _headConvBias = AsF32(weights["encoder.head.2.bias"]);

        _quantConvWeight = QwenImageVaeOps.SliceConv3dToConv2d(weights["conv1.weight"], temporalSlot: 0);
        _quantConvBias = AsF32(weights["conv1.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convInWeight is not null) yield return _convInWeight;
        if (_convInBias is not null) yield return _convInBias;
        for (int i = 0; i < _downStages.Length; i++)
            switch (_downStages[i])
            {
                case QwenImageResidualBlock rb: foreach (Tensor w in rb.EnumerateWeights()) yield return w; break;
                case QwenImageDownsample ds: foreach (Tensor w in ds.EnumerateWeights()) yield return w; break;
            }
        foreach (Tensor w in _midRes0.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midAttn.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midRes1.EnumerateWeights()) yield return w;
        if (_headGamma is not null) yield return _headGamma;
        if (_headConvWeight is not null) yield return _headConvWeight;
        if (_headConvBias is not null) yield return _headConvBias;
        if (_quantConvWeight is not null) yield return _quantConvWeight;
        if (_quantConvBias is not null) yield return _quantConvBias;
    }

    /// <summary>Encodes an RGB image <c>[B, 3, H, W]</c> (values in <c>[-1, 1]</c>) to a scaled latent
    /// <c>[B, 16, H/8, W/8]</c> — the posterior mean, scaled into the denoise-loop space.</summary>
    public unsafe Tensor Encode(IBackend backend, Tensor image)
    {
        if (image.Shape.Rank != 4 || (int)image.Shape[1] != 3)
            throw new ArgumentException($"Image must be [B, 3, H, W], got {image.Shape}.", nameof(image));
        int batch = (int)image.Shape[0];
        int h = (int)image.Shape[2];
        int w = (int)image.Shape[3];

        Tensor imgF32 = image.DType == DType.F32 ? image : image.CastTo(DType.F32);

        // conv_in: 3 → 96 (3×3, pad 1).
        Tensor hidden = new Tensor(new TensorShape(batch, FirstChannels, h, w), DType.F32);
        backend.Conv2D(hidden, imgF32, _convInWeight!, _convInBias, strideH: 1, strideW: 1, padH: 1, padW: 1);
        if (!ReferenceEquals(imgF32, image)) imgF32.Dispose();

        // Down stages.
        for (int i = 0; i < _downStages.Length; i++)
        {
            Tensor next = _downStages[i] switch
            {
                QwenImageResidualBlock rb => rb.Forward(backend, hidden),
                QwenImageDownsample ds => ds.Forward(backend, hidden),
                _ => throw new InvalidOperationException($"Unknown encode stage {i}."),
            };
            hidden.Dispose();
            hidden = next;
        }

        // Middle: ResNet → Attention → ResNet.
        Tensor m0 = _midRes0.Forward(backend, hidden); hidden.Dispose();
        Tensor ma = _midAttn.Forward(backend, m0); m0.Dispose();
        Tensor m1 = _midRes1.Forward(backend, ma); ma.Dispose();
        hidden = m1;

        // Head: RMSNorm → SiLU → conv → 32-ch moments.
        Tensor headNorm = new Tensor(hidden.Shape, DType.F32);
        QwenImageVaeOps.RmsNormPerPixelAcrossChannels(headNorm, hidden, _headGamma!);
        hidden.Dispose();
        Tensor headSilu = new Tensor(headNorm.Shape, DType.F32);
        backend.Silu(headSilu, headNorm);
        headNorm.Dispose();

        int latH = (int)headSilu.Shape[2];
        int latW = (int)headSilu.Shape[3];
        Tensor moments = new Tensor(new TensorShape(batch, 2 * LatentChannels, latH, latW), DType.F32);
        backend.Conv2D(moments, headSilu, _headConvWeight!, _headConvBias, strideH: 1, strideW: 1, padH: 1, padW: 1);
        headSilu.Dispose();

        // quant conv (1×1) over the moments.
        Tensor momentsQ = new Tensor(moments.Shape, DType.F32);
        backend.Conv2D(momentsQ, moments, _quantConvWeight!, _quantConvBias, strideH: 1, strideW: 1, padH: 0, padW: 0);
        moments.Dispose();

        // Posterior mean = first 16 channels; scale to denoise-loop space.
        Tensor scaled = ExtractMeanAndScale(momentsQ, batch, latH, latW);
        momentsQ.Dispose();
        return scaled;
    }

    /// <summary>Takes channels <c>[0, 16)</c> of the moments (the posterior mean) and applies the inverse
    /// of the decoder's per-channel undo-scaling: <c>latent = (mean − latents_mean) / latents_std</c>.</summary>
    private unsafe Tensor ExtractMeanAndScale(Tensor moments, int batch, int latH, int latW)
    {
        long spatial = (long)latH * latW;
        int momentsCh = (int)moments.Shape[1];
        Tensor outT = new Tensor(new TensorShape(batch, LatentChannels, latH, latW), DType.F32);
        float* mp = (float*)moments.DataPointer;
        float* op = (float*)outT.DataPointer;

        bool perChannel = _config.LatentsMean is float[] && _config.LatentsStd is float[];
        float[]? meanArr = _config.LatentsMean;
        float[]? stdArr = _config.LatentsStd;

        for (int b = 0; b < batch; b++)
            for (int c = 0; c < LatentChannels; c++)
            {
                float shift = perChannel ? meanArr![c] : (_config.ShiftFactor ?? 0f);
                float invStd = perChannel ? 1f / stdArr![c] : _config.ScalingFactor;
                long srcBase = ((long)b * momentsCh + c) * spatial;
                long dstBase = ((long)b * LatentChannels + c) * spatial;
                for (long s = 0; s < spatial; s++)
                    op[dstBase + s] = (mp[srcBase + s] - shift) * invStd;
            }
        return outT;
    }

    private static Tensor AsF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _convInWeight = null; _convInBias = null;
            _headGamma = null; _headConvWeight = null; _headConvBias = null;
            _quantConvWeight = null; _quantConvBias = null;
        }
        GC.SuppressFinalize(this);
    }

    private enum StageKind { Residual, Downsample }
    private readonly record struct EncodeStage(StageKind Kind, int InCh, int OutCh);
}

/// <summary>Encoder strided downsample — the WAN <c>Resample</c> in <c>downsample2d</c> mode: a
/// channel-changing <c>Conv2d(in, out, k=3, stride=2, pad=1)</c> that halves H/W. Mirrors
/// <see cref="QwenImageResample"/> (upsample) with the same <c>resample.1.weight/bias</c> key convention
/// (the <c>.0</c> slot is the parameter-less pad). The asymmetric <c>(0,1,0,1)</c> pad WAN uses is
/// approximated with symmetric pad=1, same sub-pixel compromise as <see cref="VaeEncoder"/>.</summary>
public sealed class QwenImageDownsample
{
    private readonly int _inDim;
    private readonly int _outDim;
    private Tensor? _convWeight;   // [out, in, 3, 3]
    private Tensor? _convBias;

    public QwenImageDownsample(int inDim, int outDim)
    {
        _inDim = inDim;
        _outDim = outDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // Spatial resamples are stored as native 2-D convs [out, in, kH, kW] (same as the decoder's
        // QwenImageResample); only the residual/head convs carry a temporal axis.
        Tensor w = weights[$"{prefix}.resample.1.weight"];
        _convWeight = w.Shape.Rank == 5 ? QwenImageVaeOps.SliceConv3dToConv2d(w, temporalSlot: -1)
            : (w.DType == DType.F32 ? w : w.CastTo(DType.F32));
        Tensor b = weights[$"{prefix}.resample.1.bias"];
        _convBias = b.DType == DType.F32 ? b : b.CastTo(DType.F32);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convWeight is not null) yield return _convWeight;
        if (_convBias is not null) yield return _convBias;
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int batch = (int)x.Shape[0];
        int h = (int)x.Shape[2];
        int w = (int)x.Shape[3];
        int outH = (h + 2 * 1 - 3) / 2 + 1;
        int outW = (w + 2 * 1 - 3) / 2 + 1;
        Tensor outT = new Tensor(new TensorShape(batch, _outDim, outH, outW), DType.F32);
        backend.Conv2D(outT, x, _convWeight!, _convBias, strideH: 2, strideW: 2, padH: 1, padW: 1);
        return outT;
    }
}
