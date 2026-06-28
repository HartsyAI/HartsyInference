using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>ACE-Step's Music-DCAE decoder — NVIDIA's Sana <c>AutoencoderDC</c> (deep-compression autoencoder) applied
/// to 2-D stereo mel spectrograms: latent <c>[1, 8, 16, F_lat]</c> → mel <c>[1, 2, 128, F_lat·8]</c> (8× on both the
/// mel-bin and time axes). Structure per diffusers <c>autoencoder_dc.py</c>: <c>conv_in</c> (8→1024 with the
/// repeat-interleave channel shortcut) → stage 3 (EfficientViT ×3: Sana multiscale ReLU-linear attention + GLUMBConv)
/// → interpolate-upsample stages of ResBlocks (1024→512→256→128, each up-block with the repeat-interleave +
/// pixel-shuffle shortcut) → RMSNorm → ReLU → <c>conv_out</c> (128→2). First DCAE in the codebase — reusable for
/// future Sana image work. Numerics validation-pending. See <c>docs/Research/ACE_STEP_ARCHITECTURE.md</c> § 2.3.</summary>
public sealed unsafe class MusicDcaeDecoder
{
    private readonly int _latentChannels;
    private readonly int _outChannels;
    private readonly int[] _blockOut;
    private readonly int[] _layersPerBlock;
    private readonly int _attentionStages;   // trailing stages using EfficientViT blocks
    private readonly int _headDim;

    private Tensor? _convInW, _convInB;
    private Stage[] _stages = [];             // index = stage (ascending channel order), executed descending
    private Tensor? _normOutW, _normOutB;
    private Tensor? _convOutW, _convOutB;

    private sealed class Stage
    {
        public UpsampleConv? Upsample;        // present for stages < last
        public object[] Blocks = [];          // ResBlock2d or EfficientVitBlock
    }

    private sealed class UpsampleConv
    {
        public required Tensor W;
        public Tensor? B;
        public required int InChannels;
        public required int OutChannels;
    }

    public MusicDcaeDecoder(int latentChannels = 8, int outChannels = 2,
        int[]? blockOutChannels = null, int[]? layersPerBlock = null, int attentionStages = 1, int attentionHeadDim = 32)
    {
        _latentChannels = latentChannels;
        _outChannels = outChannels;
        _blockOut = blockOutChannels ?? [128, 256, 512, 1024];
        _layersPerBlock = layersPerBlock ?? [3, 3, 3, 3];
        _attentionStages = attentionStages;
        _headDim = attentionHeadDim;
    }

    /// <summary>Mel frames produced per latent frame (8× time upsample).</summary>
    public int TimeUpsample => 1 << (_blockOut.Length - 1);

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _convInW = w["decoder.conv_in.weight"]; w.TryGetValue("decoder.conv_in.bias", out _convInB);

        int numStages = _blockOut.Length;
        _stages = new Stage[numStages];
        for (int i = 0; i < numStages; i++)
        {
            Stage stage = new();
            int child = 0;
            if (i < numStages - 1)
            {
                Tensor upW = w[$"decoder.up_blocks.{i}.{child}.conv.weight"];
                w.TryGetValue($"decoder.up_blocks.{i}.{child}.conv.bias", out Tensor? upB);
                stage.Upsample = new UpsampleConv
                {
                    W = upW,
                    B = upB,
                    InChannels = _blockOut[i + 1],
                    OutChannels = _blockOut[i],
                };
                child++;
            }
            bool attention = i >= numStages - _attentionStages;
            stage.Blocks = new object[_layersPerBlock[i]];
            for (int j = 0; j < _layersPerBlock[i]; j++, child++)
            {
                string p = $"decoder.up_blocks.{i}.{child}";
                if (attention)
                {
                    EfficientVitBlock b = new(_blockOut[i], _headDim);
                    b.LoadWeights(w, p);
                    stage.Blocks[j] = b;
                }
                else
                {
                    ResBlock2d b = new();
                    b.LoadWeights(w, p);
                    stage.Blocks[j] = b;
                }
            }
            _stages[i] = stage;
        }

        _normOutW = LoadF32(w, "decoder.norm_out.weight"); w.TryGetValue("decoder.norm_out.bias", out _normOutB);
        _convOutW = w["decoder.conv_out.weight"]; w.TryGetValue("decoder.conv_out.bias", out _convOutB);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _convInW, _convInB, _normOutW, _normOutB, _convOutW, _convOutB })
            if (t is not null) yield return t;
        foreach (Stage s in _stages)
        {
            if (s.Upsample is not null)
            {
                yield return s.Upsample.W;
                if (s.Upsample.B is not null) yield return s.Upsample.B;
            }
            foreach (object b in s.Blocks)
            {
                IEnumerable<Tensor> ws = b is ResBlock2d r ? r.EnumerateWeights() : ((EfficientVitBlock)b).EnumerateWeights();
                foreach (Tensor t in ws) yield return t;
            }
        }
    }

    /// <summary>Decodes an (unscaled) latent <c>[1, 8, H_lat, W_lat]</c> → standardized mel <c>[1, 2, H_lat·8, W_lat·8]</c>
    /// (the pipeline applies the de-standardization and log-mel range).</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if ((int)latent.Shape[1] != _latentChannels)
            throw new ArgumentException($"latent channels {latent.Shape[1]} != {_latentChannels}.", nameof(latent));

        // conv_in with the repeat-interleave channel shortcut (8 → 1024).
        int deepest = _blockOut[^1];
        Tensor h = Conv3x3(backend, latent, _convInW!, _convInB, deepest);
        AddRepeatInterleaveChannels(h, latent, deepest / _latentChannels);
        AceStepDebugDump.Dump("dcae.conv_in", h);

        for (int i = _stages.Length - 1; i >= 0; i--)
        {
            Stage s = _stages[i];
            foreach (object b in s.Blocks)
            {
                Tensor next = b is ResBlock2d r ? r.Forward(backend, h) : ((EfficientVitBlock)b).Forward(backend, h);
                h.Dispose();
                h = next;
            }
            AceStepDebugDump.Dump($"dcae.stage.{i}", h);
            // NOTE: blocks run before this stage's own upsample? No — diffusers stores [upsample, blocks...] per stage
            // and executes stages in reverse, so the upsample of stage i fires when ENTERING stage i from stage i+1.
            if (i > 0 && _stages[i - 1].Upsample is UpsampleConv up)
            {
                Tensor upsampled = UpsampleInterpolateConv(backend, h, up);
                h.Dispose();
                h = upsampled;
                AceStepDebugDump.Dump($"dcae.upsample.{i - 1}", h);
            }
        }

        RmsNormChannels(h, _normOutW!, _normOutB);
        AceStepDebugDump.Dump("dcae.norm_out", h);
        Relu(h);
        Tensor mel = Conv3x3(backend, h, _convOutW!, _convOutB, _outChannels);
        h.Dispose();
        AceStepDebugDump.Dump("dcae.conv_out", mel);
        return mel;
    }

    /// <summary>Interpolate-nearest 2× then 3×3 conv, with the repeat-interleave + pixel-shuffle shortcut.</summary>
    private static Tensor UpsampleInterpolateConv(IBackend backend, Tensor x, UpsampleConv up)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], hh = (int)x.Shape[2], ww = (int)x.Shape[3];
        Tensor interp = new Tensor(new TensorShape(b, c, hh * 2, ww * 2), DType.F32);
        backend.UpsampleNearest2D(interp, x, 2, 2);
        Tensor conv = new Tensor(new TensorShape(b, up.OutChannels, hh * 2, ww * 2), DType.F32);
        backend.Conv2D(conv, interp, up.W, up.B, 1, 1, 1, 1);
        interp.Dispose();

        // Shortcut: repeat channels to out·4, pixel-shuffle(2) → [out, 2H, 2W], add.
        int repeats = up.OutChannels * 4 / c;
        float* xp = (float*)x.DataPointer;
        float* cp = (float*)conv.DataPointer;
        long frame = (long)hh * ww;
        long outFrame = frame * 4;
        for (int oc = 0; oc < up.OutChannels; oc++)
            for (int py = 0; py < 2; py++)
                for (int px = 0; px < 2; px++)
                {
                    int stacked = (oc * 2 + py) * 2 + px;      // pixel-shuffle channel order (c, r, r)
                    int src = stacked / repeats;                // inverse of repeat_interleave
                    for (int y = 0; y < hh; y++)
                        for (int xx = 0; xx < ww; xx++)
                            cp[(long)oc * outFrame + (long)(y * 2 + py) * (ww * 2) + (xx * 2 + px)]
                                += xp[(long)src * frame + (long)y * ww + xx];
                }
        return conv;
    }

    /// <summary>conv_in shortcut: input channels repeated to match, added in place.</summary>
    private static void AddRepeatInterleaveChannels(Tensor target, Tensor input, int repeats)
    {
        int c = (int)input.Shape[1];
        long frame = input.Shape[2] * input.Shape[3];
        float* tp = (float*)target.DataPointer;
        float* ip = (float*)input.DataPointer;
        for (int ci = 0; ci < c; ci++)
            for (int r = 0; r < repeats; r++)
            {
                long dst = ((long)(ci * repeats + r)) * frame;
                for (long i = 0; i < frame; i++) tp[dst + i] += ip[(long)ci * frame + i];
            }
    }

    internal static Tensor Conv3x3(IBackend backend, Tensor x, Tensor w, Tensor? b, int outChannels)
    {
        Tensor o = new Tensor(new TensorShape(x.Shape[0], outChannels, x.Shape[2], x.Shape[3]), DType.F32);
        backend.Conv2D(o, x, w, b, 1, 1, 1, 1);
        return o;
    }

    /// <summary>Channelwise RMSNorm (diffusers RMSNorm over the channel dim of NCHW), optional affine bias, in place.</summary>
    internal static void RmsNormChannels(Tensor x, Tensor weight, Tensor? bias, float eps = 1e-5f)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1];
        long spatial = x.Shape.ElementCount / ((long)b * c);
        float* xp = (float*)x.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (long s = 0; s < spatial; s++)
            {
                long basePos = (long)bi * c * spatial + s;
                double sum = 0;
                for (int ci = 0; ci < c; ci++) { float v = xp[basePos + ci * spatial]; sum += (double)v * v; }
                float inv = 1f / MathF.Sqrt((float)(sum / c) + eps);
                for (int ci = 0; ci < c; ci++)
                {
                    long off = basePos + ci * spatial;
                    xp[off] = xp[off] * inv * wp[ci] + (bp is null ? 0f : bp[ci]);
                }
            }
    }

    internal static void Relu(Tensor x)
    {
        float* p = (float*)x.DataPointer;
        long n = x.Shape.ElementCount;
        for (long i = 0; i < n; i++) if (p[i] < 0) p[i] = 0;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string k) { Tensor t = w[k]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
}
