using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>HunyuanImage 2.1 pixel-shuffle VAE encoder — the exact mirror of <see cref="HunyuanImageVaeDecoder"/>,
/// and the piece whose absence disabled img2img/inpaint for this family (the generic <see cref="VaeEncoder"/> can't
/// model it: every resnet here is channel-preserving and the channel changes happen in the downsamplers).
/// <para>Each of the 5 downsamplers is the residual space-to-channel step of the DC-AE design the decoder mirrors:
/// main path <c>conv([C_next/4, C, 3, 3])</c> then a space-to-depth rearrange in the decoder's <b>[r1, r2, c]</b>
/// channel order; shortcut path space-to-depth of the input then a channel-group average down to <c>C_next</c> (the
/// inverse of the decoder's <c>repeat_interleave</c>). The final <c>conv_out</c> carries the mirror of the decoder's
/// <c>conv_in</c> skip: a channel-group average of the pre-norm features onto the 128 moment channels. Encode
/// returns the posterior MEAN (the generic encoder's convention — deterministic, the denoise loop adds its own
/// noise) scaled by <see cref="VaeConfig.ScalingFactor"/>.</para></summary>
public sealed class HunyuanImageVaeEncoder : IDisposable
{
    // Encoder-order channel plan (= VaeConfig.HunyuanImage.BlockOutChannels).
    private static readonly int[] ChannelPlan = [128, 256, 512, 512, 1024, 1024];
    private const int ResnetsPerLevel = 2;
    private const int NormGroups = 32;
    private const float NormEps = 1e-6f;

    private readonly VaeConfig _config;
    private Tensor? _convInWeight, _convInBias;
    private readonly ResNetBlock2D[][] _levelResnets;
    private readonly Tensor?[] _downsampleWeights = new Tensor?[ChannelPlan.Length - 1];
    private readonly Tensor?[] _downsampleBiases = new Tensor?[ChannelPlan.Length - 1];
    private readonly ResNetBlock2D _midResnet0 = new(ChannelPlan[^1], ChannelPlan[^1]);
    private readonly ResNetBlock2D _midResnet1 = new(ChannelPlan[^1], ChannelPlan[^1]);
    private readonly VaeAttention _midAttention = new(ChannelPlan[^1]);
    private Tensor? _normOutWeight, _normOutBias, _convOutWeight, _convOutBias;
    private bool _disposed;

    public HunyuanImageVaeEncoder(VaeConfig config)
    {
        _config = config;
        _levelResnets = new ResNetBlock2D[ChannelPlan.Length][];
        for (int lvl = 0; lvl < ChannelPlan.Length; lvl++)
        {
            _levelResnets[lvl] = new ResNetBlock2D[ResnetsPerLevel];
            for (int i = 0; i < ResnetsPerLevel; i++)
                _levelResnets[lvl][i] = new ResNetBlock2D(ChannelPlan[lvl], ChannelPlan[lvl]);
        }
    }

    /// <summary>Loads diffusers-named encoder weights (the same <c>ConvertVaeKey</c> remap the decoder uses).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _convInWeight = weights["encoder.conv_in.weight"];
        _convInBias = weights["encoder.conv_in.bias"];
        for (int lvl = 0; lvl < ChannelPlan.Length; lvl++)
        {
            for (int i = 0; i < ResnetsPerLevel; i++)
                _levelResnets[lvl][i].LoadWeights(weights, $"encoder.down_blocks.{lvl}.resnets.{i}");
            if (lvl < ChannelPlan.Length - 1)
            {
                _downsampleWeights[lvl] = weights[$"encoder.down_blocks.{lvl}.downsamplers.0.conv.weight"];
                _downsampleBiases[lvl] = weights[$"encoder.down_blocks.{lvl}.downsamplers.0.conv.bias"];
            }
        }
        _midResnet0.LoadWeights(weights, "encoder.mid_block.resnets.0");
        _midResnet1.LoadWeights(weights, "encoder.mid_block.resnets.1");
        _midAttention.LoadWeights(weights, "encoder.mid_block.attentions.0");
        _normOutWeight = weights["encoder.conv_norm_out.weight"];
        _normOutBias = weights["encoder.conv_norm_out.bias"];
        _convOutWeight = weights["encoder.conv_out.weight"];
        _convOutBias = weights["encoder.conv_out.bias"];
    }

    /// <summary>Enumerates all weight tensors for GPU preload/free.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convInWeight is not null) yield return _convInWeight;
        if (_convInBias is not null) yield return _convInBias;
        foreach (ResNetBlock2D[] level in _levelResnets)
            foreach (ResNetBlock2D block in level)
                foreach (Tensor t in block.EnumerateWeights()) yield return t;
        for (int i = 0; i < _downsampleWeights.Length; i++)
        {
            if (_downsampleWeights[i] is not null) yield return _downsampleWeights[i]!;
            if (_downsampleBiases[i] is not null) yield return _downsampleBiases[i]!;
        }
        foreach (Tensor t in _midResnet0.EnumerateWeights()) yield return t;
        foreach (Tensor t in _midResnet1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _midAttention.EnumerateWeights()) yield return t;
        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
        if (_convOutWeight is not null) yield return _convOutWeight;
        if (_convOutBias is not null) yield return _convOutBias;
    }

    /// <summary>Encodes RGB <c>[1, 3, H, W]</c> in [-1, 1] to the scaled latent <c>[1, 64, H/32, W/32]</c>
    /// (posterior mean × scaling factor — the same contract as <see cref="VaeEncoder.Encode"/>).</summary>
    public Tensor Encode(IBackend backend, Tensor image)
    {
        ThrowIfDisposed();
        if (image.Shape.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3)
            throw new ArgumentException($"Image must be [1, 3, H, W]; got {image.Shape}.", nameof(image));
        int h = (int)image.Shape[2], w = (int)image.Shape[3];

        Tensor x = new Tensor(new TensorShape(1, ChannelPlan[0], h, w), DType.F32);
        backend.Conv2D(x, image, _convInWeight!, _convInBias, 1, 1, 1, 1);

        int curH = h, curW = w;
        for (int lvl = 0; lvl < ChannelPlan.Length; lvl++)
        {
            for (int i = 0; i < ResnetsPerLevel; i++)
                x = ForwardInto(backend, x, _levelResnets[lvl][i]);
            if (lvl < ChannelPlan.Length - 1)
            {
                Tensor down = Downsample(backend, x, ChannelPlan[lvl], ChannelPlan[lvl + 1], curH, curW, lvl);
                x.Dispose(); x = down;
                curH /= 2; curW /= 2;
            }
        }

        x = ForwardInto(backend, x, _midResnet0);
        Tensor attnOut = _midAttention.Forward(backend, x);
        x.Dispose(); x = attnOut;
        x = ForwardInto(backend, x, _midResnet1);

        // conv_out with the mirror of the decoder's conv_in skip: moments = conv(norm_silu(x)) + avg(x → 2·zc).
        int momentCh = (int)_convOutWeight!.Shape[0];
        Tensor normed = new Tensor(x.Shape, DType.F32);
        backend.GroupNormSilu(normed, x, _normOutWeight!, _normOutBias!, NormGroups, NormEps);
        Tensor moments = new Tensor(new TensorShape(1, momentCh, curH, curW), DType.F32);
        backend.Conv2D(moments, normed, _convOutWeight!, _convOutBias, 1, 1, 1, 1);
        normed.Dispose();
        Tensor avg = AvgChannelGroups(x, momentCh);
        x.Dispose();
        Tensor momentsSum = new Tensor(moments.Shape, DType.F32);
        backend.Add(momentsSum, moments, avg);
        moments.Dispose(); avg.Dispose();

        // Posterior mean = first latentCh moment channels (moments are contiguous, so this is a flat copy).
        int zc = _config.LatentChannels;
        Tensor mu = new Tensor(new TensorShape(1, zc, curH, curW), DType.F32);
        unsafe
        {
            long count = (long)zc * curH * curW;
            Buffer.MemoryCopy((float*)momentsSum.DataPointer, (float*)mu.DataPointer, count * sizeof(float), count * sizeof(float));
        }
        momentsSum.Dispose();

        Tensor scaled = new Tensor(mu.Shape, DType.F32);
        float shift = _config.ShiftFactor ?? 0f;
        if (shift != 0f)
        {
            Tensor shifted = new Tensor(mu.Shape, DType.F32);
            backend.AddScalar(shifted, mu, -shift);
            backend.Scale(scaled, shifted, _config.ScalingFactor);
            shifted.Dispose();
        }
        else
        {
            backend.Scale(scaled, mu, _config.ScalingFactor);
        }
        mu.Dispose();
        return scaled;
    }

    private static Tensor ForwardInto(IBackend backend, Tensor x, ResNetBlock2D block)
    {
        Tensor next = block.Forward(backend, x);
        if (!ReferenceEquals(next, x)) x.Dispose();
        return next;
    }

    /// <summary>One residual pixel-shuffle downsample <c>[1, C, H, W] → [1, outC, H/2, W/2]</c>:
    /// <c>s2d(conv(x)) + avg(s2d(x))</c> — the exact inverse composition of the decoder's
    /// <c>shuffle(conv(x)) + shuffle(repeat(x))</c>.</summary>
    private Tensor Downsample(IBackend backend, Tensor x, int inC, int outC, int h, int w, int lvl)
    {
        int convOutC = (int)_downsampleWeights[lvl]!.Shape[0];   // outC / 4
        Tensor conv = new Tensor(new TensorShape(1, convOutC, h, w), DType.F32);
        backend.Conv2D(conv, x, _downsampleWeights[lvl]!, _downsampleBiases[lvl], 1, 1, 1, 1);
        Tensor main = UnshuffleRearrange(backend, conv, convOutC, h / 2, w / 2);
        conv.Dispose();

        Tensor xPacked = UnshuffleRearrange(backend, x, inC, h / 2, w / 2);
        Tensor shortcut = AvgChannelGroups(xPacked, outC);
        xPacked.Dispose();

        Tensor sum = new Tensor(main.Shape, DType.F32);
        backend.Add(sum, main, shortcut);
        main.Dispose(); shortcut.Dispose();
        return sum;
    }

    /// <summary>Space-to-depth in the decoder's [r1, r2, c] channel order — the exact inverse of
    /// <c>HunyuanImageVaeDecoder.ShuffleRearrange</c>, composed from the two inverse <see cref="IBackend.Permute0213"/>
    /// passes in reverse order: <c>[1, C, 2h, 2w] → [1, 4C, h, w]</c> with
    /// <c>out(r1·2C + r2·C + c, y, x) = in(c, 2y+r1, 2x+r2)</c>.</summary>
    private static Tensor UnshuffleRearrange(IBackend backend, Tensor input, int c, int h, int w)
    {
        // Inverse of decoder pass 2: view input as [B=1, S=c·h, H=2, D=2w] → cat [2, c, h, 2w].
        Tensor cat = new Tensor(new TensorShape(2, c, h, 2 * w), DType.F32);
        backend.Permute0213(cat, input, c * h, 2, 2 * w);
        // Inverse of decoder pass 1: view cat as [B=2, S=c·h·w, H=2, D=1] → packed [1, 4C, h, w].
        Tensor packed = new Tensor(new TensorShape(1, 4 * c, h, w), DType.F32);
        backend.Permute0213(packed, cat, c * h * w, 2, 1);
        cat.Dispose();
        return packed;
    }

    /// <summary>Channel-group average — the inverse of the decoder's <c>repeat_interleave</c> skip:
    /// <c>[1, C, H, W] → [1, outC, H, W]</c> with <c>out(o, ·) = mean over g of in(o·g + i, ·)</c>,
    /// <c>g = C / outC</c>. Host op (one D2H sync); encode runs once per generation so this is off the hot path.</summary>
    private static unsafe Tensor AvgChannelGroups(Tensor x, int outC)
    {
        int c = (int)x.Shape[1];
        int h = (int)x.Shape[2], w = (int)x.Shape[3];
        int g = c / outC;
        if (g * outC != c)
            throw new ArgumentException($"Channel count {c} is not a multiple of target {outC}.");
        long hw = (long)h * w;
        Tensor outT = new Tensor(new TensorShape(1, outC, h, w), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* dp = (float*)outT.DataPointer;
        float inv = 1f / g;
        for (int o = 0; o < outC; o++)
        {
            long dstBase = o * hw;
            for (long s = 0; s < hw; s++)
            {
                float sum = 0f;
                for (int i = 0; i < g; i++)
                    sum += sp[((long)(o * g + i)) * hw + s];
                dp[dstBase + s] = sum * inv;
            }
        }
        return outT;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _convInWeight = null; _convInBias = null;
        _normOutWeight = null; _normOutBias = null;
        _convOutWeight = null; _convOutBias = null;
    }
}
