using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>HunyuanImage 2.1 pixel-shuffle VAE decoder (diffusers <c>HunyuanImageDecoder2D</c>). Differs structurally from the generic <see cref="VaeDecoder"/>: <c>conv_in</c> adds a channel-<c>repeat_interleave</c> skip of the latent, every resnet is channel-preserving, and each of the 5 upsamplers is <c>conv([4·C_next, C, 3, 3])</c> + a depth-to-space rearrange in <b>[r1, r2, c]</b> channel order (NOT torch PixelShuffle's <c>[c, r1, r2]</c>) summed with an identically-rearranged channel-repeat of the input. The rearrange is composed from device ops only (channel halves are contiguous, so column- and row-interleave are each one <see cref="IBackend.Permute0213"/>).</summary>
public sealed class HunyuanImageVaeDecoder : IDisposable
{
    // Decoder-order channel plan = reversed encoder block_out_channels [128, 256, 512, 512, 1024, 1024].
    private static readonly int[] ChannelPlan = [1024, 1024, 512, 512, 256, 128];
    private const int ResnetsPerLevel = 3;
    private const int NormGroups = 32;
    private const float NormEps = 1e-6f;

    private readonly VaeConfig _config;
    private Tensor? _convInWeight, _convInBias;
    private readonly ResNetBlock2D _midResnet0 = new(ChannelPlan[0], ChannelPlan[0]);
    private readonly ResNetBlock2D _midResnet1 = new(ChannelPlan[0], ChannelPlan[0]);
    private readonly VaeAttention _midAttention = new(ChannelPlan[0]);
    private readonly ResNetBlock2D[][] _levelResnets;
    private readonly Tensor?[] _upsampleWeights = new Tensor?[ChannelPlan.Length - 1];
    private readonly Tensor?[] _upsampleBiases = new Tensor?[ChannelPlan.Length - 1];
    private Tensor? _normOutWeight, _normOutBias, _convOutWeight, _convOutBias;
    private bool _disposed;

    public HunyuanImageVaeDecoder(VaeConfig config)
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

    /// <summary>Loads diffusers-named decoder weights (after the LDM remap with <c>reverseUpIndices: false</c> — the checkpoint stores <c>up.0</c> as the deepest level).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _convInWeight = weights["decoder.conv_in.weight"];
        _convInBias = weights["decoder.conv_in.bias"];
        _midResnet0.LoadWeights(weights, "decoder.mid_block.resnets.0");
        _midResnet1.LoadWeights(weights, "decoder.mid_block.resnets.1");
        _midAttention.LoadWeights(weights, "decoder.mid_block.attentions.0");
        for (int lvl = 0; lvl < ChannelPlan.Length; lvl++)
        {
            for (int i = 0; i < ResnetsPerLevel; i++)
                _levelResnets[lvl][i].LoadWeights(weights, $"decoder.up_blocks.{lvl}.resnets.{i}");
            if (lvl < ChannelPlan.Length - 1)
            {
                _upsampleWeights[lvl] = weights[$"decoder.up_blocks.{lvl}.upsamplers.0.conv.weight"];
                _upsampleBiases[lvl] = weights[$"decoder.up_blocks.{lvl}.upsamplers.0.conv.bias"];
            }
        }
        _normOutWeight = weights["decoder.conv_norm_out.weight"];
        _normOutBias = weights["decoder.conv_norm_out.bias"];
        _convOutWeight = weights["decoder.conv_out.weight"];
        _convOutBias = weights["decoder.conv_out.bias"];
    }

    /// <summary>Enumerates all weight tensors for GPU preload/free.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convInWeight is not null) yield return _convInWeight;
        if (_convInBias is not null) yield return _convInBias;
        foreach (Tensor t in _midResnet0.EnumerateWeights()) yield return t;
        foreach (Tensor t in _midResnet1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _midAttention.EnumerateWeights()) yield return t;
        foreach (ResNetBlock2D[] level in _levelResnets)
            foreach (ResNetBlock2D block in level)
                foreach (Tensor t in block.EnumerateWeights()) yield return t;
        for (int i = 0; i < _upsampleWeights.Length; i++)
        {
            if (_upsampleWeights[i] is not null) yield return _upsampleWeights[i]!;
            if (_upsampleBiases[i] is not null) yield return _upsampleBiases[i]!;
        }
        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
        if (_convOutWeight is not null) yield return _convOutWeight;
        if (_convOutBias is not null) yield return _convOutBias;
    }

    /// <summary>Decodes a latent <c>[1, 64, h, w]</c> to RGB <c>[1, 3, 32·h, 32·w]</c>. Applies the inverse scaling factor first.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        ThrowIfDisposed();
        int b = (int)latent.Shape[0], zc = (int)latent.Shape[1], h = (int)latent.Shape[2], w = (int)latent.Shape[3];
        if (b != 1)
            throw new NotSupportedException("HunyuanImageVaeDecoder decodes batch 1 (the interleave composition assumes B=1).");

        Tensor z = new Tensor(latent.Shape, DType.F32);
        backend.Scale(z, latent, 1.0f / _config.ScalingFactor);

        // conv_in + repeat_interleave skip: h = conv_in(z) + repeat(z, 1024/64).
        int c0 = ChannelPlan[0];
        Tensor conv = new Tensor(new TensorShape(1, c0, h, w), DType.F32);
        backend.Conv2D(conv, z, _convInWeight!, _convInBias, 1, 1, 1, 1);
        Tensor rep = RepeatChannels(backend, z, zc, h * w, c0 / zc);
        Tensor x = new Tensor(new TensorShape(1, c0, h, w), DType.F32);
        backend.Add(x, conv, rep);
        conv.Dispose(); rep.Dispose(); z.Dispose();

        x = ForwardInto(backend, x, _midResnet0);
        Tensor attnOut = _midAttention.Forward(backend, x);
        x.Dispose(); x = attnOut;
        x = ForwardInto(backend, x, _midResnet1);

        int curH = h, curW = w;
        for (int lvl = 0; lvl < ChannelPlan.Length; lvl++)
        {
            for (int i = 0; i < ResnetsPerLevel; i++)
                x = ForwardInto(backend, x, _levelResnets[lvl][i]);
            if (lvl < ChannelPlan.Length - 1)
            {
                Tensor up = Upsample(backend, x, ChannelPlan[lvl], ChannelPlan[lvl + 1], curH, curW, lvl);
                x.Dispose(); x = up;
                curH *= 2; curW *= 2;
            }
        }

        int cLast = ChannelPlan[^1];
        Tensor normed = new Tensor(x.Shape, DType.F32);
        backend.GroupNormSilu(normed, x, _normOutWeight!, _normOutBias!, NormGroups, NormEps);
        x.Dispose();
        Tensor rgb = new Tensor(new TensorShape(1, 3, curH, curW), DType.F32);
        backend.Conv2D(rgb, normed, _convOutWeight!, _convOutBias, 1, 1, 1, 1);
        normed.Dispose();
        return rgb;
    }

    private static Tensor ForwardInto(IBackend backend, Tensor x, ResNetBlock2D block)
    {
        Tensor next = block.Forward(backend, x);
        if (!ReferenceEquals(next, x)) x.Dispose();
        return next;
    }

    /// <summary>One pixel-shuffle upsample: <c>shuffle(conv(x)) + shuffle(repeat_interleave(x))</c>, <c>[1, C, H, W] → [1, outC, 2H, 2W]</c>.</summary>
    private Tensor Upsample(IBackend backend, Tensor x, int inC, int outC, int h, int w, int lvl)
    {
        Tensor conv = new Tensor(new TensorShape(1, 4 * outC, h, w), DType.F32);
        backend.Conv2D(conv, x, _upsampleWeights[lvl]!, _upsampleBiases[lvl], 1, 1, 1, 1);
        Tensor main = ShuffleRearrange(backend, conv, outC, h, w);
        conv.Dispose();

        Tensor rep = RepeatChannels(backend, x, inC, h * w, 4 * outC / inC);
        Tensor shortcut = ShuffleRearrange(backend, rep, outC, h, w);
        rep.Dispose();

        Tensor sum = new Tensor(main.Shape, DType.F32);
        backend.Add(sum, main, shortcut);
        main.Dispose(); shortcut.Dispose();
        return sum;
    }

    /// <summary>Depth-to-space in [r1, r2, c] channel order: <c>[1, 4C, H, W] → [1, C, 2H, 2W]</c> with <c>out(c, 2h+r1, 2w+r2) = in(r1·2C + r2·C + c, h, w)</c>. Channel halves are contiguous, so each interleave is one flat Permute0213 (B=1): columns via <c>[2, C·H·W, 1] → [C·H·W, 2]</c> on each half, rows via <c>[2, C·H, 2W] → [C·H, 2, 2W]</c> on the concatenated halves.</summary>
    private static Tensor ShuffleRearrange(IBackend backend, Tensor packed, int c, int h, int w)
    {
        // Pass 1 views packed as [B=2, S=2, H=c·h·w, D=1] (batch inferred from element count): each
        // contiguous half interleaves its two channel-quarters element-wise → [r0-rows | r1-rows] flat.
        Tensor cat = new Tensor(new TensorShape(2, c, h, 2 * w), DType.F32);
        backend.Permute0213(cat, packed, 2, c * h * w, 1);
        // Pass 2 views that as [B=1, S=2, H=c·h, D=2w]: interleaves the two row-sets per (c, h).
        Tensor outT = new Tensor(new TensorShape(1, c, 2 * h, 2 * w), DType.F32);
        backend.Permute0213(outT, cat, 2, c * h, 2 * w);
        cat.Dispose();
        return outT;
    }

    /// <summary>Channel <c>repeat_interleave</c>: <c>[1, C, H·W] → [1, C·repeats, H·W]</c> via a nearest-neighbor row upsample on a <c>[1, 1, C, H·W]</c> view.</summary>
    private static Tensor RepeatChannels(IBackend backend, Tensor x, int c, int hw, int repeats)
    {
        Tensor flat = new Tensor(new TensorShape(1, 1, c, hw), DType.F32);
        unsafe { backend.CopyInto(flat, x); }
        Tensor outT = new Tensor(new TensorShape(1, c * repeats, (long)x.Shape[2], (long)x.Shape[3]), DType.F32);
        backend.UpsampleNearest2D(outT, flat, repeats, 1);
        flat.Dispose();
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
