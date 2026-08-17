using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Music;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>LTX-2 audio VAE decoder (<c>AudioDecoder</c> in <c>AutoencoderKLLTX2Audio</c>), ported from diffusers.
/// A 2-D causal-conv VAE that decodes the 8-channel audio latent <c>[B, 8, T_lat, mel_lat]</c> into a stereo
/// log-mel spectrogram <c>[B, 2, T, mel]</c> (the vocoder then turns mel → waveform). The "image" axes are
/// (height = time, width = mel); causality is on the TIME axis (<c>causality_axis="height"</c>).
///
/// <para>Structure: <c>conv_in</c>(8→512) → mid (2 resnets @512, no attention) → 3 up-levels iterated L2→L1→L0
/// (L2: 3 resnets@512 + upsample; L1: 512→256 + upsample; L0: 256→128, no upsample) → parameter-free pixel
/// <c>norm_out</c> → SiLU → <c>conv_out</c>(128→2). All norms are <see cref="LtxAudioPixelNorm"/> (per-pixel RMS
/// over channels, eps 1e-6); all convs are <see cref="LtxAudioCausalConv2d"/> (time-causal: top-pad the time
/// axis since the backend <see cref="IBackend.Conv2D"/> pads symmetrically). Upsample =
/// nearest-×2 → causal conv → drop the first time row.</para>
///
/// <para>Per-channel latent un-normalization is NOT done here — the stats (<c>per_channel_statistics</c>) are
/// <c>[128]</c> over the PACKED 8×16 latent, so the pipeline denormalizes before unpacking to <c>[B,8,T,16]</c>.
/// Keys (under the <c>audio_vae.</c> prefix the loader strips): <c>decoder.conv_in.conv.*</c>,
/// <c>decoder.mid.block_{1,2}.*</c>, <c>decoder.up.{i}.block.{j}.*</c> (+ <c>nin_shortcut</c> on channel change),
/// <c>decoder.up.{i}.upsample.conv.conv.*</c>, <c>decoder.conv_out.conv.*</c>. Verified vs the LTX-2.3 header.
/// Numerics vs the real checkpoint are validation-pending.</para></summary>
public sealed class LtxAudioVaeDecoder
{
    private const float PixelNormEps = 1e-6f;

    private readonly int _latentChannels;
    private readonly int _baseChannels;
    private readonly int[] _chMult;
    private readonly int _numResBlocks;

    private LtxAudioCausalConv2d? _convIn, _convOut;
    private LtxAudioResnetBlock[] _midResnets = [];
    private UpLevel[] _upLevels = [];   // indexed by resolution level (0..n-1); forward iterates high→low

    private sealed class UpLevel
    {
        public LtxAudioResnetBlock[] Resnets = [];
        public LtxAudioUpsample? Upsample;   // null on the lowest level
    }

    public LtxAudioVaeDecoder(
        int latentChannels = 8, int baseChannels = 128, int[]? chMult = null, int numResBlocks = 2)
    {
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _chMult = chMult ?? [1, 2, 4];
        _numResBlocks = numResBlocks;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int numResolutions = _chMult.Length;
        int baseBlockChannels = _baseChannels * _chMult[^1];   // 128*4 = 512

        _convIn = new LtxAudioCausalConv2d(w["decoder.conv_in.conv.weight"], Bias(w, "decoder.conv_in.conv.bias"));

        _midResnets = new LtxAudioResnetBlock[2];
        _midResnets[0] = new LtxAudioResnetBlock(baseBlockChannels, baseBlockChannels);
        _midResnets[0].LoadWeights(w, "decoder.mid.block_1");
        _midResnets[1] = new LtxAudioResnetBlock(baseBlockChannels, baseBlockChannels);
        _midResnets[1].LoadWeights(w, "decoder.mid.block_2");

        // Build up-levels in diffusers order (reversed resolutions, threading block_in), storing at level index.
        _upLevels = new UpLevel[numResolutions];
        int blockIn = baseBlockChannels;
        for (int level = numResolutions - 1; level >= 0; level--)
        {
            int blockOut = _baseChannels * _chMult[level];
            UpLevel stage = new() { Resnets = new LtxAudioResnetBlock[_numResBlocks + 1] };
            for (int j = 0; j < stage.Resnets.Length; j++)
            {
                stage.Resnets[j] = new LtxAudioResnetBlock(blockIn, blockOut);
                stage.Resnets[j].LoadWeights(w, $"decoder.up.{level}.block.{j}");
                blockIn = blockOut;
            }
            if (level != 0)
            {
                stage.Upsample = new LtxAudioUpsample();
                stage.Upsample.LoadWeights(w, $"decoder.up.{level}.upsample");
            }
            _upLevels[level] = stage;
        }

        _convOut = new LtxAudioCausalConv2d(w["decoder.conv_out.conv.weight"], Bias(w, "decoder.conv_out.conv.bias"));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convIn is not null) foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        foreach (LtxAudioResnetBlock r in _midResnets) foreach (Tensor t in r.EnumerateWeights()) yield return t;
        foreach (UpLevel s in _upLevels)
        {
            foreach (LtxAudioResnetBlock r in s.Resnets) foreach (Tensor t in r.EnumerateWeights()) yield return t;
            if (s.Upsample is not null) foreach (Tensor t in s.Upsample.EnumerateWeights()) yield return t;
        }
        if (_convOut is not null) foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
    }

    /// <summary>Decodes an audio latent <c>[B, 8, T_lat, mel_lat]</c> to a stereo log-mel spectrogram
    /// <c>[B, 2, T, mel]</c>. (Latent denormalization is the pipeline's job — see class remarks.)</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if ((int)latent.Shape[1] != _latentChannels)
            throw new ArgumentException($"audio latent channels {latent.Shape[1]} != {_latentChannels}.", nameof(latent));

        Tensor h = _convIn!.Forward(backend, latent);
        foreach (LtxAudioResnetBlock r in _midResnets) { Tensor n = r.Forward(backend, h); h.Dispose(); h = n; }

        for (int level = _upLevels.Length - 1; level >= 0; level--)
        {
            UpLevel s = _upLevels[level];
            foreach (LtxAudioResnetBlock r in s.Resnets) { Tensor n = r.Forward(backend, h); h.Dispose(); h = n; }
            if (s.Upsample is not null) { Tensor n = s.Upsample.Forward(backend, h); h.Dispose(); h = n; }
        }

        Tensor normed = LtxAudioPixelNorm.Forward(backend, h, PixelNormEps);
        h.Dispose();
        backend.Silu(normed, normed);
        Tensor outT = _convOut!.Forward(backend, normed);   // [B, 2, T, mel]
        normed.Dispose();
        return outT;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? t : null;
}

/// <summary>Time-causal 2-D conv for the LTX-2 audio VAE (<c>LTX2AudioCausalConv2d</c>, causality_axis="height").
/// The backend's <see cref="IBackend.Conv2D"/> pads symmetrically, so the causal TIME axis (height) is padded
/// manually with <c>k-1</c> zero rows on top (none on the bottom); the mel axis (width) uses the symmetric
/// <c>Conv2D</c> padding. Size-preserving for stride 1.</summary>
public sealed class LtxAudioCausalConv2d
{
    private readonly Tensor _weight;   // [outC, inC, kh, kw]
    private readonly Tensor? _bias;
    private readonly int _kh, _kw, _outC;

    public LtxAudioCausalConv2d(Tensor weight, Tensor? bias)
    {
        _weight = weight;
        _bias = bias;
        _outC = (int)weight.Shape[0];
        _kh = (int)weight.Shape[2];
        _kw = (int)weight.Shape[3];
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0], h = (int)x.Shape[2], w = (int)x.Shape[3];
        int padTop = _kh - 1;          // causal: all height padding on top
        int padW = (_kw - 1) / 2;      // symmetric mel padding, handled by Conv2D

        Tensor src = x;
        if (padTop > 0) src = LtxAudioDeviceOps.PadTimeTopZero(backend, x, padTop);

        int outH = h;   // (h + padTop) - (kh - 1) = h
        int outW = w;   // w + 2*padW - (kw - 1) = w for odd kw
        Tensor outT = new Tensor(new TensorShape(b, _outC, outH, outW), DType.F32);
        backend.Conv2D(outT, src, _weight, _bias, strideH: 1, strideW: 1, padH: 0, padW: padW);
        if (!ReferenceEquals(src, x)) src.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _weight;
        if (_bias is not null) yield return _bias;
    }
}

/// <summary>Parameter-free per-pixel RMS normalization over the channel dim (<c>LTX2AudioPixelNorm</c>):
/// <c>x / sqrt(mean_c(x²) + eps)</c>. Routed through the device <see cref="IBackend.WanRmsNormChannel"/> op
/// (<c>x·sqrt(C)/max(L2_C, eps′)</c>) with the eps folded as <c>eps′ = sqrt(C·eps)</c>, which matches the reference
/// exactly in both the signal limit (<c>L2 ≫ eps′</c>, difference ≤ eps/(2·mean_C(x²))) and the silence limit
/// (both reduce to <c>x/sqrt(eps)</c>).</summary>
public static class LtxAudioPixelNorm
{
    public static Tensor Forward(IBackend backend, Tensor x, float eps)
    {
        int c = (int)x.Shape[1];
        Tensor outT = new Tensor(x.Shape, DType.F32);
        backend.WanRmsNormChannel(outT, x, null, MathF.Sqrt(c * eps));
        return outT;
    }
}

/// <summary>LTX-2 audio VAE resnet block (<c>LTX2AudioResnetBlock</c>, pixel-norm variant):
/// pixelNorm → SiLU → conv1 → pixelNorm → SiLU → conv2 → + shortcut. The shortcut is a 1×1 causal conv
/// (<c>nin_shortcut</c>) when in≠out, else identity.</summary>
public sealed class LtxAudioResnetBlock
{
    private readonly int _inC, _outC;
    private LtxAudioCausalConv2d? _conv1, _conv2, _ninShortcut;

    public LtxAudioResnetBlock(int inChannels, int outChannels)
    {
        _inC = inChannels;
        _outC = outChannels;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _conv1 = new LtxAudioCausalConv2d(w[$"{prefix}.conv1.conv.weight"], Bias(w, $"{prefix}.conv1.conv.bias"));
        _conv2 = new LtxAudioCausalConv2d(w[$"{prefix}.conv2.conv.weight"], Bias(w, $"{prefix}.conv2.conv.bias"));
        if (_inC != _outC)
        {
            _ninShortcut = new LtxAudioCausalConv2d(w[$"{prefix}.nin_shortcut.conv.weight"], Bias(w, $"{prefix}.nin_shortcut.conv.bias"));
        }
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor h = LtxAudioPixelNorm.Forward(backend, x, 1e-6f);
        backend.Silu(h, h);
        Tensor c1 = _conv1!.Forward(backend, h);
        h.Dispose();
        Tensor n2 = LtxAudioPixelNorm.Forward(backend, c1, 1e-6f);
        c1.Dispose();
        backend.Silu(n2, n2);
        Tensor c2 = _conv2!.Forward(backend, n2);
        n2.Dispose();

        Tensor shortcut = _ninShortcut is not null ? _ninShortcut.Forward(backend, x) : x;
        Tensor outT = new Tensor(c2.Shape, DType.F32);
        backend.Add(outT, c2, shortcut);
        c2.Dispose();
        if (_ninShortcut is not null) shortcut.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_conv1 is not null) foreach (Tensor t in _conv1.EnumerateWeights()) yield return t;
        if (_conv2 is not null) foreach (Tensor t in _conv2.EnumerateWeights()) yield return t;
        if (_ninShortcut is not null) foreach (Tensor t in _ninShortcut.EnumerateWeights()) yield return t;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? t : null;
}

/// <summary>LTX-2 audio VAE upsample (<c>LTX2AudioUpsample</c>, causality_axis="height"): nearest-×2 on both
/// axes → time-causal conv → drop the first (time) row to keep the temporal causal alignment.</summary>
public sealed class LtxAudioUpsample
{
    private LtxAudioCausalConv2d? _conv;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _conv = new LtxAudioCausalConv2d(w[$"{prefix}.conv.conv.weight"], Bias(w, $"{prefix}.conv.conv.bias"));
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], h = (int)x.Shape[2], w = (int)x.Shape[3];
        Tensor up = new Tensor(new TensorShape(b, c, h * 2, w * 2), DType.F32);
        backend.UpsampleNearest2D(up, x, 2, 2);
        Tensor conv = _conv!.Forward(backend, up);     // [b, c, 2h, 2w] (size-preserving causal conv)
        up.Dispose();

        // Drop the first time row: [:, :, 1:, :] → [b, c, 2h-1, 2w].
        Tensor outT = LtxAudioDeviceOps.DropFirstTimeRow(backend, conv);
        conv.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_conv is not null) foreach (Tensor t in _conv.EnumerateWeights()) yield return t;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? t : null;
}
