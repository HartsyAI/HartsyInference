using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>HunyuanVideo-1.5 causal 3D VAE (MovieRoll's <c>AutoencoderKLConv3D</c>, non-streaming numerics): 16× spatial / 4× temporal, 32 latent
/// channels, RMS norm, DCAE shuffle resamplers. Encoder: <c>[1, 3, F, H, W] → [1, 2·Z, T', H/16, W/16]</c> (mean then logvar). Decoder:
/// <c>[1, Z, T', H', W'] → [1, 3, F, 16·H', 16·W']</c> with <c>F = 4·(T'−1) + 1</c>.
///
/// <para>The scale is not applied here: MovieRoll divides a latent by <see cref="HunyuanVideo15VaeConfig.ScalingFactor"/> before decoding, and
/// the caller does the same. Batch 1 only. The chunked, cache-carrying streaming decode is a separate increment: this class is the full-clip path.</para>
/// </summary>
public sealed class HunyuanVideo15Vae : IDisposable
{
    private readonly HunyuanVideo15VaeConfig _config;
    private bool _loaded;
    private bool _disposed;
    private List<(string Name, IVaeModule Module)> _modules = [];

    private CausalConv3d? _encConvIn, _encConvOut;
    private WanRmsNorm? _encNormOut;
    private HunyuanVideo15VaeResnet? _encMid1, _encMid2;
    private HunyuanVideo15VaeMidAttention? _encAttn;
    private EncoderLevel[] _encLevels = [];

    private CausalConv3d? _decConvIn, _decConvOut;
    private WanRmsNorm? _decNormOut;
    private HunyuanVideo15VaeResnet? _decMid1, _decMid2;
    private HunyuanVideo15VaeMidAttention? _decAttn;
    private DecoderLevel[] _decLevels = [];

    private sealed class EncoderLevel
    {
        public required HunyuanVideo15VaeResnet[] Resnets;
        public HunyuanVideo15VaeDownsample? Downsample;
    }

    private sealed class DecoderLevel
    {
        public required HunyuanVideo15VaeResnet[] Resnets;
        public HunyuanVideo15VaeUpsample? Upsample;
    }

    /// <summary>Creates an empty VAE for <paramref name="config"/> (default: the published configuration). Call <see cref="LoadWeights"/> before use.</summary>
    public HunyuanVideo15Vae(HunyuanVideo15VaeConfig? config = null)
    {
        _config = config ?? HunyuanVideo15VaeConfig.Default;
        _config.Validate();
    }

    /// <summary>The configuration this VAE was built with.</summary>
    public HunyuanVideo15VaeConfig Config => _config;

    /// <summary>The weights <see cref="Decode"/> reads, live from the decoder's layers: a conv's weight form can change after its first call, so
    /// enumerate again rather than keeping the list. For device placement around a decode.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => EnumerateWith("decoder.");

    /// <summary>The weights the encoder reads, live from the encoder's layers. <see cref="Decode"/> does not read them.</summary>
    public IEnumerable<Tensor> EnumerateEncodeWeights() => EnumerateWith("encoder.");

    private IEnumerable<Tensor> EnumerateWith(string prefix)
    {
        foreach ((string name, IVaeModule module) in _modules)
        {
            if (!name.StartsWith(prefix, StringComparison.Ordinal)) continue;
            foreach (Tensor weight in module.EnumerateWeights()) yield return weight;
        }
    }

    /// <summary>Releases every layer's owned weight tensors. The VAE holds no other device memory after this.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach ((_, IVaeModule module) in _modules) module.Dispose();
        _modules.Clear();
        _loaded = false;
    }

    /// <summary>Loads the checkpoint under MovieRoll's state-dict names (<c>encoder.*</c>, <c>decoder.*</c>). Strict: a missing key throws, and so
    /// does any checkpoint tensor the layers do not read. Call once.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (_loaded) throw new InvalidOperationException("HunyuanVideo-1.5 VAE weights are already loaded.");
        HunyuanVideo15VaeWeights w = new(weights);
        float eps = _config.NormEps;
        int[] enc = _config.BlockOutChannels;
        int[] dec = Enumerable.Reverse(enc).ToArray();
        int layers = _config.LayersPerBlock;
        int levels = enc.Length;
        int z = _config.LatentChannels;
        RequireShape(weights, "encoder.conv_in.conv.weight", enc[0], _config.InChannels, 3, 3, 3);
        RequireShape(weights, "encoder.conv_out.conv.weight", 2 * z, enc[^1], 3, 3, 3);
        RequireShape(weights, "decoder.conv_in.conv.weight", dec[0], z, 3, 3, 3);
        RequireShape(weights, "decoder.conv_out.conv.weight", _config.OutChannels, dec[^1], 3, 3, 3);

        _encConvIn = w.Conv("encoder.conv_in.conv");
        _encLevels = new EncoderLevel[levels];
        int blockIn = enc[0];
        for (int i = 0; i < levels; i++)
        {
            HunyuanVideo15VaeResnet[] resnets = new HunyuanVideo15VaeResnet[layers];
            for (int j = 0; j < layers; j++)
            {
                resnets[j] = new HunyuanVideo15VaeResnet(w, $"encoder.down.{i}.block.{j}", blockIn, enc[i], eps);
                blockIn = enc[i];
            }
            HunyuanVideo15VaeDownsample? down = null;
            if (_config.EncoderSpatialDownsample(i) || _config.EncoderTemporalDownsample(i))
            {
                if (i >= levels - 1) throw new ArgumentException($"Encoder level {i} downsamples but is the last level.");
                down = new HunyuanVideo15VaeDownsample(w, $"encoder.down.{i}.downsample", enc[i + 1], _config.EncoderTemporalDownsample(i));
                blockIn = enc[i + 1];
            }
            _encLevels[i] = new EncoderLevel { Resnets = resnets, Downsample = down };
        }
        _encMid1 = new HunyuanVideo15VaeResnet(w, "encoder.mid.block_1", blockIn, blockIn, eps);
        _encAttn = new HunyuanVideo15VaeMidAttention(w, "encoder.mid.attn_1", eps);
        _encMid2 = new HunyuanVideo15VaeResnet(w, "encoder.mid.block_2", blockIn, blockIn, eps);
        _encNormOut = w.Norm("encoder.norm_out", eps);
        _encConvOut = w.Conv("encoder.conv_out.conv");

        _decConvIn = w.Conv("decoder.conv_in.conv");
        int decIn = dec[0];
        _decMid1 = new HunyuanVideo15VaeResnet(w, "decoder.mid.block_1", decIn, decIn, eps);
        _decAttn = new HunyuanVideo15VaeMidAttention(w, "decoder.mid.attn_1", eps);
        _decMid2 = new HunyuanVideo15VaeResnet(w, "decoder.mid.block_2", decIn, decIn, eps);
        _decLevels = new DecoderLevel[levels];
        int blockDec = dec[0];
        for (int i = 0; i < levels; i++)
        {
            HunyuanVideo15VaeResnet[] resnets = new HunyuanVideo15VaeResnet[layers + 1];
            for (int j = 0; j <= layers; j++)
            {
                resnets[j] = new HunyuanVideo15VaeResnet(w, $"decoder.up.{i}.block.{j}", blockDec, dec[i], eps);
                blockDec = dec[i];
            }
            HunyuanVideo15VaeUpsample? up = null;
            if (_config.DecoderSpatialUpsample(i) || _config.DecoderTemporalUpsample(i))
            {
                if (i >= levels - 1) throw new ArgumentException($"Decoder level {i} upsamples but is the last level.");
                up = new HunyuanVideo15VaeUpsample(w, $"decoder.up.{i}.upsample", dec[i + 1], _config.DecoderTemporalUpsample(i));
                blockDec = dec[i + 1];
            }
            _decLevels[i] = new DecoderLevel { Resnets = resnets, Upsample = up };
        }
        _decNormOut = w.Norm("decoder.norm_out", eps);
        _decConvOut = w.Conv("decoder.conv_out.conv");

        w.ThrowOnUnconsumed();
        _modules = w.Modules.ToList();
        _loaded = true;
    }

    /// <summary>Encodes <paramref name="video"/> (<c>[1, 3, F, H, W]</c>, F-frame clip with <c>F = 4n + 1</c>) and returns the moments
    /// <c>[1, 2·Z, T', H/16, W/16]</c>: mean in the first Z channels, logvar in the rest.</summary>
    public Tensor EncodeMoments(IBackend backend, Tensor video)
    {
        RequireLoaded();
        (int channels, int frames, int height, int width) = HunyuanVideo15VaeShuffle.Dims(video);
        int spatial = _config.SpatialCompression, temporal = _config.TemporalCompression;
        if (channels != _config.InChannels || frames < 1 || (frames - 1) % temporal != 0 || height % spatial != 0 || width % spatial != 0)
            throw new ArgumentException(
                $"HunyuanVideo-1.5 VAE encode needs a [1, {_config.InChannels}, {temporal}n + 1, H, W] clip with H and W multiples of {spatial}; got {video.Shape}.",
                nameof(video));
        int z = _config.LatentChannels;
        Tensor h = _encConvIn!.Forward(backend, video);
        foreach (EncoderLevel level in _encLevels)
        {
            foreach (HunyuanVideo15VaeResnet resnet in level.Resnets) h = Next(h, resnet.Forward(backend, h));
            if (level.Downsample is not null) h = Next(h, level.Downsample.Forward(backend, h));
        }
        h = Next(h, _encMid1!.Forward(backend, h));
        h = Next(h, _encAttn!.Forward(backend, h));
        h = Next(h, _encMid2!.Forward(backend, h));

        Tensor shortcut = HunyuanVideo15VaeShuffle.GroupMean(h, 2 * z);
        Tensor normed = HunyuanVideo15VaeResnet.Silu(backend, _encNormOut!.Forward(backend, h));
        h.Dispose();
        Tensor convOut = _encConvOut!.Forward(backend, normed);
        normed.Dispose();
        Tensor moments = new(convOut.Shape, DType.F32);
        backend.Add(moments, convOut, shortcut);
        convOut.Dispose();
        shortcut.Dispose();
        return moments;
    }

    /// <summary>The posterior mean of <see cref="EncodeMoments"/>: the first <c>Z</c> channels, a <c>[1, Z, T', H/16, W/16]</c> copy.</summary>
    public Tensor EncodeMean(IBackend backend, Tensor video)
    {
        using Tensor moments = EncodeMoments(backend, video);
        (int c, int t, int hh, int ww) = HunyuanVideo15VaeShuffle.Dims(moments);
        int z = _config.LatentChannels;
        Tensor mean = HunyuanVideo15VaeShuffle.Alloc(z, t, hh, ww);
        moments.AsReadOnlySpan<float>()[..(z * t * hh * ww)].CopyTo(mean.AsSpan<float>());
        return mean;
    }

    /// <summary>Decodes an unscaled latent <c>[1, Z, T', H', W']</c> (already divided by <see cref="HunyuanVideo15VaeConfig.ScalingFactor"/>)
    /// to <c>[1, 3, 4·(T'−1)+1, 16·H', 16·W']</c>.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        RequireLoaded();
        HunyuanVideo15VaeShuffle.Dims(latent);
        int c0 = _config.BlockOutChannels[^1];
        Tensor conv = _decConvIn!.Forward(backend, latent);
        Tensor repeat = HunyuanVideo15VaeShuffle.RepeatToChannels(backend, latent, c0);
        Tensor h = new(conv.Shape, DType.F32);
        backend.Add(h, conv, repeat);
        conv.Dispose();
        repeat.Dispose();

        h = Next(h, _decMid1!.Forward(backend, h));
        h = Next(h, _decAttn!.Forward(backend, h));
        h = Next(h, _decMid2!.Forward(backend, h));
        foreach (DecoderLevel level in _decLevels)
        {
            foreach (HunyuanVideo15VaeResnet resnet in level.Resnets) h = Next(h, resnet.Forward(backend, h));
            if (level.Upsample is not null) h = Next(h, level.Upsample.Forward(backend, h));
        }
        Tensor normed = HunyuanVideo15VaeResnet.Silu(backend, _decNormOut!.Forward(backend, h));
        h.Dispose();
        Tensor output = _decConvOut!.Forward(backend, normed);
        normed.Dispose();
        return output;
    }

    private static void RequireShape(IReadOnlyDictionary<string, Tensor> weights, string key, params int[] expected)
    {
        if (!weights.TryGetValue(key, out Tensor? tensor)) return;   // a missing key is reported by the loader, by name
        bool matches = tensor.Shape.Rank == expected.Length
            && Enumerable.Range(0, expected.Length).All(i => tensor.Shape[i] == expected[i]);
        if (!matches)
            throw new InvalidDataException($"HunyuanVideo-1.5 VAE tensor '{key}' has shape {tensor.Shape}; the configuration expects [{string.Join(", ", expected)}].");
    }

    private static Tensor Next(Tensor previous, Tensor next)
    {
        previous.Dispose();
        return next;
    }

    private void RequireLoaded()
    {
        if (!_loaded) throw new InvalidOperationException("HunyuanVideo-1.5 VAE weights are not loaded; call LoadWeights first.");
    }
}
