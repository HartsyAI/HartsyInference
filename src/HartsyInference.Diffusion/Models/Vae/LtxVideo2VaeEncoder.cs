using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>LTX-2 causal video VAE encoder (<c>LTX2VideoEncoder3d</c> in diffusers <c>AutoencoderKLLTX2Video</c>), the
/// mirror of <see cref="LtxVideo2VaeDecoder"/>. RGB <c>[B, 3, T, H, W]</c> in [-1, 1] → pixel unshuffle (patch p) →
/// <c>conv_in</c> → down blocks (resnets at the incoming width, then <see cref="LtxVaeDownsampler3d"/>) → mid resnets
/// → parameter-free channel-RMS <c>norm_out</c> → SiLU → <c>conv_out</c> → posterior mode. Implements the LTX-2 VAE
/// encoder architecture.
///
/// <para>Encodes the whole clip in one pass. diffusers' framewise and tiled encode are off by default
/// (<c>use_framewise_encoding=False</c>, not set in the VAE config), so they are not implemented. Frames must be
/// <c>1 + TemporalCompression·n</c>, which makes a single image (<c>T = 1</c>) valid. <see cref="Encode"/> returns the
/// un-normalized mode; the caller applies <see cref="LtxVideo2VaeDecoder.NormalizeLatent"/>.</para></summary>
public sealed class LtxVideo2VaeEncoder
{
    private const float NormEps = 1e-8f;

    private sealed class DownStage
    {
        public LtxVaeResnetBlock3d[] Resnets = [];
        public LtxVaeDownsampler3d? Downsampler;
    }

    private readonly LtxVideo2VaeEncoderConfig _config;
    private readonly int _temporalCompression;
    private readonly int _spatialCompression;
    private readonly DownStage[] _downStages;
    private readonly LtxVaeResnetBlock3d[] _midResnets;
    private CausalConv3d? _convIn, _convOut;

    /// <summary>Builds the encoder structure from <paramref name="config"/>. Call <see cref="LoadWeights"/> before <see cref="Encode"/>.</summary>
    public LtxVideo2VaeEncoder(LtxVideo2VaeEncoderConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        ValidateConfig(config);
        _config = config;

        int blocks = config.BlockOutChannels.Length;
        _downStages = new DownStage[blocks];
        int temporal = 1, spatial = config.PatchSize;
        int width = config.LatentChannels;
        for (int i = 0; i < blocks; i++)
        {
            int outWidth = config.BlockOutChannels[i];
            DownStage stage = new();
            stage.Resnets = new LtxVaeResnetBlock3d[config.LayersPerBlock[i]];
            for (int j = 0; j < stage.Resnets.Length; j++)
            {
                stage.Resnets[j] = new LtxVaeResnetBlock3d(width, width, timestepCond: false, isCausal: config.IsCausal);
            }
            if (config.SpatioTemporalScaling[i])
            {
                (int st0, int st1, int st2) = StrideFor(config.DownsampleType[i]);
                stage.Downsampler = new LtxVaeDownsampler3d(width, outWidth, (st0, st1, st2), config.IsCausal);
                temporal *= st0;
                spatial *= st1;
            }
            _downStages[i] = stage;
            width = outWidth;
        }
        _temporalCompression = temporal;
        _spatialCompression = spatial;

        _midResnets = new LtxVaeResnetBlock3d[config.LayersPerBlock[^1]];
        for (int j = 0; j < _midResnets.Length; j++)
        {
            _midResnets[j] = new LtxVaeResnetBlock3d(width, width, timestepCond: false, isCausal: config.IsCausal);
        }
    }

    /// <summary>Frame count granularity: input frames are <c>1 + TemporalCompression·n</c>.</summary>
    public int TemporalCompression => _temporalCompression;

    /// <summary>Pixel granularity of H and W: input sides must be a multiple of this.</summary>
    public int SpatialCompression => _spatialCompression;

    /// <summary>Every weight key the encoder reads, mapped to its shape, derived from the config alone. Key names are the
    /// diffusers <c>AutoencoderKLLTX2Video</c> names the converter passes through unchanged.</summary>
    public static IReadOnlyDictionary<string, TensorShape> ExpectedWeightShapes(LtxVideo2VaeEncoderConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        ValidateConfig(config);
        Dictionary<string, TensorShape> shapes = new(StringComparer.Ordinal);
        AddConv(shapes, "encoder.conv_in.conv", config.LatentChannels, config.InChannels * config.PatchSize * config.PatchSize);
        int width = config.LatentChannels;
        for (int i = 0; i < config.BlockOutChannels.Length; i++)
        {
            string p = $"encoder.down_blocks.{i}";
            for (int j = 0; j < config.LayersPerBlock[i]; j++) AddResnet(shapes, $"{p}.resnets.{j}", width);
            int outWidth = config.BlockOutChannels[i];
            if (config.SpatioTemporalScaling[i])
            {
                (int st0, int st1, int st2) = StrideFor(config.DownsampleType[i]);
                AddConv(shapes, $"{p}.downsamplers.0.conv.conv", outWidth / (st0 * st1 * st2), width);
            }
            width = outWidth;
        }
        for (int j = 0; j < config.LayersPerBlock[^1]; j++) AddResnet(shapes, $"encoder.mid_block.resnets.{j}", width);
        AddConv(shapes, "encoder.conv_out.conv", config.LatentChannels + 1, width);
        return shapes;
    }

    /// <summary>Loads every <see cref="ExpectedWeightShapes"/> entry from <paramref name="w"/>, checking each shape. Extra
    /// keys (for example <c>latents_mean</c>) are ignored; the caller owns the normalization stats.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        ArgumentNullException.ThrowIfNull(w);
        ValidateWeights(w);

        _convIn = new CausalConv3d(w["encoder.conv_in.conv.weight"], VaeOps.Bias(w, "encoder.conv_in.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _config.IsCausal);
        for (int i = 0; i < _downStages.Length; i++)
        {
            DownStage stage = _downStages[i];
            string p = $"encoder.down_blocks.{i}";
            for (int j = 0; j < stage.Resnets.Length; j++) stage.Resnets[j].LoadWeights(w, $"{p}.resnets.{j}");
            stage.Downsampler?.LoadWeights(w, $"{p}.downsamplers.0");
        }
        for (int j = 0; j < _midResnets.Length; j++) _midResnets[j].LoadWeights(w, $"encoder.mid_block.resnets.{j}");
        _convOut = new CausalConv3d(w["encoder.conv_out.conv.weight"], VaeOps.Bias(w, "encoder.conv_out.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _config.IsCausal);
    }

    /// <summary>Every owned weight tensor, for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convIn is not null) foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        foreach (DownStage stage in _downStages)
        {
            foreach (LtxVaeResnetBlock3d r in stage.Resnets) foreach (Tensor t in r.EnumerateWeights()) yield return t;
            if (stage.Downsampler is not null) foreach (Tensor t in stage.Downsampler.EnumerateWeights()) yield return t;
        }
        foreach (LtxVaeResnetBlock3d r in _midResnets) foreach (Tensor t in r.EnumerateWeights()) yield return t;
        if (_convOut is not null) foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
    }

    /// <summary>Encodes an F32 video <c>[B, InChannels, T, H, W]</c> to the posterior mode <c>[B, LatentChannels, T', H/S, W/S]</c>,
    /// with <c>T' = (T − 1)/TemporalCompression + 1</c> and <c>S = SpatialCompression</c>. The caller owns
    /// <paramref name="video"/> and the returned tensor.</summary>
    public Tensor Encode(IBackend backend, Tensor video)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(video);
        if (_convIn is null || _convOut is null)
            throw new InvalidOperationException("LoadWeights must run before Encode.");
        ValidateVideo(video);

        Tensor h = Patchify(backend, video);
        Tensor next = _convIn.Forward(backend, h);
        h.Dispose();
        h = next;
        foreach (DownStage stage in _downStages)
        {
            foreach (LtxVaeResnetBlock3d r in stage.Resnets) { next = r.Forward(backend, h, null); h.Dispose(); h = next; }
            if (stage.Downsampler is not null) { next = stage.Downsampler.Forward(backend, h); h.Dispose(); h = next; }
        }
        foreach (LtxVaeResnetBlock3d r in _midResnets) { next = r.Forward(backend, h, null); h.Dispose(); h = next; }

        Tensor normed = new Tensor(h.Shape, h.DType);
        backend.NormModulate(NormModulateCall.Channel(normed, h, NormKind.L2Norm, NormEps));
        h.Dispose();
        backend.Silu(normed, normed);
        Tensor raw = _convOut.Forward(backend, normed);
        normed.Dispose();

        // conv_out emits mean ‖ one shared logvar channel; the mode is the mean.
        Tensor mode = new Tensor(new TensorShape([raw.Shape[0], _config.LatentChannels, raw.Shape[2], raw.Shape[3], raw.Shape[4]]), DType.F32);
        StridedCopies.SliceAxis(backend, mode, raw, 1, 0, _config.LatentChannels);
        raw.Dispose();
        return mode;
    }

    /// <summary>Pixel unshuffle with the channel order diffusers' <c>permute(0,1,3,7,5,2,4,6)</c> produces for <c>pt = 1</c>:
    /// <c>c·p² + w·p + h</c>, which is <see cref="ChannelOrder.CWHT"/> with factors (1, p, p).</summary>
    private Tensor Patchify(IBackend backend, Tensor video)
    {
        int b = (int)video.Shape[0], t = (int)video.Shape[2];
        int p = _config.PatchSize;
        int hp = (int)video.Shape[3] / p, wp = (int)video.Shape[4] / p;
        Tensor packed = new Tensor(new TensorShape([b, _config.InChannels * p * p, t, hp, wp]), DType.F32);
        PixelShuffles.SpaceToDepth(backend, packed, video,
            ShuffleSpec.ChannelsFirst(b, t, hp, wp, _config.InChannels, 1, p, p, ChannelOrder.CWHT));
        return packed;
    }

    private void ValidateVideo(Tensor video)
    {
        if (video.Shape.Rank != 5)
            throw new ArgumentException($"Video must be [B, C, T, H, W]; got {video.Shape}.", nameof(video));
        if (video.DType != DType.F32)
            throw new ArgumentException($"Video must be F32; got {video.DType}.", nameof(video));
        if (video.Shape[1] != _config.InChannels)
            throw new ArgumentException($"Video has {video.Shape[1]} channels; the encoder takes {_config.InChannels}.", nameof(video));
        long t = video.Shape[2], h = video.Shape[3], w = video.Shape[4];
        if (t < 1 || (t - 1) % _temporalCompression != 0)
            throw new ArgumentException($"Frame count {t} must be 1 + {_temporalCompression}·n.", nameof(video));
        if (h % _spatialCompression != 0 || w % _spatialCompression != 0)
            throw new ArgumentException($"Height and width ({h}×{w}) must be multiples of {_spatialCompression}.", nameof(video));
    }

    private void ValidateWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        List<string> problems = [];
        foreach (KeyValuePair<string, TensorShape> entry in ExpectedWeightShapes(_config))
        {
            if (!w.TryGetValue(entry.Key, out Tensor? tensor))
            {
                problems.Add($"missing {entry.Key}");
            }
            else if (tensor.Shape != entry.Value)
            {
                problems.Add($"{entry.Key} is {tensor.Shape}, expected {entry.Value}");
            }
        }
        if (problems.Count > 0)
            throw new ArgumentException($"LTX-2 encoder weights do not match the config: {string.Join("; ", problems)}.", nameof(w));
    }

    private static void ValidateConfig(LtxVideo2VaeEncoderConfig config)
    {
        if (config.LatentChannels < 1 || config.InChannels < 1 || config.PatchSize < 1)
            throw new ArgumentException("LatentChannels, InChannels and PatchSize must be positive.", nameof(config));
        if (config.PatchSizeT != 1)
            throw new NotSupportedException($"LTX-2 encoder supports patch_size_t = 1; got {config.PatchSizeT}.");
        if (config.SpatialPaddingMode != "zeros")
            throw new NotSupportedException($"LTX-2 encoder spatial padding '{config.SpatialPaddingMode}' is not implemented; only 'zeros'.");
        int blocks = config.BlockOutChannels.Length;
        if (blocks < 1 || config.LayersPerBlock.Length != blocks + 1 || config.SpatioTemporalScaling.Length != blocks
            || config.DownsampleType.Length != blocks)
        {
            throw new ArgumentException(
                "Per-block arrays must hold one entry per down block; LayersPerBlock holds one more for the mid block.",
                nameof(config));
        }

        int width = config.LatentChannels;
        for (int i = 0; i < blocks; i++)
        {
            int outWidth = config.BlockOutChannels[i];
            if (config.SpatioTemporalScaling[i])
            {
                (int st0, int st1, int st2) = StrideFor(config.DownsampleType[i]);
                int stProd = st0 * st1 * st2;
                if (outWidth % stProd != 0 || width * stProd % outWidth != 0)
                    throw new ArgumentException(
                        $"Down block {i} ({width}→{outWidth}) needs out % {stProd} == 0 and in·{stProd} % out == 0.",
                        nameof(config));
            }
            else if (outWidth != width)
            {
                throw new ArgumentException($"Down block {i} has no downsampler, so its width must stay {width}; got {outWidth}.", nameof(config));
            }
            width = outWidth;
        }
    }

    private static (int T, int H, int W) StrideFor(string downsampleType) => downsampleType switch
    {
        "spatial" => (1, 2, 2),
        "temporal" => (2, 1, 1),
        "spatiotemporal" => (2, 2, 2),
        _ => throw new NotSupportedException(
            $"LTX-2 encoder downsample type '{downsampleType}' is not implemented; use spatial, temporal or spatiotemporal."),
    };

    private static void AddResnet(Dictionary<string, TensorShape> shapes, string prefix, int width)
    {
        AddConv(shapes, $"{prefix}.conv1.conv", width, width);
        AddConv(shapes, $"{prefix}.conv2.conv", width, width);
    }

    private static void AddConv(Dictionary<string, TensorShape> shapes, string prefix, int outC, int inC)
    {
        shapes[$"{prefix}.weight"] = new TensorShape([outC, inC, 3, 3, 3]);
        shapes[$"{prefix}.bias"] = new TensorShape([outC]);
    }
}
