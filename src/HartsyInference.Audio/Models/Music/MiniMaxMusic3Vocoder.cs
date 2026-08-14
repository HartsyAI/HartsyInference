using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Codecs.Dac;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Music;

/// <summary>MiniMax Music 3's Flow-VAE waveform decoder. Architecturally a descript-audio-codec decoder — the same
/// Snake activations, <c>k=7 dilated + k=1</c> residual units at dilations 1/3/9, and <c>kernel = 2·stride,
/// padding = ceil(stride/2)</c> transposed convolutions — so the body is <see cref="DacDecoder"/> and this type only
/// adds what differs: the <c>dec_in_proj</c> lift, the stereo channel fold, and the checkpoint's key names.
///
/// <para>Stereo is a channel fold, not a batch axis: the 128 latent channels are two interleaved 64-channel streams
/// (<c>reshape(2, 64, L)</c>), decoded independently and stacked back as left/right.</para></summary>
public sealed class MiniMaxMusic3Vocoder : IDisposable
{
    /// <summary>Waveform samples per latent frame (∏ of the upsample strides).</summary>
    public const int LatentHopLength = 512;

    private const int LatentChannels = 128;
    private const int DecoderInputDim = 1024;
    // The reference divides by `alpha + 1e-9`; the shared Snake kernel divides by alpha, and the checkpoint's
    // smallest |alpha| is 4.7e-6, so the epsilon is folded into the weights at load instead of into the kernel.
    private const float SnakeAlphaEpsilon = 1e-9f;

    private static readonly int[] _upsampleStrides = [8, 8, 4, 2];

    private readonly DacDecoder _decoder;
    private readonly List<Tensor> _owned = [];
    private Tensor? _decInWeight;
    private Tensor? _decInBias;
    private int _disposed;

    /// <summary>Output sample rate.</summary>
    public int SampleRate { get; }

    public MiniMaxMusic3Vocoder(int sampleRate = 44_100)
    {
        SampleRate = sampleRate;
        _decoder = new DacDecoder(
            new DacConfig
            {
                SampleRate = sampleRate,
                DecoderDim = 1_536,
                DecoderRates = _upsampleStrides,
                TransposeWeightNormDim0 = true,
                DecoderFinalTanh = true,
            },
            prefix: "decoder");
    }

    /// <summary>Fuses weight-norm and maps the checkpoint's <c>blocks.i.res_unit_j</c> names onto the
    /// <c>model.i.block.j</c> layout <see cref="DacDecoder"/> expects. Every convolution is handed over
    /// pre-fused so the decoder allocates nothing of its own and this type owns the whole weight set.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _decInWeight = Borrowed(weights, "dec_in_proj.weight");
        _decInBias = Borrowed(weights, "dec_in_proj.bias");

        Dictionary<string, Tensor> remapped = new(StringComparer.Ordinal);
        Conv(weights, remapped, "conv_in", "decoder.model.0");
        for (int stage = 0; stage < _upsampleStrides.Length; stage++)
        {
            string target = $"decoder.model.{stage + 1}.block";
            remapped[$"{target}.0.alpha"] = SnakeAlpha(weights, $"blocks.{stage}.snake1.alpha");
            Conv(weights, remapped, $"blocks.{stage}.conv_t1", $"{target}.1");
            for (int unit = 0; unit < 3; unit++)
            {
                string source = $"blocks.{stage}.res_unit{unit + 1}";
                string unitTarget = $"{target}.{unit + 2}.block";
                remapped[$"{unitTarget}.0.alpha"] = SnakeAlpha(weights, $"{source}.snake1.alpha");
                Conv(weights, remapped, $"{source}.conv1", $"{unitTarget}.1");
                remapped[$"{unitTarget}.2.alpha"] = SnakeAlpha(weights, $"{source}.snake2.alpha");
                Conv(weights, remapped, $"{source}.conv2", $"{unitTarget}.3");
            }
        }
        remapped[$"decoder.model.{_upsampleStrides.Length + 1}.alpha"] = SnakeAlpha(weights, "snake_out.alpha");
        Conv(weights, remapped, "conv_out", $"decoder.model.{_upsampleStrides.Length + 2}");
        _decoder.LoadWeights(remapped);
    }

    /// <summary>Decodes flow-matched latents <c>[1, 128, length]</c> into a stereo waveform
    /// <c>[1, 2, length · 512]</c> in <c>[-1, 1]</c>. Caller owns the result.</summary>
    public Tensor Decode(IBackend backend, Tensor latents)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(latents);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_decInWeight is null)
        {
            throw new InvalidOperationException($"{nameof(MiniMaxMusic3Vocoder)}.{nameof(LoadWeights)} must run before decoding.");
        }
        if (latents.Shape.Rank != 3 || latents.Shape[0] != 1 || latents.Shape[1] != LatentChannels)
        {
            throw new ArgumentException($"expected a [1, {LatentChannels}, length] latent, got {latents.Shape}.", nameof(latents));
        }

        int length = (int)latents.Shape[2];
        int half = LatentChannels / 2;
        Tensor[] channels = new Tensor[2];
        try
        {
            for (int side = 0; side < 2; side++)
            {
                using Tensor stream = new Tensor(new TensorShape(1, half, length), DType.F32);
                backend.SliceRows(stream, latents, side * half);
                using Tensor lifted = new Tensor(new TensorShape(1, DecoderInputDim, length), DType.F32);
                backend.Conv1d(lifted, stream, _decInWeight!, _decInBias, 1, 0, 0, 1, 1);
                channels[side] = _decoder.Forward(backend, lifted, batch: 1, tFrames: length);
            }
            Tensor waveform = new Tensor(new TensorShape(1, 2, (int)channels[0].Shape[2]), DType.F32);
            backend.Concat(waveform, channels, 1);
            return waveform;
        }
        finally
        {
            foreach (Tensor? channel in channels)
            {
                channel?.Dispose();
            }
        }
    }

    /// <summary>Every loaded weight, for <see cref="IBackend.PreloadWeights"/>/<see cref="IBackend.FreeWeights"/>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_decInWeight is not null) { yield return _decInWeight; }
        if (_decInBias is not null) { yield return _decInBias; }
        foreach (Tensor weight in _decoder.EnumerateWeights())
        {
            yield return weight;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        foreach (Tensor tensor in _owned)
        {
            tensor.Dispose();
        }
        _owned.Clear();
        _decInWeight = null;
        _decInBias = null;
    }

    private void Conv(IReadOnlyDictionary<string, Tensor> weights, Dictionary<string, Tensor> target, string from, string to)
    {
        target[$"{to}.weight"] = Own(WeightNormFusion.Compose(weights, from), weights.GetValueOrDefault($"{from}.weight"));
        target[$"{to}.bias"] = Borrowed(weights, $"{from}.bias");
    }

    private unsafe Tensor SnakeAlpha(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        Tensor source = weights[key];
        Tensor shifted = new Tensor(source.Shape, DType.F32);
        float* src = (float*)source.DataPointer;
        float* dst = (float*)shifted.DataPointer;
        for (long i = 0; i < source.ElementCount; i++)
        {
            dst[i] = src[i] + SnakeAlphaEpsilon;
        }
        _owned.Add(shifted);
        return shifted;
    }

    private Tensor Borrowed(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        Tensor source = weights[key];
        Tensor cast = WhisperOps.EnsureF32(source);
        if (!ReferenceEquals(cast, source))
        {
            _owned.Add(cast);
        }
        return cast;
    }

    /// <summary>Tracks <paramref name="fused"/> for disposal unless it is the checkpoint's own borrowed tensor.</summary>
    private Tensor Own(Tensor fused, Tensor? borrowedSource)
    {
        if (!ReferenceEquals(fused, borrowedSource))
        {
            _owned.Add(fused);
        }
        return fused;
    }
}
