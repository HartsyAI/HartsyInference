using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>The codec as a conversation uses it: whole clips for the voice reference, and one frame at a time in each direction with
/// the convolution histories carried between frames. Latents are channels-last rows of <see cref="LatentDim"/>.</summary>
internal sealed unsafe class HaruCodecSession : IDisposable
{
    private const int StreamId = 0;

    private readonly VibeVoiceAcousticTokenizerModel _model;
    private readonly IBackend _backend;
    private readonly int _latentDim;
    private readonly float _latentStd;
    private VibeVoiceTokenizerStreamingCache _encoderCache = new();
    private VibeVoiceTokenizerStreamingCache _decoderCache = new();
    private float[]? _silence;

    /// <summary>Wraps <paramref name="model"/>, which the caller owns and which must outlive the session.</summary>
    internal HaruCodecSession(VibeVoiceAcousticTokenizerModel model, IBackend backend, int latentDim, float latentStd)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _latentDim = latentDim;
        _latentStd = latentStd;
    }

    internal int LatentDim => _latentDim;

    /// <summary>Forgets both streaming histories: the next frame of either direction starts from silence.</summary>
    internal void Reset()
    {
        _encoderCache.Dispose();
        _decoderCache.Dispose();
        _encoderCache = new VibeVoiceTokenizerStreamingCache();
        _decoderCache = new VibeVoiceTokenizerStreamingCache();
    }

    /// <summary>Latent means of a whole clip, <c>ceil(samples / 3200)</c> rows; the last frame is zero-padded.</summary>
    internal float[] EncodeClip(ReadOnlySpan<float> pcm)
    {
        using Tensor input = Make(pcm, 1, 1, pcm.Length);
        using Tensor mean = _model.Encode(_backend, input, 1, pcm.Length);
        return Copy(mean);
    }

    /// <summary>Latent mean of one frame of audio, continuing the encoder's history.</summary>
    internal float[] EncodeFrame(ReadOnlySpan<float> frame)
    {
        using Tensor input = Make(frame, 1, 1, frame.Length);
        using Tensor mean = _model.Encode(_backend, input, 1, frame.Length, _encoderCache, [StreamId]);
        return Copy(mean);
    }

    /// <summary>The samples of one latent, continuing the decoder's history.</summary>
    internal float[] DecodeFrame(ReadOnlySpan<float> latent)
    {
        using Tensor input = Make(latent, 1, 1, _latentDim);
        using Tensor pcm = _model.Decode(_backend, input, 1, _decoderCache, [StreamId]);
        return Copy(pcm);
    }

    /// <summary>The latent a stream carries before it has any sound: the mean of the last frame of 16 frames of digital silence.</summary>
    internal float[] SilenceLatent()
    {
        if (_silence is null)
        {
            float[] means = EncodeClip(new float[16 * HaruCodec.FrameSamples]);
            _silence = means[^_latentDim..];
        }
        return _silence;
    }

    /// <summary>A draw from the posterior around <paramref name="means"/> (<c>rows * LatentDim</c>): one scale per call, then one noise value
    /// per element, <c>mean + scale * (std / 0.8) * noise</c>.</summary>
    internal float[] Sample(ReadOnlySpan<float> means, IHaruNoise noise)
    {
        Span<float> scale = stackalloc float[1];
        noise.FillGaussian(scale);
        float[] draws = new float[means.Length];
        noise.FillGaussian(draws);
        float spread = scale[0] * (_latentStd / 0.8f);
        float[] latents = new float[means.Length];
        for (int i = 0; i < latents.Length; i++)
        {
            latents[i] = means[i] + spread * draws[i];
        }
        return latents;
    }

    public void Dispose()
    {
        _encoderCache.Dispose();
        _decoderCache.Dispose();
    }

    private static Tensor Make(ReadOnlySpan<float> values, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        values.CopyTo(new Span<float>((void*)t.DataPointer, values.Length));
        return t;
    }

    private static float[] Copy(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();
}
