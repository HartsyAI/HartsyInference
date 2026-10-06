using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>ControlFoley text-to-audio and video-to-audio: CLIP text features, optional video conditions (CLIP frames,
/// CAV-MAE visual tokens, Synchformer tokens), the flow-matching network (classifier-free guidance against the negative
/// prompt), then the VAE and BigVGAN decoder, as in the official <c>demo.py</c> without reference audio.
/// The pipeline borrows the models; the caller disposes them.</summary>
public sealed unsafe class ControlFoleyPipeline
{
    private readonly ControlFoleyNetwork _network;
    private readonly ControlFoleyClip _clip;
    private readonly ControlFoleyAudioDecoder _decoder;
    private readonly ControlFoleySynchformer? _synchformer;
    private readonly ControlFoleyCavMae? _cavMae;

    public ControlFoleyPipeline(ControlFoleyNetwork network, ControlFoleyClip clip, ControlFoleyAudioDecoder decoder)
        : this(network, clip, decoder, null, null)
    {
    }

    /// <summary>Creates a pipeline that can also condition on video; both video encoders are needed together.</summary>
    public ControlFoleyPipeline(ControlFoleyNetwork network, ControlFoleyClip clip, ControlFoleyAudioDecoder decoder,
        ControlFoleySynchformer? synchformer, ControlFoleyCavMae? cavMae)
    {
        if ((synchformer is null) != (cavMae is null))
        {
            throw new ArgumentException("Synchformer and CAV-MAE-ST must be supplied together.");
        }

        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(clip);
        ArgumentNullException.ThrowIfNull(decoder);
        _network = network;
        _clip = clip;
        _decoder = decoder;
        _synchformer = synchformer;
        _cavMae = cavMae;
    }

    /// <summary>Output sample rate in Hz.</summary>
    public int SampleRate => _decoder.SampleRate;

    /// <summary>One generation.</summary>
    public sealed record Request
    {
        public required string Prompt { get; init; }

        public string NegativePrompt { get; init; } = string.Empty;

        public double DurationSeconds { get; init; } = 8.0;

        public int Steps { get; init; } = 25;

        public float CfgStrength { get; init; } = 4.5f;

        public int Seed { get; init; } = 42;

        /// <summary>Decoded source video; needs a pipeline built with the video encoders. The duration shrinks to the
        /// clip's usable length when it is shorter than <see cref="DurationSeconds"/>.</summary>
        public ControlFoleyRawVideo? Video { get; init; }

        /// <summary>Ignore the CLIP stream and keep only the visual and sync streams (<c>--mask_away_clip</c>).</summary>
        public bool MaskAwayClip { get; init; }

        /// <summary>Frame rates and sizes used to sample <see cref="Video"/>.</summary>
        public ControlFoleyVideoOptions VideoOptions { get; init; } = ControlFoleyVideoOptions.Default;
    }

    /// <summary>Generates mono PCM at <see cref="SampleRate"/>.</summary>
    public float[] Generate(IBackend backend, Request request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.DurationSeconds <= 0 || request.Steps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Duration and steps must be positive.");
        }

        double duration = request.DurationSeconds;
        ControlFoleyVideoInput? video = null;
        if (request.Video is not null)
        {
            if (_synchformer is null || _cavMae is null)
            {
                throw new InvalidOperationException("This pipeline was built without the Synchformer and CAV-MAE-ST encoders.");
            }

            video = ControlFoleyVideoPreprocessor.Prepare(request.Video, duration, request.VideoOptions);
            duration = Math.Min(duration, video.TotalDuration);
        }

        ControlFoleyTemporalConfig temporal = ControlFoleyTemporalConfig.Default44k with { TotalTimeSeconds = duration };
        _network.UpdateSequenceLengths(temporal);
        ControlFoleyNetworkConfig cfg = _network.Config;

        float[] text = _clip.EncodeText(backend, [request.Prompt]);
        float[] negative = _clip.EncodeText(backend, [request.NegativePrompt]);
        cancel.ThrowIfCancellationRequested();
        using Tensor textF = FromHost(text, 1, cfg.TextSeqLen, cfg.TextDim);
        using Tensor negativeF = FromHost(negative, 1, cfg.TextSeqLen, cfg.TextDim);
        using Tensor clipF = video is not null && !request.MaskAwayClip
            ? FromHost(_clip.EncodeImages(backend, video.Clip), 1, cfg.ClipSeqLen, cfg.ClipDim)
            : _network.GetEmptyClipSequence(1);
        using Tensor visualF = video is not null
            ? FromHost(_cavMae!.EncodePooled(backend, video.Visual), 1, cfg.VisualSeqLen, cfg.VisualDim)
            : _network.GetEmptyVisualSequence(1);
        using Tensor syncF = video is not null
            ? FromHost(_synchformer!.Encode(backend, video.Sync), 1, cfg.SyncSeqLen, cfg.SyncDim)
            : _network.GetEmptySyncSequence(1);
        using Tensor audioF = _network.GetEmptyAudioSequence(1);
        using Tensor timbreF = _network.GetEmptyTimbreSequence(1);
        using ControlFoleyConditions conditions = _network.PreprocessConditions(backend, clipF, visualF, syncF, textF, audioF, timbreF);
        using ControlFoleyConditions empty = _network.GetEmptyConditions(backend, 1, negativeF);

        using Tensor noise = ControlFoleySampler.CreateNoise(1, cfg.LatentSeqLen, cfg.LatentDim, request.Seed);
        using Tensor latent = ControlFoleySampler.Sample(backend, _network, conditions, empty, noise, request.Steps, request.CfgStrength);
        cancel.ThrowIfCancellationRequested();

        float[] flat = new ReadOnlySpan<float>((void*)latent.DataPointer, (int)latent.ElementCount).ToArray();
        float[] audio = _decoder.Decode(backend, flat);
        int samples = Math.Min(audio.Length, (int)(duration * SampleRate));
        return audio.AsSpan(0, samples).ToArray();
    }

    private static Tensor FromHost(float[] data, int b, int t, int d)
    {
        if (data.Length != (long)b * t * d)
        {
            throw new InvalidDataException($"Expected {(long)b * t * d} features, got {data.Length}.");
        }

        Tensor o = new(new TensorShape(b, t, d), DType.F32);
        data.CopyTo(new Span<float>((void*)o.DataPointer, data.Length));
        return o;
    }
}
