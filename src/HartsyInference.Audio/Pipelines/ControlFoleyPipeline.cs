using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>ControlFoley text-to-audio: CLIP text features, the flow-matching network (classifier-free guidance against the
/// negative prompt), then the VAE and BigVGAN decoder, as in the official <c>demo.py</c> without video. An optional reference
/// clip conditions the network through the CLAP audio embedding and the MusicGen-Style timbre feature. The pipeline borrows the
/// models; the caller disposes them.</summary>
public sealed unsafe class ControlFoleyPipeline
{
    private readonly ControlFoleyNetwork _network;
    private readonly ControlFoleyClip _clip;
    private readonly ControlFoleyAudioDecoder _decoder;
    private readonly ControlFoleyClap? _clap;
    private readonly ControlFoleyStyleEncoder? _style;

    /// <summary>Rate the CLAP embedding of the reference clip is computed at.</summary>
    public const int ClapSampleRate = 16_000;

    /// <summary>Rate of the timbre excerpt.</summary>
    public const int TimbreSampleRate = 32_000;

    /// <summary>Shortest and longest timbre excerpt in seconds; shorter clips are zero-padded and longer ones truncated.</summary>
    public const int MinTimbreSeconds = 2, MaxTimbreSeconds = 4;

    public ControlFoleyPipeline(ControlFoleyNetwork network, ControlFoleyClip clip, ControlFoleyAudioDecoder decoder,
        ControlFoleyClap? clap = null, ControlFoleyStyleEncoder? style = null)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(clip);
        ArgumentNullException.ThrowIfNull(decoder);
        _network = network;
        _clip = clip;
        _decoder = decoder;
        _clap = clap;
        _style = style;
    }

    /// <summary>Whether reference-audio conditioning is available (both encoders were supplied).</summary>
    public bool SupportsReferenceAudio => _clap is not null && _style is not null;

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

        /// <summary>Optional reference clip (mono, at <see cref="ReferenceAudio.SampleRate"/>) whose content and timbre the output follows.</summary>
        public ReferenceAudio? Reference { get; init; }
    }

    /// <summary>A mono reference clip. Callers mix multi-channel audio down first, as <c>demo.py</c> does after resampling.</summary>
    public sealed record ReferenceAudio(float[] Samples, int SampleRate);

    /// <summary>Generates mono PCM at <see cref="SampleRate"/>.</summary>
    public float[] Generate(IBackend backend, Request request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.DurationSeconds <= 0 || request.Steps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Duration and steps must be positive.");
        }

        if (request.Reference is not null)
        {
            ValidateReference(request.Reference);
        }

        ControlFoleyTemporalConfig temporal = ControlFoleyTemporalConfig.Default44k with { TotalTimeSeconds = request.DurationSeconds };
        _network.UpdateSequenceLengths(temporal);
        ControlFoleyNetworkConfig cfg = _network.Config;

        float[] text = _clip.EncodeText(backend, [request.Prompt]);
        float[] negative = _clip.EncodeText(backend, [request.NegativePrompt]);
        cancel.ThrowIfCancellationRequested();
        using Tensor textF = FromHost(text, 1, cfg.TextSeqLen, cfg.TextDim);
        using Tensor negativeF = FromHost(negative, 1, cfg.TextSeqLen, cfg.TextDim);
        using Tensor clipF = _network.GetEmptyClipSequence(1);
        using Tensor visualF = _network.GetEmptyVisualSequence(1);
        using Tensor syncF = _network.GetEmptySyncSequence(1);
        using Tensor audioF = request.Reference is null ? _network.GetEmptyAudioSequence(1) : EncodeClap(backend, request.Reference, cfg);
        using Tensor timbreF = request.Reference is null ? _network.GetEmptyTimbreSequence(1) : EncodeTimbre(backend, request.Reference, cfg);
        using ControlFoleyConditions conditions = _network.PreprocessConditions(backend, clipF, visualF, syncF, textF, audioF, timbreF);
        using ControlFoleyConditions empty = _network.GetEmptyConditions(backend, 1, negativeF);

        using Tensor noise = ControlFoleySampler.CreateNoise(1, cfg.LatentSeqLen, cfg.LatentDim, request.Seed);
        using Tensor latent = ControlFoleySampler.Sample(backend, _network, conditions, empty, noise, request.Steps, request.CfgStrength);
        cancel.ThrowIfCancellationRequested();

        float[] flat = new ReadOnlySpan<float>((void*)latent.DataPointer, (int)latent.ElementCount).ToArray();
        float[] audio = _decoder.Decode(backend, flat);
        int samples = Math.Min(audio.Length, (int)(request.DurationSeconds * SampleRate));
        return audio.AsSpan(0, samples).ToArray();
    }

    private Tensor EncodeClap(IBackend backend, ReferenceAudio reference, ControlFoleyNetworkConfig cfg)
    {
        float[] audio = SincResampler.Resample(reference.Samples, reference.SampleRate, ClapSampleRate);
        return FromHost(_clap!.Embed(backend, audio), 1, 1, cfg.AudioDim);
    }

    /// <summary>The timbre excerpt of <c>demo.py</c>: resampled to 32 kHz, zero-padded to 2 s or cut to 4 s, encoded as
    /// <c>duration = samples / 32000</c>.</summary>
    private Tensor EncodeTimbre(IBackend backend, ReferenceAudio reference, ControlFoleyNetworkConfig cfg)
    {
        float[] audio = SincResampler.Resample(reference.Samples, reference.SampleRate, TimbreSampleRate);
        int length = Math.Clamp(audio.Length, MinTimbreSeconds * TimbreSampleRate, MaxTimbreSeconds * TimbreSampleRate);
        Array.Resize(ref audio, length);
        return FromHost(_style!.EncodeTimbre(backend, audio), 1, 1, cfg.TimbreDim);
    }

    private void ValidateReference(ReferenceAudio reference)
    {
        if (_clap is null || _style is null)
        {
            throw new InvalidOperationException("Reference audio needs the CLAP and timbre encoders; construct the pipeline with both.");
        }

        if (reference.Samples.Length == 0 || reference.SampleRate <= 0)
        {
            throw new ArgumentException("Reference audio needs samples and a positive sample rate.", nameof(reference));
        }
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
