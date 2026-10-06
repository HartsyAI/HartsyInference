using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>ControlFoley text-to-audio: CLIP text features, the flow-matching network (classifier-free guidance against the
/// negative prompt), then the VAE and BigVGAN decoder, as in the official <c>demo.py</c> without video or reference audio.
/// The pipeline borrows the three models; the caller disposes them.</summary>
public sealed unsafe class ControlFoleyPipeline
{
    private readonly ControlFoleyNetwork _network;
    private readonly ControlFoleyClip _clip;
    private readonly ControlFoleyAudioDecoder _decoder;

    public ControlFoleyPipeline(ControlFoleyNetwork network, ControlFoleyClip clip, ControlFoleyAudioDecoder decoder)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(clip);
        ArgumentNullException.ThrowIfNull(decoder);
        _network = network;
        _clip = clip;
        _decoder = decoder;
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
    }

    /// <summary>Generates mono PCM at <see cref="SampleRate"/>.</summary>
    public float[] Generate(IBackend backend, Request request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.DurationSeconds <= 0 || request.Steps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Duration and steps must be positive.");
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
        using Tensor audioF = _network.GetEmptyAudioSequence(1);
        using Tensor timbreF = _network.GetEmptyTimbreSequence(1);
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
