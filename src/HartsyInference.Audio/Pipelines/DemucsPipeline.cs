using HartsyInference.Audio.Models.Demucs;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>HTDemucs music source separation pipeline. Wraps <see cref="HtDemucs"/>: takes a stereo waveform (left
/// and right channel buffers), runs the dual-branch hybrid transformer, and returns 4 stereo stems
/// (drums, bass, other, vocals) as (left, right) pairs. Stereo 44.1 kHz. Weights load from the released
/// <c>.th</c> state dict via the <c>encoder.*</c>/<c>decoder.*</c>/<c>tencoder.*</c>/<c>tdecoder.*</c>/
/// <c>crosstransformer.*</c>/<c>freq_emb.*</c> prefixes.</summary>
public sealed unsafe class DemucsPipeline : IDisposable
{
    private readonly HtDemucsConfig _cfg;
    private readonly HtDemucs _model;
    private int _disposed;

    public DemucsPipeline(HtDemucsConfig cfg)
    {
        _cfg = cfg;
        _model = new HtDemucs(cfg);
    }

    public int SampleRate => _cfg.SampleRate;

    public IReadOnlyList<string> Sources => _cfg.Sources;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _model.LoadWeights(w);
    }

    /// <summary>Separates a stereo signal into the configured stems. <paramref name="stereoLeft"/> and
    /// <paramref name="stereoRight"/> must be equal length; returns one <c>(Left, Right)</c> pair per stem in the
    /// <see cref="Sources"/> order.</summary>
    public (float[] Left, float[] Right)[] Separate(IBackend backend, float[] stereoLeft, float[] stereoRight)
    {
        ThrowIfDisposed();
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (stereoLeft is null || stereoRight is null) throw new ArgumentNullException(nameof(stereoLeft));
        if (stereoLeft.Length != stereoRight.Length)
            throw new ArgumentException($"Stereo channel lengths must match: {stereoLeft.Length} != {stereoRight.Length}.");

        int length = stereoLeft.Length;
        int channels = _cfg.AudioChannels;
        int srcs = _cfg.NumSources;
        Tensor wav = new(new TensorShape(1, channels, length), DType.F32);
        float* wp = (float*)wav.DataPointer;
        for (int j = 0; j < length; j++) wp[j] = stereoLeft[j];
        for (int j = 0; j < length; j++) wp[(long)length + j] = stereoRight[j];

        Tensor stems = _model.Forward(backend, wav, length);
        wav.Dispose();

        (float[] Left, float[] Right)[] result = new (float[], float[])[srcs];
        float* sp = (float*)stems.DataPointer;
        for (int s = 0; s < srcs; s++)
        {
            float[] left = new float[length];
            float[] right = new float[length];
            long baseL = (((long)s * channels) + 0) * length;
            long baseR = (((long)s * channels) + 1) * length;
            for (int j = 0; j < length; j++) left[j] = sp[baseL + j];
            for (int j = 0; j < length; j++) right[j] = sp[baseR + j];
            result[s] = (left, right);
        }
        stems.Dispose();
        return result;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(DemucsPipeline));
    }
}
