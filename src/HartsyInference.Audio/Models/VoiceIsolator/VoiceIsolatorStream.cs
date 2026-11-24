using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>Streaming isolation with the causal checkpoint: the voice leaves as the audio arrives. Samples are pushed in pieces of
/// any size; every block of new frames runs once with its state carried (time attention over a sliding cache, the time Mamba's
/// recurrence, the STFT's overlap-add) and an output sample is released once every frame that touches it, and the look-ahead frames
/// after it, are done. The latency is the analysis window plus the look-ahead.
/// <para>With <c>windowFrames</c> and <c>crossfadeFrames</c> the time attention never looks back across a window boundary: a second
/// lane starts <c>crossfadeFrames</c> before each boundary with the Mamba states cloned and the attention empty, the two lanes' raw
/// decoder outputs blend linearly over the crossfade, and the second lane carries on from the boundary.</para></summary>
public sealed class VoiceIsolatorStream : IDisposable
{
    private readonly VoiceIsolatorNetwork _network;
    private readonly IBackend _backend;
    private readonly VoiceIsolatorConfig _config;
    private readonly VoiceIsolatorSpectral _spectral;
    private readonly VoiceIsolatorStreamAttention _attention;
    private readonly int _blockFrames;
    private readonly int? _windowFrames;
    private readonly int _crossfadeFrames;
    private readonly int _lookahead;
    private readonly int _bins;
    private readonly List<float[]> _raw = [];
    private readonly List<(float[] Re, float[] Im)> _spectra = [];
    private readonly List<int> _boundaries = [];
    private VoiceIsolatorLane _laneA;
    private VoiceIsolatorLane? _laneB;
    private float[] _audio = [];
    private long _audioBase;
    private long _frames;
    private long _emitted;
    private long _pendingBase;
    private float[] _ola;
    private float[] _envelope;
    private long _olaBase;
    private long _released;
    private bool _closed;

    /// <summary>Starts a session on a causal network, at the look-ahead given or the checkpoint default.</summary>
    /// <param name="blockFrames">Frames computed together: a throughput knob that adds no latency.</param>
    /// <param name="cacheFrames">Frames of attention past each query keeps; null takes the checkpoint's attention window.</param>
    /// <param name="windowFrames">Window of the two-lane mode; null turns the windows off.</param>
    /// <param name="crossfadeFrames">Crossfade of the two-lane mode; required with <paramref name="windowFrames"/>.</param>
    /// <param name="lookahead">Look-ahead in frames, one of the checkpoint trained positions; null takes its default.</param>
    public VoiceIsolatorStream(VoiceIsolatorNetwork network, IBackend backend, int blockFrames = 4, int? cacheFrames = null,
        int? windowFrames = null, int? crossfadeFrames = null, int? lookahead = null)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(backend);
        if (!network.Config.Causal)
        {
            throw new ArgumentException("Streaming needs the causal checkpoint.", nameof(network));
        }
        if (blockFrames < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(blockFrames), "Process at least one frame per block.");
        }
        if ((windowFrames is null) != (crossfadeFrames is null))
        {
            throw new ArgumentException("windowFrames and crossfadeFrames go together.");
        }
        if (windowFrames is not null && !(0 < crossfadeFrames && crossfadeFrames < windowFrames))
        {
            throw new ArgumentException("crossfadeFrames must be positive and shorter than windowFrames.");
        }
        _network = network;
        _backend = backend;
        _config = network.Config;
        _spectral = network.Spectral;
        _bins = _spectral.Bins;
        _blockFrames = blockFrames;
        _windowFrames = windowFrames;
        _crossfadeFrames = crossfadeFrames ?? 0;
        _lookahead = network.ValidateLookahead(lookahead ?? _config.LookaheadFrames);
        if (cacheFrames is { } cacheValue)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cacheValue);
        }
        int cache = cacheFrames ?? (_config.AttentionWindowFrames > 0
            ? _config.AttentionWindowFrames : Math.Max(1, (int)Math.Round(_config.ChunkSeconds * _config.SampleRate / _config.HopLength)));
        int lane = windowFrames is null ? 0 : windowFrames.Value + _crossfadeFrames;
        int capacity = lane > 0 && lane <= cache + blockFrames ? lane : cache + Math.Max(cache / 2, blockFrames);
        _attention = new VoiceIsolatorStreamAttention(cache, capacity, _config.AttentionWindowFrames, _config.HeadDim);
        _laneA = network.NewLane();
        _ola = new float[_config.NFft];
        _envelope = new float[_config.NFft];
    }

    /// <summary>Input samples that must follow an output sample before it is released: the analysis window plus the look-ahead frames.</summary>
    public int LatencySamples => _config.NFft + _config.HopLength * _lookahead;

    /// <summary>First frame of every window the stream has crossed into, once the swap to the second lane happened.</summary>
    public IReadOnlyList<int> Boundaries => _boundaries;

    /// <summary>Feeds mono samples at the network's rate and returns the output samples that became final.</summary>
    public float[] Push(ReadOnlySpan<float> samples)
    {
        if (_closed)
        {
            throw new InvalidOperationException("The stream was flushed.");
        }
        float[] joined = new float[_audio.Length + samples.Length];
        _audio.CopyTo(joined, 0);
        samples.CopyTo(joined.AsSpan(_audio.Length));
        _audio = joined;
        long total = _audioBase + _audio.Length;
        long computable = (total - _config.NFft / 2) >= 0 ? (total - _config.NFft / 2) / _config.HopLength + 1 : 0;
        Advance(computable);
        float[] released = Release(_emitted * _config.HopLength - _config.NFft / 2);
        Trim();
        return released;
    }

    /// <summary>Ends the stream and returns every remaining output sample.</summary>
    public float[] Flush()
    {
        if (_closed)
        {
            return [];
        }
        _closed = true;
        long total = _audioBase + _audio.Length;
        Advance(total / _config.HopLength + 1);
        while (_emitted < _frames)
        {
            SynthesizeFrame(_emitted, Math.Min(_emitted + _lookahead, _frames - 1));
        }
        return Release(total);
    }

    private void Advance(long limit)
    {
        while (_frames < limit)
        {
            int count = (int)Math.Min(Math.Min(_blockFrames, limit - _frames), NextCut() - _frames);
            float[] re = new float[(long)count * _bins];
            float[] im = new float[(long)count * _bins];
            for (int f = 0; f < count; f++)
            {
                AnalyzeFrame(_frames + f, re, im, f);
            }
            float[] raw = RunBlock(re, im, count);
            long frameSize = (long)_bins * 4;
            for (int f = 0; f < count; f++)
            {
                float[] frameRaw = new float[frameSize];
                Array.Copy(raw, f * frameSize, frameRaw, 0, frameSize);
                _raw.Add(frameRaw);
                float[] fr = new float[_bins];
                float[] fi = new float[_bins];
                Array.Copy(re, (long)f * _bins, fr, 0, _bins);
                Array.Copy(im, (long)f * _bins, fi, 0, _bins);
                _spectra.Add((fr, fi));
            }
            _frames += count;
        }
        while (_emitted + _lookahead < _frames)
        {
            SynthesizeFrame(_emitted, _emitted + _lookahead);
        }
        Trim();
    }

    private void AnalyzeFrame(long frame, float[] re, float[] im, int row)
    {
        int n = _config.NFft;
        long start = frame * _config.HopLength - n / 2;
        float[] window = new float[n];
        long total = _audioBase + _audio.Length;
        for (int i = 0; i < n; i++)
        {
            long at = start + i;
            window[i] = at >= _audioBase && at < total ? _audio[at - _audioBase] : 0f;
        }
        _spectral.AnalyzeFrame(window, re, im, row);
    }

    private long NextCut()
    {
        if (_windowFrames is null)
        {
            return _frames + _blockFrames;
        }
        long window = _windowFrames.Value;
        long start = _frames - _frames % window;
        long crossfadeStart = start + window - _crossfadeFrames;
        return _frames < crossfadeStart ? crossfadeStart : start + window;
    }

    private float[] RunBlock(float[] re, float[] im, int count)
    {
        using Tensor features = _network.Encode(_backend, re, im, count, _lookahead);
        if (_windowFrames is null)
        {
            return Decode(features, count, _laneA);
        }
        int window = _windowFrames.Value;
        int first = (int)(_frames % window);
        int crossfadeStart = window - _crossfadeFrames;
        if (first + count > window || (first < crossfadeStart && crossfadeStart < first + count))
        {
            throw new InvalidOperationException("A block crossed a crossfade start or a window boundary.");
        }
        if (first >= crossfadeStart && _laneB is null)
        {
            _laneB = _laneA.Fork();
        }
        if (first < crossfadeStart)
        {
            return Decode(features, count, _laneA);
        }
        float[] rawA = Decode(features, count, _laneA);
        float[] rawB = Decode(features, count, _laneB!);
        long frameSize = (long)_bins * 4;
        for (int f = 0; f < count; f++)
        {
            float weight = Math.Clamp((float)(first + f - crossfadeStart + 1) / _crossfadeFrames, 0f, 1f);
            for (long i = 0; i < frameSize; i++)
            {
                long at = f * frameSize + i;
                rawA[at] = rawA[at] * (1f - weight) + rawB[at] * weight;
            }
        }
        if (first + count == window)
        {
            _boundaries.Add((int)(_frames + count));
            _laneA.Dispose();
            _laneA = _laneB!;
            _laneB = null;
        }
        return rawA;
    }

    private float[] Decode(Tensor features, int count, VoiceIsolatorLane lane)
    {
        Tensor x = _network.StreamBlocks(_backend, features, count, lane, _attention);
        float[] raw = _network.Decode(_backend, x, count);
        x.Dispose();
        return raw;
    }

    private void SynthesizeFrame(long index, long source)
    {
        int n = _config.NFft;
        float[] raw = _raw[(int)(source - _pendingBase)];
        (float[] spectrumRe, float[] spectrumIm) = _spectra[(int)(index - _pendingBase)];
        float[] re = (float[])spectrumRe.Clone();
        float[] im = (float[])spectrumIm.Clone();
        VoiceIsolatorNetwork.Correct(raw, re, im, 1, _bins, _config.CorrectionBound);
        float[] frame = _spectral.InverseFrame(re, im, 0);
        long start = index * _config.HopLength - _olaBase;
        long needed = start + n;
        if (needed > _ola.Length)
        {
            Array.Resize(ref _ola, (int)needed);
            Array.Resize(ref _envelope, (int)needed);
        }
        ReadOnlySpan<float> window = _spectral.Window;
        for (int i = 0; i < n; i++)
        {
            _ola[start + i] += frame[i] * window[i];
            _envelope[start + i] += window[i] * window[i];
        }
        _emitted++;
    }

    private float[] Release(long upTo)
    {
        long total = _audioBase + _audio.Length;
        long start = _released;
        long stop = Math.Min(upTo, total);
        if (stop <= start)
        {
            return [];
        }
        float[] output = new float[stop - start];
        long offset = _config.NFft / 2 - _olaBase;
        for (long s = start; s < stop; s++)
        {
            long at = s + offset;
            output[s - start] = _audio[s - _audioBase] + _ola[at] / MathF.Max(_envelope[at], 1e-11f);
        }
        _released = stop;
        return output;
    }

    private void Trim()
    {
        long drop = _emitted - _pendingBase;
        _raw.RemoveRange(0, (int)Math.Min(drop, _raw.Count));
        _spectra.RemoveRange(0, (int)Math.Min(drop, _spectra.Count));
        _pendingBase = _emitted;
        // The overlap-add index of sample s is s + n_fft / 2 - base, so keeping from _released - base keeps n_fft / 2
        // entries before the first unreleased sample.
        long keepFrom = _released - _olaBase;
        if (keepFrom > 0)
        {
            long remaining = _ola.Length - keepFrom;
            Array.Copy(_ola, keepFrom, _ola, 0, remaining);
            Array.Copy(_envelope, keepFrom, _envelope, 0, remaining);
            Array.Resize(ref _ola, (int)remaining);
            Array.Resize(ref _envelope, (int)remaining);
            _olaBase += keepFrom;
        }
        long neededFrom = Math.Min(_released, Math.Max(0, _frames * _config.HopLength - _config.NFft / 2));
        long dropAudio = neededFrom - _audioBase;
        if (dropAudio > 0)
        {
            _audio = _audio[(int)dropAudio..];
            _audioBase = neededFrom;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _laneA.Dispose();
        _laneB?.Dispose();
    }
}
