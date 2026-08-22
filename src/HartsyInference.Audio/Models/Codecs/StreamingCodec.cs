using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs;

/// <summary>Streaming wrapper that turns any chunked codec (EnCodec, DAC, SNAC, Mimi, XCodec, WavTokenizer) into an incremental <c>Write(samples) → emit codes</c> pipeline suitable for live mic input.</summary>
/// <remarks>
/// <para>This wrapper itself is causal-naive: every full chunk of audio is encoded
/// fresh (no internal cache like VibeVoice's <see cref="VibeVoiceTokenizerStreamingCache"/>).
/// That means the per-chunk cost is the same as a single non-streaming pass, and the
/// quality matches non-streaming encode at chunk boundaries (good enough for the
/// non-causal codecs DAC / SNAC and acceptable for the causal EnCodec / Mimi at
/// chunk_size &gt;= 200 ms; below that EnCodec / Mimi callers should use a per-codec
/// streaming wrapper with proper cache propagation).</para>
///
/// <para><see cref="ChunkSamples"/> determines latency vs efficiency: larger chunks
/// amortize encoder cost better but introduce more buffering delay. The default
/// <c>= sample_rate / 5</c> (~200 ms) is a reasonable starting point for live TTS or
/// live-VC scenarios.</para>
///
/// <para>Used by Sesame CSM's live demo (via Mimi codec), live-VC applications (via
/// DAC), and any future low-latency wrapper that needs a chunk-by-chunk surface over a
/// non-causal-aware codec.</para>
/// </remarks>
public sealed class StreamingCodecEncoder<TCodec> : IDisposable where TCodec : class
{
    private readonly TCodec _codec;
    private readonly Func<TCodec, IBackend, Tensor, int, int, Tensor> _encodeFn;
    private readonly AudioRingBuffer _ring;
    private readonly IBackend _backend;
    private readonly int _chunkSamples;
    private readonly int _channels;
    private readonly float[] _stagingPcm;
    private int _disposed;

    public int ChunkSamples => _chunkSamples;

    public StreamingCodecEncoder(
        TCodec codec,
        IBackend backend,
        Func<TCodec, IBackend, Tensor, int, int, Tensor> encodeFn,
        int chunkSamples,
        int channels = 1,
        int bufferCapacity = 0)
    {
        _codec = codec;
        _backend = backend;
        _encodeFn = encodeFn;
        _chunkSamples = chunkSamples;
        _channels = channels;
        _ring = new AudioRingBuffer(bufferCapacity > 0 ? bufferCapacity : chunkSamples * 4);
        _stagingPcm = new float[chunkSamples * channels];
    }

    /// <summary>Appends incoming PCM samples to the internal buffer. Safe to call from the audio-callback thread.</summary>
    public void Write(ReadOnlySpan<float> samples) => _ring.Write(samples);

    /// <summary>Drains whatever full chunks are buffered, encodes each one, and returns the resulting code tensors. Caller owns disposal of every returned tensor.</summary>
    public List<Tensor> ExtractAvailable()
    {
        ThrowIfDisposed();
        List<Tensor> result = new();
        while (_ring.Available >= _chunkSamples)
        {
            int copied = _ring.Read(_stagingPcm.AsSpan(0, _chunkSamples));
            if (copied != _chunkSamples) break;
            Tensor pcm = BuildPcmTensor();
            Tensor codes = _encodeFn(_codec, _backend, pcm, 1, _chunkSamples);
            pcm.Dispose();
            result.Add(codes);
        }
        return result;
    }

    public unsafe void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
    }

    private unsafe Tensor BuildPcmTensor()
    {
        Tensor t = new(new TensorShape(1, _channels, _chunkSamples), DType.F32);
        float* dst = (float*)t.DataPointer;
        for (int c = 0; c < _channels; c++)
        {
            for (int i = 0; i < _chunkSamples; i++)
            {
                dst[c * _chunkSamples + i] = _stagingPcm[i * _channels + c];
            }
        }
        return t;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }
}

/// <summary>Mirror class for the decode side — wraps a codec's <c>Decode</c> method so callers can stream codes in and pull PCM chunks out via an <see cref="AudioStreamer"/>.</summary>
public sealed class StreamingCodecDecoder<TCodec> : IDisposable where TCodec : class
{
    private readonly TCodec _codec;
    private readonly Func<TCodec, IBackend, Tensor, int, int, Tensor> _decodeFn;
    private readonly IBackend _backend;
    private readonly AudioStreamer _streamer;
    private readonly int _sampleRate;
    private readonly int _channels;
    private long _samplesEmitted;
    private int _disposed;

    public StreamingCodecDecoder(
        TCodec codec,
        IBackend backend,
        Func<TCodec, IBackend, Tensor, int, int, Tensor> decodeFn,
        int sampleRate,
        int channels = 1,
        int channelCapacity = 64)
    {
        _codec = codec;
        _backend = backend;
        _decodeFn = decodeFn;
        _sampleRate = sampleRate;
        _channels = channels;
        _streamer = new AudioStreamer(channelCapacity);
    }

    /// <summary>Decodes one chunk of codes <c>[nQ, 1, T_frames]</c> and pushes the resulting PCM to the streamer as an <see cref="AudioChunk"/>. Caller is responsible for disposing the input codes tensor.</summary>
    public async ValueTask SubmitChunkAsync(Tensor codes, int tFrames, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        Tensor pcm = _decodeFn(_codec, _backend, codes, 1, tFrames);
        try
        {
            int samples = (int)pcm.Shape[2];
            float[] buffer = new float[samples * _channels];
            unsafe
            {
                float* src = (float*)pcm.DataPointer;
                for (int c = 0; c < _channels; c++)
                {
                    for (int i = 0; i < samples; i++)
                    {
                        buffer[i * _channels + c] = src[c * samples + i];
                    }
                }
            }
            AudioChunk chunk = new(buffer, _sampleRate, _channels, _samplesEmitted);
            _samplesEmitted += samples;
            await _streamer.Put(chunk, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            pcm.Dispose();
        }
    }

    /// <summary>Signals no more codes will be submitted. Drains the streamer.</summary>
    public void Complete() => _streamer.Complete();

    /// <summary>Async consumer surface — yields <see cref="AudioChunk"/>s as they're decoded.</summary>
    public IAsyncEnumerable<AudioChunk> ReadAllAsync(CancellationToken cancellationToken = default)
        => _streamer.ReadAllAsync(cancellationToken);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _streamer.Dispose();
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }
}
