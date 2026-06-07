using System.Diagnostics;
using SharpInference.Audio.Models.SparkTts;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;

namespace SharpInference.Audio.Pipelines;

/// <summary>Spark-TTS-0.5B text-to-speech pipeline. The Qwen2.5-0.5B LM emits the unified global +
/// semantic BiCodec token sequence; BiCodec decodes it to a 16 kHz waveform. Token-IDs-in convention
/// like the other audio pipelines — the caller builds the chat-templated prompt (task + content +
/// optional control/attribute tokens, or the global+semantic prompt tokens of a reference clip for
/// zero-shot cloning) with the Qwen tokenizer.</summary>
public sealed class SparkTtsPipeline : IDisposable
{
    private readonly SparkTtsConfig _cfg;
    private readonly SparkTtsLm _lm;
    private readonly BiCodecDecoder _bicodec;
    private int _disposed;

    public SparkTtsPipeline(SparkTtsConfig cfg, SparkTtsLm lm, BiCodecDecoder bicodec)
    {
        _cfg = cfg;
        _lm = lm;
        _bicodec = bicodec;
    }

    /// <summary>Synthesizes a 16 kHz waveform from the chat-templated prompt token IDs.</summary>
    public float[] Synthesize(IBackend backend, int[] promptTokenIds, int maxTokens = 1500, int seed = 0)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();

        (List<int> global, List<int> semantic) = _lm.GenerateAudioTokens(backend, promptTokenIds, maxTokens, seed);
        Logs.Info($"Spark-TTS: LM emitted {global.Count} global + {semantic.Count} semantic tokens in {sw.ElapsedMilliseconds}ms.");

        float[] audio = _bicodec.Decode(backend, global, semantic);
        sw.Stop();
        Logs.Info($"Spark-TTS synthesis complete: {audio.Length} samples ({audio.Length / (double)_cfg.SampleRate:F2}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
        _bicodec.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(SparkTtsPipeline));
    }
}
