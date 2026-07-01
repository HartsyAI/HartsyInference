using System.Diagnostics;
using HartsyInference.Audio.Models.Codecs.XCodec;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>YuE full-song music pipeline. Stage-1 LLaMA-2 emits the interleaved vocal+accompaniment
/// codebook-0 streams; (Stage-2 upsamples to 8 residual codebooks — deferred scaffold); X-Codec decodes
/// to 16 kHz. Token-IDs-in: the caller tokenizes lyrics + genre tags (with `[verse]/[chorus]` structure
/// markers) via the YuE tokenizer. Reuses `Qwen2Model` (Stage-1) + the built `XCodec` decoder.</summary>
public sealed unsafe class YuePipeline : IDisposable
{
    private readonly YueConfig _cfg;
    private readonly YueStage1Lm _stage1;
    private readonly XCodec _xcodec;
    private int _disposed;

    public YuePipeline(YueConfig cfg, YueStage1Lm stage1, XCodec xcodec)
    {
        _cfg = cfg;
        _stage1 = stage1;
        _xcodec = xcodec;
    }

    /// <summary>Synthesizes a 16 kHz song from the tokenized lyric+genre prompt. Returns the vocal track
    /// audio (accompaniment mixing + the Stage-2 residual upsampler are deferred — see checklist). YuE's
    /// generation params are exposed per-call (default to config): <paramref name="temperature"/>,
    /// <paramref name="topK"/>, <paramref name="topP"/>, <paramref name="repetitionPenalty"/>, and classifier-free
    /// guidance (<paramref name="guidanceScale"/>) when a negative prompt <paramref name="uncondTokenIds"/> is given.</summary>
    public float[] Synthesize(IBackend backend, int[] promptTokenIds, int maxFrames = 3000, int seed = 0,
        float? temperature = null, int? topK = null, float? topP = null, float? repetitionPenalty = null,
        float? guidanceScale = null, int[]? uncondTokenIds = null)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();

        (List<int> vocal, List<int> accomp) = _stage1.GenerateCb0(backend, promptTokenIds, maxFrames, seed,
            temperature, topK, topP, repetitionPenalty, guidanceScale, uncondTokenIds ?? ReadOnlySpan<int>.Empty);
        Logs.Info($"YuE S1: {vocal.Count} frames (vocal+accomp cb0) in {sw.ElapsedMilliseconds}ms.");
        if (vocal.Count == 0) return [];

        // YuE Stage-1 emits only the vocal codebook-0 stream; upstream decodes it through x-codec with n_q=1
        // (CodecManipulator("xcodec", 0, 1)). Codebook 0's index 0 is a valid codeword (not silence), so we
        // must NOT zero-fill the residual codebooks — decode exactly cb0. Stage-2 (residual upsample) is deferred.
        Tensor audioT = _xcodec.DecodeCb0(backend, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(vocal));

        int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
        float[] audio = new float[n];
        Buffer.MemoryCopy((void*)audioT.DataPointer, System.Runtime.CompilerServices.Unsafe.AsPointer(ref audio[0]), n * 4, n * 4);
        audioT.Dispose();

        _ = accomp;   // accompaniment track + mix is part of the deferred Stage-2 path.
        sw.Stop();
        Logs.Info($"YuE synthesis complete: {audio.Length} samples ({audio.Length / (double)_cfg.SampleRate:F1}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _stage1.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(YuePipeline));
    }
}
