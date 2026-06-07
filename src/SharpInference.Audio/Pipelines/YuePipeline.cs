using System.Diagnostics;
using SharpInference.Audio.Models.Codecs.XCodec;
using SharpInference.Audio.Models.Music;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Pipelines;

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
    /// audio (accompaniment mixing + the Stage-2 residual upsampler are deferred — see checklist).</summary>
    public float[] Synthesize(IBackend backend, int[] promptTokenIds, int maxFrames = 3000, int seed = 0)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();

        (List<int> vocal, List<int> accomp) = _stage1.GenerateCb0(backend, promptTokenIds, maxFrames, seed);
        Logs.Info($"YuE S1: {vocal.Count} frames (vocal+accomp cb0) in {sw.ElapsedMilliseconds}ms.");
        if (vocal.Count == 0) return [];

        // Build the 8-codebook code grid for X-Codec. Stage-2 (the residual upsampler) is the deferred
        // piece — codebooks 1..7 are zero-filled here, so this decodes the cb0 (semantic) content only.
        int t = vocal.Count;
        Tensor codes = new(new TensorShape(1, _cfg.NumCodebooks, t), DType.F32);
        float* cp = (float*)codes.DataPointer;
        for (int j = 0; j < t; j++) cp[j] = vocal[j];   // codebook 0 = Stage-1 vocal stream

        Tensor audioT = _xcodec.Decode(backend, codes, batch: 1, tFrames: t);
        codes.Dispose();

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
