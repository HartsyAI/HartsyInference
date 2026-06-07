using System.Diagnostics;
using SharpInference.Audio.Dsp;
using SharpInference.Audio.Models.Codecs.Mimi;
using SharpInference.Audio.Models.Csm;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Pipelines;

/// <summary>Sesame CSM conversational TTS pipeline. The dual-transformer <see cref="CsmModel"/> generates
/// 8-codebook Mimi frames one 80 ms frame at a time (backbone → codebook 0, decoder → codebooks 1..7);
/// the built Mimi codec decodes them to 24 kHz audio. Token-IDs-in: the caller Llama-3-tokenizes the
/// text (and may prepend a `Segment(speaker, text, audio)` conversation history as context).</summary>
public sealed unsafe class CsmPipeline : IDisposable
{
    private readonly CsmConfig _cfg;
    private readonly CsmModel _model;
    private readonly Mimi _mimi;
    private int _disposed;

    public CsmPipeline(CsmConfig cfg, CsmModel model, Mimi mimi)
    {
        _cfg = cfg;
        _model = model;
        _mimi = mimi;
    }

    /// <summary>Synthesizes 24 kHz audio for the Llama-tokenized <paramref name="textTokenIds"/>.</summary>
    public float[] Synthesize(IBackend backend, int[] textTokenIds, int maxFrames = 1024, int seed = 0)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        uint rng = DeterministicRng.Seed(seed);

        // Text context embeddings (the audio context is built up as frames are generated).
        int bh = _cfg.Backbone.HiddenSize;
        List<int[]> frames = new(Math.Min(maxFrames, 256));
        for (int step = 0; step < maxFrames; step++)
        {
            Tensor context = BuildContext(textTokenIds, frames, bh);
            int[] frame = _model.GenerateFrame(backend, context, ref rng);
            context.Dispose();
            if (frame[0] == _cfg.AudioEosToken) break;
            frames.Add(frame);
        }
        Logs.Info($"CSM: generated {frames.Count} frames ({frames.Count * _cfg.FrameSamples / (double)_cfg.SampleRate:F2}s) in {sw.ElapsedMilliseconds}ms.");

        if (frames.Count == 0) return [];

        // Assemble [1, numCodebooks, T] codes → Mimi decode.
        int t = frames.Count;
        Tensor codes = new(new TensorShape(1, _cfg.NumCodebooks, t), DType.F32);
        float* cp = (float*)codes.DataPointer;
        for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
            for (int j = 0; j < t; j++) cp[(long)cb * t + j] = frames[j][cb];
        Tensor audioT = _mimi.Decode(backend, codes, batch: 1, tFrames: t);
        codes.Dispose();

        int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
        float[] audio = new float[n];
        Buffer.MemoryCopy((void*)audioT.DataPointer, System.Runtime.CompilerServices.Unsafe.AsPointer(ref audio[0]), n * 4, n * 4);
        audioT.Dispose();

        sw.Stop();
        Logs.Info($"CSM synthesis complete: {audio.Length} samples in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    /// <summary>Concatenates the text-token embeddings + the per-frame summed audio embeddings into the
    /// running context <c>[1, T, backboneHidden]</c>.</summary>
    private Tensor BuildContext(int[] textTokenIds, List<int[]> frames, int bh)
    {
        int total = textTokenIds.Length + frames.Count;
        Tensor context = new(new TensorShape(1, total, bh), DType.F32);
        float* dst = (float*)context.DataPointer;
        int row = 0;
        foreach (int tok in textTokenIds)
        {
            using Tensor e = _model.EmbedText(tok);
            Buffer.MemoryCopy((void*)e.DataPointer, dst + (long)row++ * bh, bh * 4, bh * 4);
        }
        foreach (int[] f in frames)
        {
            using Tensor e = _model.EmbedAudioFrame(f);
            Buffer.MemoryCopy((void*)e.DataPointer, dst + (long)row++ * bh, bh * 4, bh * 4);
        }
        return context;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _model.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(CsmPipeline));
    }
}
