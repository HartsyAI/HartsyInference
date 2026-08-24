using System.Diagnostics;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.Csm;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Sesame CSM conversational TTS pipeline. The dual-transformer <see cref="CsmModel"/> generates
/// 8-codebook Mimi frames one 80 ms frame at a time (backbone → codebook 0, decoder → codebooks 1..7);
/// the built Mimi codec decodes them to 24 kHz audio. Token-IDs-in: the caller Llama-3-tokenizes the
/// text (and may prepend a `Segment(speaker, text, audio)` conversation history as context).</summary>
public sealed unsafe class CsmPipeline(CsmConfig cfg, CsmModel model, Mimi mimi) : IDisposable
{
    private readonly CsmConfig _cfg = cfg;
    private readonly CsmModel _model = model;
    private readonly Mimi _mimi = mimi;
    private int _disposed;

    /// <summary>Synthesizes 24 kHz audio for the Llama-tokenized <paramref name="textTokenIds"/>. When
    /// <paramref name="onFrame"/> is supplied, it is invoked with each genuine (non-EOS-sentinel) frame right
    /// after it's added — the same shape as <c>MoshiTtsGenerator.Generate</c>'s <c>onValidFrame</c> — so a
    /// streaming caller can batch frames into <see cref="Mimi.DecodeStreaming"/> calls as they're produced
    /// instead of waiting for the whole utterance. Passing null (the default) changes nothing about this method's
    /// behavior.</summary>
    public float[] Synthesize(IBackend backend, int[] textTokenIds, int maxFrames = 1024, int seed = 0, Action<int[]>? onFrame = null)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        uint rng = DeterministicRng.Seed(seed);

        // Text context embeddings (the audio context is built up as frames are generated). The backbone KV cache
        // persists across frames: the first step feeds the whole text prefix, later steps append just the previous
        // frame's summed audio embedding (one row) — O(1) per frame instead of re-scanning the growing context.
        int bh = _cfg.Backbone.HiddenSize;
        List<int[]> frames = new(Math.Min(maxFrames, 256));
        List<int[]> noFrames = new(0);
        using CsmModel.DecodeSession session = _model.CreateSession(textTokenIds.Length + maxFrames + 4, 0, useCfg: false);
        for (int step = 0; step < maxFrames; step++)
        {
            Tensor condNew = step == 0 ? BuildContext(textTokenIds, noFrames, bh) : _model.EmbedAudioFrame(frames[step - 1]);
            int[] frame = _model.StepFrame(backend, session, condNew, ref rng);
            condNew.Dispose();
            // Upstream's stop condition (generator.py): the frame is EOS when EVERY codebook equals
            // CodebookEosToken (0) — the training context appends an all-zero frame at each segment's end.
            // Discard that frame (it's a sentinel, not audio) and stop.
            bool isEos = true;
            for (int cb = 0; cb < frame.Length; cb++)
                if (frame[cb] != _cfg.CodebookEosToken) { isEos = false; break; }
            if (isEos) break;
            frames.Add(frame);
            onFrame?.Invoke(frame);
        }
        Logs.Info($"CSM: generated {frames.Count} frames ({frames.Count * _cfg.FrameSamples / (double)_cfg.SampleRate:F2}s) in {sw.ElapsedMilliseconds}ms.");

        if (frames.Count == 0) return [];

        // Assemble [1, numCodebooks, T] codes → Mimi decode. Mimi.Decode/MimiSplitRvq.Decode read codes as
        // Int32 (raw codebook indices), not F32 — an F32 tensor here reinterpreted as int bit patterns turns
        // small code values into huge out-of-range embed-table offsets (AccessViolationException on decode).
        int t = frames.Count;
        Tensor codes = new(new TensorShape(1, _cfg.NumCodebooks, t), DType.I32);
        int* cp = (int*)codes.DataPointer;
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
