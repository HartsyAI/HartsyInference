using System.Diagnostics;
using HartsyInference.Audio.Models.Codecs.XCodec;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>YuE full-song music pipeline. Stage-1 LLaMA-2 emits the interleaved vocal+accompaniment
/// codebook-0 streams; Stage-2 (<see cref="YueStage2Lm"/>, optional) upsamples each track to the full 8
/// residual codebooks; X-Codec decodes to 16 kHz. Token-IDs-in: the caller tokenizes lyrics + genre tags
/// (with `[verse]/[chorus]` structure markers) via the YuE tokenizer. Reuses `Qwen2Model` (both stages) +
/// the built `XCodec` decoder.
///
/// <para>When <c>stage2</c> is null the pipeline decodes vocal cb0 only (low-quality single-codebook);
/// when supplied it runs the full two-stage path and mixes the vocal + accompaniment stems (equal-gain
/// sum, upstream <c>infer.py</c>).</para></summary>
public sealed unsafe class YuePipeline : IDisposable
{
    private readonly YueConfig _cfg;
    private readonly YueStage1Lm _stage1;
    private readonly YueStage2Lm? _stage2;
    private readonly XCodec _xcodec;
    private int _disposed;

    public YuePipeline(YueConfig cfg, YueStage1Lm stage1, XCodec xcodec, YueStage2Lm? stage2 = null)
    {
        _cfg = cfg;
        _stage1 = stage1;
        _stage2 = stage2;
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

        // Pin the (Q8_0) Stage-1 weights resident up front. The audio stack otherwise relies on lazy auto-promote,
        // whose headroom gate loses to the transient-upload pool for a 7B — the weights never promote and stream
        // from host EVERY token (~0.6 s/token). A one-shot preload (7 GB Q8_0 fits a 12 GB card) makes decode read
        // resident weights via the fused quant GEMV. Freed at the Stage-1→Stage-2 boundary below.
        backend.PreloadWeights(_stage1.EnumerateWeights());
        (List<int> vocal, List<int> accomp) = _stage1.GenerateCb0(backend, promptTokenIds, maxFrames, seed,
            temperature, topK, topP, repetitionPenalty, guidanceScale, uncondTokenIds ?? ReadOnlySpan<int>.Empty);
        Logs.Info($"YuE S1: {vocal.Count} frames (vocal+accomp cb0) in {sw.ElapsedMilliseconds}ms.");
        if (vocal.Count == 0) return [];

        // Stage-1 (7B) is done — free its resident weights so Stage-2 (1B) + x-codec get full VRAM instead of
        // streaming against the 7B's cache (same phase-boundary free as HeartMuLa's LM→codec handoff).
        backend.FreeWeights(_stage1.EnumerateWeights());

        float[] audio;
        if (_stage2 is not null)
        {
            // Same rationale: pin Stage-2's weights resident so its 8×/frame ×2-track decode stays on-device.
            backend.PreloadWeights(_stage2.EnumerateWeights());
            // Full path: Stage-2 upsamples each track's cb0 to 8 codebooks, x-codec decodes each, mix (sum).
            System.Span<int> vSpan = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(vocal);
            System.Span<int> aSpan = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(accomp);
            // Batch both tracks (B=2) through Stage-2 in one pass — same codes as two sequential Upsample calls, ~half the wall-clock.
            (int[][] vocalCodes, int[][] accompCodes) = _stage2.UpsampleBoth(backend, vSpan, aSpan);
            Logs.Info($"YuE S2: upsampled {vocal.Count} frames x2 tracks to 8 codebooks in {sw.ElapsedMilliseconds}ms.");
            backend.FreeWeights(_stage2.EnumerateWeights());   // Stage-2 done — free before x-codec decode
            float[] vocalWav = DecodeFull(backend, vocalCodes);
            float[] accompWav = DecodeFull(backend, accompCodes);
            audio = MixSum(vocalWav, accompWav);
        }
        else
        {
            // Stage-1-only fallback: decode vocal cb0 through x-codec with n_q=1 (CodecManipulator("xcodec",0,1)).
            // Index 0 is a valid codeword (not silence) — decode exactly cb0, do NOT zero-fill residuals.
            Tensor audioT = _xcodec.DecodeCb0(backend, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(vocal));
            int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
            audio = new float[n];
            Buffer.MemoryCopy((void*)audioT.DataPointer, System.Runtime.CompilerServices.Unsafe.AsPointer(ref audio[0]), n * 4, n * 4);
            audioT.Dispose();
        }

        sw.Stop();
        Logs.Info($"YuE synthesis complete: {audio.Length} samples ({audio.Length / (double)_cfg.SampleRate:F1}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    /// <summary>Decodes a full <c>[8][T]</c> codebook grid to a 16 kHz mono waveform. Builds the
    /// <c>[n_q=8, B=1, T]</c> I32 grid x-codec expects (upstream <c>codec_model.decode</c> permute layout).</summary>
    private float[] DecodeFull(IBackend backend, int[][] codes)
    {
        int nq = codes.Length;
        int t = codes[0].Length;
        Tensor grid = new(new TensorShape(nq, 1, t), DType.I32);
        int* gp = (int*)grid.DataPointer;
        for (int q = 0; q < nq; q++)
        {
            int[] row = codes[q];
            for (int i = 0; i < t; i++) gp[q * t + i] = row[i];
        }
        Tensor pcm = _xcodec.Decode(backend, grid, batch: 1, tFrames: t);
        grid.Dispose();
        int n = (int)pcm.Shape[pcm.Shape.Rank - 1];
        float[] wav = new float[n];
        Buffer.MemoryCopy((void*)pcm.DataPointer, System.Runtime.CompilerServices.Unsafe.AsPointer(ref wav[0]), n * 4, n * 4);
        pcm.Dispose();
        return wav;
    }

    /// <summary>Mixes two stems by equal-gain sum (upstream <c>infer.py</c>: <c>(vocal + inst) / 1</c>),
    /// clamped to ±0.99 to match the reference <c>save_audio</c> (no rescale).</summary>
    private static float[] MixSum(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        float[] mix = new float[n];
        for (int i = 0; i < n; i++)
        {
            float v = a[i] + b[i];
            mix[i] = v > 0.99f ? 0.99f : (v < -0.99f ? -0.99f : v);
        }
        return mix;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _stage1.Dispose();
        _stage2?.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(YuePipeline));
    }
}
