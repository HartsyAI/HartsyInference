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

        // apply_model: split into `segment`-length chunks with `overlap`, run each (the model pads short chunks up
        // to the training segment via use_train_segment), and triangular-weighted overlap-add across chunks.
        int seg = _cfg.SegmentSamples;
        int stride = Math.Max(1, (int)((1.0 - _cfg.Overlap) * seg));
        float[] weight = BuildTransitionWeight(seg);

        float[][] acc = new float[srcs * channels][];
        for (int i = 0; i < acc.Length; i++) acc[i] = new float[length];
        float[] wsum = new float[length];

        for (int offset = 0; offset < length; offset += stride)
        {
            int clen = Math.Min(seg, length - offset);
            Tensor chunkStems = RunChunk(backend, stereoLeft, stereoRight, offset, clen, channels, srcs, seg, length);
            float* cs = (float*)chunkStems.DataPointer;
            for (int s = 0; s < srcs; s++)
                for (int c = 0; c < channels; c++)
                {
                    float[] dst = acc[s * channels + c];
                    long b = (((long)s * channels) + c) * clen;
                    for (int j = 0; j < clen; j++) dst[offset + j] += weight[j] * cs[b + j];
                }
            for (int j = 0; j < clen; j++) wsum[offset + j] += weight[j];
            chunkStems.Dispose();
            if (offset + clen >= length) break;
        }

        (float[] Left, float[] Right)[] result = new (float[], float[])[srcs];
        for (int s = 0; s < srcs; s++)
        {
            float[] left = new float[length];
            float[] right = new float[length];
            float[] al = acc[s * channels + 0];
            float[] ar = acc[s * channels + (channels > 1 ? 1 : 0)];
            for (int j = 0; j < length; j++)
            {
                float ws = wsum[j] > 1e-8f ? wsum[j] : 1f;
                left[j] = al[j] / ws;
                right[j] = ar[j] / ws;
            }
            result[s] = (left, right);
        }
        return result;
    }

    /// <summary>Runs the model on one segment covering <c>[offset, offset+clen)</c>, replicating demucs
    /// <c>TensorChunk.padded</c> + <c>center_trim</c>: the segment is CENTER-padded up to the training length
    /// (<paramref name="seg"/>), pulling in surrounding signal as context where available (zeros only at the true
    /// signal edges), the model runs, and the output is center-trimmed back to the <paramref name="clen"/> region.
    /// Returns <c>[1, srcs, channels, clen]</c>.</summary>
    private Tensor RunChunk(IBackend backend, float[] left, float[] right, int offset, int clen, int channels, int srcs, int seg, int totalLen)
    {
        int delta = seg - clen;
        int start = offset - delta / 2;               // center the segment (demucs TensorChunk.padded)
        int correctStart = Math.Max(0, start);
        int correctEnd = Math.Min(totalLen, start + seg);
        int padLeft = correctStart - start;
        Tensor wav = new(new TensorShape(1, channels, seg), DType.F32);
        float* wp = (float*)wav.DataPointer;
        for (int j = correctStart; j < correctEnd; j++)
        {
            int p = padLeft + (j - correctStart);
            wp[p] = left[j];
            wp[(long)seg + p] = right[j];
        }
        Tensor full = _model.Forward(backend, wav, seg);   // [1, srcs, channels, seg]
        wav.Dispose();

        int trimLeft = delta / 2;                     // center_trim(out, clen)
        Tensor cropped = new(new TensorShape(1, srcs, channels, clen), DType.F32);
        float* fp = (float*)full.DataPointer; float* cp = (float*)cropped.DataPointer;
        for (int s = 0; s < srcs; s++)
            for (int c = 0; c < channels; c++)
            {
                long src = (((long)s * channels) + c) * seg + trimLeft;
                long dst = (((long)s * channels) + c) * clen;
                for (int j = 0; j < clen; j++) cp[dst + j] = fp[src + j];
            }
        full.Dispose();
        return cropped;
    }

    /// <summary>demucs transition weight over a segment: triangular ramp-up/down, normalized to [0,1]
    /// (<c>transition_power=1</c>): <c>[1..seg/2, seg-seg/2..1] / max</c>.</summary>
    private static float[] BuildTransitionWeight(int seg)
    {
        float[] w = new float[seg];
        int half = seg / 2;
        for (int i = 0; i < half; i++) w[i] = i + 1;
        for (int i = half; i < seg; i++) w[i] = seg - i;   // seg-half down to 1
        float max = 0f; foreach (float v in w) if (v > max) max = v;
        for (int i = 0; i < seg; i++) w[i] /= max;
        return w;
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
