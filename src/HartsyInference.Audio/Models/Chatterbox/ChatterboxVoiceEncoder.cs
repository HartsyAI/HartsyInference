using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Chatterbox;

/// <summary>Chatterbox voice encoder — a GE2E/Resemblyzer-style 3-layer LSTM over 40-bin mel + a linear
/// projection + ReLU (<c>ve_final_relu</c>) + L2 normalization, producing the 256-d speaker embedding that
/// conditions T3. Reuses the shared <see cref="UnidirectionalLstm"/>; <see cref="EmbedUtterance"/> layers the
/// upstream mel front-end + overlapping-partials protocol on top of the single-window <see cref="Forward"/>.</summary>
public sealed unsafe class ChatterboxVoiceEncoder : IDisposable
{
    private const int NumMels = 40;
    private const int Hidden = 256;
    private const int Layers = 3;

    // voice_encoder.py inference defaults: 160-frame partials, 50% overlap, 80% minimum tail coverage.
    private const int PartialFrames = 160;
    private const int PartialStep = 80;
    private const double MinCoverage = 0.8;

    private readonly UnidirectionalLstm _lstm;
    private Tensor? _projW, _projB;
    private int _disposed;

    public ChatterboxVoiceEncoder() => _lstm = new UnidirectionalLstm(NumMels, Hidden, Layers);

    public int EmbeddingDim => Hidden;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        _lstm.LoadWeights(w, $"{prefix}lstm");
        _projW = WhisperOps.EnsureF32(w[$"{prefix}proj.weight"]);
        _projB = w.TryGetValue($"{prefix}proj.bias", out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
    }

    /// <summary>Embeds a 40-bin mel <c>[1, T, 40]</c> → L2-normalized 256-d speaker embedding <c>[1, 256]</c>
    /// (ReLU'd projection of the final LSTM hidden state, per <c>ve_final_relu=True</c>).</summary>
    public Tensor Forward(IBackend backend, Tensor mel, int t)
    {
        Tensor seq = _lstm.Forward(backend, mel, 1, t);     // [1, T, Hidden]
        Tensor last = new(new TensorShape(1, 1, Hidden), DType.F32);
        Buffer.MemoryCopy((float*)seq.DataPointer + (long)(t - 1) * Hidden, (void*)last.DataPointer, Hidden * 4, Hidden * 4);
        seq.Dispose();
        Tensor proj = WhisperOps.ProjectLinear(backend, last, _projW!, _projB, 1, 1, Hidden, Hidden);
        last.Dispose();

        float* p = (float*)proj.DataPointer;
        double norm = 0;
        for (int i = 0; i < Hidden; i++)
        {
            if (p[i] < 0f) p[i] = 0f;
            norm += (double)p[i] * p[i];
        }
        float inv = (float)(1.0 / (Math.Sqrt(norm) + 1e-8));
        for (int i = 0; i < Hidden; i++) p[i] *= inv;
        return proj.Reshape(new TensorShape(1, Hidden));
    }

    /// <summary>Embeds a full 16 kHz reference utterance → L2-normalized <c>[1, 256]</c> speaker embedding,
    /// replicating upstream <c>VoiceEncoder.inference</c>: raw power-mel front-end (librosa n_fft 400 / hop 160 /
    /// Slaney 40 bins / fmax 8 kHz / <c>mel_type="amp"</c>, no log), sliced into 160-frame partials at 50%
    /// overlap (zero-padded to full coverage when the ≥80%-covered tail earns an extra window), each embedded by
    /// <see cref="Forward"/>, then mean-pooled and re-normalized.</summary>
    public Tensor EmbedUtterance(IBackend backend, ReadOnlySpan<float> audio16k)
    {
        if (audio16k.IsEmpty) throw new ArgumentException("Reference audio must be non-empty.", nameof(audio16k));
        MelSpectrogramExtractor extractor = new(new MelSpectrogramExtractor.Config(
            SampleRate: 16_000,
            NFft: 400,
            WinLength: 400,
            HopLength: 160,
            NMels: NumMels,
            Fmin: 0.0,
            Fmax: 8_000.0,
            Norm: MelSpectrogramExtractor.Normalization.None,
            DropLastStftFrame: false,
            LogBase: MelSpectrogramExtractor.LogBase.None,
            LogFloor: null,
            DynamicRangeDb: 0f,
            NormOffset: 0f,
            NormScale: 1f,
            PowerSpectrum: true,
            Center: true));
        float[,] mel = extractor.Compute(audio16k);          // [40, frames]
        int frames = mel.GetLength(1);

        // get_num_wins: n_wins = (frames - 160 + 80) / 80; the remainder tail earns one more window when it
        // covers ≥ 80% of a partial (counting the 80 frames shared with the previous window).
        int span = Math.Max(frames - PartialFrames + PartialStep, 0);
        int nWins = span / PartialStep;
        int remainder = span % PartialStep;
        if (nWins == 0 || (remainder + (PartialFrames - PartialStep)) / (double)PartialFrames >= MinCoverage) nWins++;
        int target = PartialFrames + PartialStep * (nWins - 1);

        double[] sum = new double[Hidden];
        for (int wi = 0; wi < nWins; wi++)
        {
            Tensor partial = new(new TensorShape(1, PartialFrames, NumMels), DType.F32);
            Span<float> d = partial.AsSpan<float>();
            for (int f = 0; f < PartialFrames; f++)
            {
                int src = wi * PartialStep + f;
                for (int m = 0; m < NumMels; m++)
                    d[f * NumMels + m] = src < Math.Min(frames, target) ? mel[m, src] : 0f;
            }
            Tensor emb = Forward(backend, partial, PartialFrames);
            partial.Dispose();
            float* ep = (float*)emb.DataPointer;
            for (int i = 0; i < Hidden; i++) sum[i] += ep[i];
            emb.Dispose();
        }

        Tensor result = new(new TensorShape(1, Hidden), DType.F32);
        float* rp = (float*)result.DataPointer;
        double norm = 0;
        for (int i = 0; i < Hidden; i++)
        {
            double mean = sum[i] / nWins;
            rp[i] = (float)mean;
            norm += mean * mean;
        }
        float inv = (float)(1.0 / (Math.Sqrt(norm) + 1e-8));
        for (int i = 0; i < Hidden; i++) rp[i] *= inv;
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _lstm.EnumerateWeights()) yield return t;
        if (_projW is not null) yield return _projW;
        if (_projB is not null) yield return _projB;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
