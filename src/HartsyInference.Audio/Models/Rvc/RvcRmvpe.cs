using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Rvc;

/// <summary>RMVPE (Robust Model for Vocal Pitch Estimation) — the per-frame f0 extractor RVC v2 uses. The model
/// is a DeepUNet over a 128-mel spectrogram (conv encoder with stride-2 downsampling + residual intermediate
/// blocks + a transposed-conv decoder with skip connections) followed by a bidirectional recurrence and a
/// Linear classifier over 360 pitch bins. The 360-bin posterior is decoded to a continuous Hz value per frame
/// via a local weighted average around the argmax bin; bins whose peak confidence is below a threshold are
/// reported as 0 Hz (unvoiced).
///
/// <para>Reuses <see cref="MelSpectrogramExtractor"/> (front-end), <see cref="BiLstm"/> (the bidirectional
/// recurrence — RMVPE's real layer is a GRU; structurally we use the BiLstm sweep, a documented reconcile
/// detail), and the shared <see cref="IBackend"/> Conv2D / GroupNorm / Linear ops.</para>
///
/// <para>Structural scaffold (validation-pending): the cents-grid constants (first bin 32.70 Hz = C1, 20-cent
/// spacing) and the local-average window (±4 bins) are exact; the precise checkpoint key spelling and the
/// GRU-vs-LSTM gate ordering are reconcile items against the downloaded <c>rmvpe.pt</c>.</para></summary>
public sealed unsafe class RvcRmvpe : IDisposable
{
    private readonly RvcRmvpeConfig _cfg;
    private readonly MelSpectrogramExtractor _mel;
    private readonly EncoderLayer[] _encoder;
    private readonly DecoderLayer[] _decoder;
    private readonly ResBlock[] _intermediate;

    private Tensor? _stemW, _stemB, _stemNormW, _stemNormB;
    private BiLstm? _gru;
    private Tensor? _fcW, _fcB;          // Linear(2*gruHidden → 360)
    private readonly float[] _centsMapping;
    private int _disposed;

    public RvcRmvpeConfig Config => _cfg;

    public RvcRmvpe(RvcRmvpeConfig cfg)
    {
        _cfg = cfg;
        _mel = new MelSpectrogramExtractor(cfg.MelConfig());
        _encoder = new EncoderLayer[cfg.EncoderChannels.Count];
        _decoder = new DecoderLayer[cfg.EncoderChannels.Count];
        _intermediate = new ResBlock[cfg.IntermediateBlocks];

        // Pitch bins map to a cents grid: bin k → first_freq * 2^(k * spacing / 1200).
        _centsMapping = new float[cfg.NumBins];
        for (int k = 0; k < cfg.NumBins; k++)
            _centsMapping[k] = cfg.FirstFreqHz * MathF.Pow(2f, k * cfg.CentsPerBin / 1200f);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = string.IsNullOrEmpty(prefix) ? "" : prefix + ".";
        _stemW = WhisperOps.EnsureF32(w[$"{p}stem.conv.weight"]);
        _stemB = TryGet(w, $"{p}stem.conv.bias");
        _stemNormW = WhisperOps.EnsureF32(w[$"{p}stem.norm.weight"]);
        _stemNormB = WhisperOps.EnsureF32(w[$"{p}stem.norm.bias"]);

        int inC = _cfg.StemChannels;
        for (int i = 0; i < _encoder.Length; i++)
        {
            int outC = _cfg.EncoderChannels[i];
            _encoder[i] = new EncoderLayer
            {
                ConvW = WhisperOps.EnsureF32(w[$"{p}encoder.{i}.conv.weight"]),
                ConvB = TryGet(w, $"{p}encoder.{i}.conv.bias"),
                NormW = WhisperOps.EnsureF32(w[$"{p}encoder.{i}.norm.weight"]),
                NormB = WhisperOps.EnsureF32(w[$"{p}encoder.{i}.norm.bias"]),
                OutC = outC,
            };
            inC = outC;
        }
        for (int i = 0; i < _intermediate.Length; i++)
        {
            string bp = $"{p}intermediate.{i}";
            _intermediate[i] = new ResBlock
            {
                Conv1W = WhisperOps.EnsureF32(w[$"{bp}.conv1.weight"]),
                Conv1B = TryGet(w, $"{bp}.conv1.bias"),
                Norm1W = WhisperOps.EnsureF32(w[$"{bp}.norm1.weight"]),
                Norm1B = WhisperOps.EnsureF32(w[$"{bp}.norm1.bias"]),
                Conv2W = WhisperOps.EnsureF32(w[$"{bp}.conv2.weight"]),
                Conv2B = TryGet(w, $"{bp}.conv2.bias"),
                Norm2W = WhisperOps.EnsureF32(w[$"{bp}.norm2.weight"]),
                Norm2B = TryGet(w, $"{bp}.norm2.bias"),
            };
        }
        for (int i = 0; i < _decoder.Length; i++)
        {
            int outC = i == _decoder.Length - 1 ? _cfg.StemChannels : _cfg.EncoderChannels[_decoder.Length - 2 - i];
            _decoder[i] = new DecoderLayer
            {
                UpW = WhisperOps.EnsureF32(w[$"{p}decoder.{i}.up.weight"]),
                UpB = TryGet(w, $"{p}decoder.{i}.up.bias"),
                NormW = WhisperOps.EnsureF32(w[$"{p}decoder.{i}.norm.weight"]),
                NormB = WhisperOps.EnsureF32(w[$"{p}decoder.{i}.norm.bias"]),
                OutC = outC,
            };
            inC = outC;
        }

        int gruIn = _cfg.StemChannels * _cfg.MelBins;
        _gru = new BiLstm(gruIn, _cfg.GruHidden);
        _gru.LoadWeights(w, $"{p}gru");
        _fcW = WhisperOps.EnsureF32(w[$"{p}fc.weight"]);
        _fcB = TryGet(w, $"{p}fc.bias");
    }

    /// <summary>Extracts per-frame f0 in Hz (0 = unvoiced) from 16 kHz mono PCM.</summary>
    public float[] ExtractF0(IBackend backend, float[] pcm16k)
    {
        if (_stemW is null) throw new InvalidOperationException("RvcRmvpe weights not loaded.");

        // Mel front-end → [melBins, frames] → image [1, 1, melBins, frames].
        float[,] mel2d = _mel.Compute(pcm16k);
        int melBins = _cfg.MelBins;
        int frames = mel2d.GetLength(1);
        Tensor img = new(new TensorShape(1, 1, melBins, frames), DType.F32);
        float* ip = (float*)img.DataPointer;
        for (int m = 0; m < melBins; m++)
            for (int t = 0; t < frames; t++)
                ip[(long)m * frames + t] = mel2d[m, t];

        // Stem.
        Tensor x = new(new TensorShape(1, _cfg.StemChannels, melBins, frames), DType.F32);
        backend.Conv2D(x, img, _stemW!, _stemB, 1, 1, 1, 1);
        img.Dispose();
        NormRelu(backend, x, _stemNormW!, _stemNormB!);

        int h = melBins, wDim = frames;
        // Encoder with stored skips (downsample freq + time by 2 each layer).
        Tensor[] skips = new Tensor[_encoder.Length];
        for (int i = 0; i < _encoder.Length; i++)
        {
            EncoderLayer e = _encoder[i];
            int outH = (h + 2 - 3) / 2 + 1;
            int outW = (wDim + 2 - 3) / 2 + 1;
            Tensor next = new(new TensorShape(1, e.OutC, outH, outW), DType.F32);
            backend.Conv2D(next, x, e.ConvW!, e.ConvB, 2, 2, 1, 1);
            NormRelu(backend, next, e.NormW!, e.NormB!);
            skips[i] = x;       // store pre-downsample tensor as the skip target
            x = next;
            h = outH;
            wDim = outW;
        }

        // Intermediate residual blocks (at the bottleneck).
        foreach (ResBlock r in _intermediate)
        {
            Tensor next = ForwardRes(backend, x, r);
            x.Dispose();
            x = next;
        }

        // Decoder: transposed-conv upsample to match each skip, add the skip, norm + relu.
        for (int i = 0; i < _decoder.Length; i++)
        {
            DecoderLayer d = _decoder[i];
            Tensor skip = skips[_decoder.Length - 1 - i];
            int targetH = (int)skip.Shape[2];
            int targetW = (int)skip.Shape[3];
            Tensor up = new(new TensorShape(1, d.OutC, targetH, targetW), DType.F32);
            backend.ConvTranspose2d(up, x, d.UpW!, d.UpB, 2, 2, 0, 0);
            x.Dispose();
            // Crop/clamp: ConvTranspose output length may exceed target; we allocated to target so the op
            // writes only the valid region. Add the skip when channel counts line up.
            if ((int)skip.Shape[1] == d.OutC)
            {
                Tensor sum = new(up.Shape, DType.F32);
                backend.Add(sum, up, skip);
                up.Dispose();
                up = sum;
            }
            skip.Dispose();
            NormRelu(backend, up, d.NormW!, d.NormB!);
            x = up;
            h = targetH;
            wDim = targetW;
        }

        // x is now [1, stemChannels, melBins, frames]; flatten (channels, freq) → feature per frame.
        int featDim = _cfg.StemChannels * melBins;
        Tensor seq = new(new TensorShape(1, frames, featDim), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* sq = (float*)seq.DataPointer;
        int curC = (int)x.Shape[1], curH = (int)x.Shape[2], curW = (int)x.Shape[3];
        for (int t = 0; t < frames; t++)
            for (int c = 0; c < _cfg.StemChannels; c++)
                for (int m = 0; m < melBins; m++)
                {
                    float v = (c < curC && m < curH && t < curW)
                        ? xp[((long)c * curH + m) * curW + t]
                        : 0f;
                    sq[((long)t) * featDim + (long)c * melBins + m] = v;
                }
        x.Dispose();

        // Bidirectional recurrence → [1, frames, 2*gruHidden].
        Tensor rec = _gru!.Forward(backend, seq, 1, frames);
        seq.Dispose();

        // Linear → 360 bins → sigmoid posterior.
        Tensor logits = WhisperOps.ProjectLinear(backend, rec, _fcW!, _fcB, 1, frames, 2 * _cfg.GruHidden, _cfg.NumBins);
        rec.Dispose();
        float* lp = (float*)logits.DataPointer;
        for (long i = 0; i < logits.ElementCount; i++) lp[i] = 1f / (1f + MathF.Exp(-lp[i]));

        // Local-weighted-average decode to Hz.
        float[] f0 = DecodeF0(lp, frames);
        logits.Dispose();
        return f0;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_stemW, _stemB, _stemNormW, _stemNormB, _fcW, _fcB];
        foreach (Tensor? x in top) if (x is not null) yield return x;
        foreach (EncoderLayer e in _encoder) foreach (Tensor t in e.All()) yield return t;
        foreach (ResBlock r in _intermediate) foreach (Tensor t in r.All()) yield return t;
        foreach (DecoderLayer d in _decoder) foreach (Tensor t in d.All()) yield return t;
        if (_gru is not null) foreach (Tensor t in _gru.EnumerateWeights()) yield return t;
    }

    /// <summary>Decodes the <c>[frames, 360]</c> posterior to Hz: per frame, take the argmax bin, average the
    /// cents-grid frequencies in a ±window around it weighted by their posterior, and gate to 0 when the peak
    /// confidence is below the voicing threshold.</summary>
    private float[] DecodeF0(float* posterior, int frames)
    {
        int bins = _cfg.NumBins;
        int win = _cfg.LocalAverageWindow;
        float[] f0 = new float[frames];
        for (int t = 0; t < frames; t++)
        {
            long rowOff = (long)t * bins;
            int argmax = 0;
            float peak = posterior[rowOff];
            for (int k = 1; k < bins; k++)
            {
                float v = posterior[rowOff + k];
                if (v > peak) { peak = v; argmax = k; }
            }
            if (peak < _cfg.VoicingThreshold) { f0[t] = 0f; continue; }
            int lo = Math.Max(0, argmax - win);
            int hi = Math.Min(bins - 1, argmax + win);
            float wsum = 0f, fsum = 0f;
            for (int k = lo; k <= hi; k++)
            {
                float wgt = posterior[rowOff + k];
                wsum += wgt;
                fsum += wgt * _centsMapping[k];
            }
            f0[t] = wsum > 0 ? fsum / wsum : 0f;
        }
        return f0;
    }

    private Tensor ForwardRes(IBackend backend, Tensor x, ResBlock r)
    {
        Tensor h1 = new(x.Shape, DType.F32);
        backend.Conv2D(h1, x, r.Conv1W!, r.Conv1B, 1, 1, 1, 1);
        NormRelu(backend, h1, r.Norm1W!, r.Norm1B!);
        Tensor h2 = new(x.Shape, DType.F32);
        backend.Conv2D(h2, h1, r.Conv2W!, r.Conv2B, 1, 1, 1, 1);
        h1.Dispose();
        if (r.Norm2W is not null && r.Norm2B is not null)
        {
            Tensor n2 = new(h2.Shape, DType.F32);
            backend.GroupNorm(n2, h2, r.Norm2W, r.Norm2B, _cfg.NormGroups, 1e-5f);
            h2.Dispose();
            h2 = n2;
        }
        Tensor sum = new(x.Shape, DType.F32);
        backend.Add(sum, x, h2);
        h2.Dispose();
        backend.LeakyRelu(sum, sum, 0f);
        return sum;
    }

    private void NormRelu(IBackend backend, Tensor x, Tensor nw, Tensor nb)
    {
        Tensor n = new(x.Shape, DType.F32);
        backend.GroupNorm(n, x, nw, nb, _cfg.NormGroups, 1e-5f);
        backend.LeakyRelu(n, n, 0f);
        backend.CopyTo(x, n);
        n.Dispose();
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private sealed class EncoderLayer
    {
        public Tensor? ConvW, ConvB, NormW, NormB;
        public int OutC;
        public IEnumerable<Tensor> All()
        {
            Tensor?[] a = [ConvW, ConvB, NormW, NormB];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    private sealed class DecoderLayer
    {
        public Tensor? UpW, UpB, NormW, NormB;
        public int OutC;
        public IEnumerable<Tensor> All()
        {
            Tensor?[] a = [UpW, UpB, NormW, NormB];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    private sealed class ResBlock
    {
        public Tensor? Conv1W, Conv1B, Norm1W, Norm1B, Conv2W, Conv2B, Norm2W, Norm2B;
        public IEnumerable<Tensor> All()
        {
            Tensor?[] a = [Conv1W, Conv1B, Norm1W, Norm1B, Conv2W, Conv2B, Norm2W, Norm2B];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }
}
