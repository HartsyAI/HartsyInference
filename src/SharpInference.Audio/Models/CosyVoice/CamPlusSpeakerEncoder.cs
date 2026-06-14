using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.CosyVoice;

/// <summary>CAM++ speaker encoder — a frozen D-TDNN feature extractor producing the 192-dim
/// L2-normalized speaker embedding CosyVoice feeds to its flow-matching stage (never to the LM in CV2).
/// Input is an 80-bin log-mel <c>[1, 80, T]</c> from a ≥3 s reference clip; output is <c>[1, 192]</c>.
///
/// <para><b>Scaffold note:</b> this implements the encoder's I/O contract plus the temporal-context
/// TDNN → statistics-pooling → FC → L2-norm shape, which is the part the rest of the pipeline depends
/// on. The exact CAM++ topology (FCM frontend + densely-connected D-TDNN blocks with Context-Aware
/// Masking) is checkpoint-gated — <c>campplus.onnx</c> is also available as an ONNX fallback. Callers
/// that already hold a precomputed 192-d embedding can bypass this encoder entirely.</para></summary>
public sealed unsafe class CamPlusSpeakerEncoder : IDisposable
{
    private readonly int _embedDim;
    private readonly int _melBins;
    private int _disposed;

    // TDNN frontend (dilated Conv1d + ReLU) → statistics pooling → two FC layers → 192.
    private Tensor?[] _tdnnW = new Tensor[5];
    private Tensor?[] _tdnnB = new Tensor[5];
    private static readonly int[] _tdnnChannels = [512, 512, 512, 512, 1536];
    private static readonly int[] _tdnnKernels = [5, 3, 3, 1, 1];
    private static readonly int[] _tdnnDilations = [1, 2, 3, 1, 1];
    private Tensor? _fc1W, _fc1B;       // 2*1536 → 192
    private Tensor? _fc2W, _fc2B;       // 192 → 192

    public CamPlusSpeakerEncoder(int melBins = 80, int embedDim = 192)
    {
        _melBins = melBins;
        _embedDim = embedDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        for (int i = 0; i < 5; i++)
        {
            _tdnnW[i] = WhisperOps.EnsureF32(w[$"{p}tdnn.{i}.weight"]);
            _tdnnB[i] = WhisperOps.EnsureF32(w[$"{p}tdnn.{i}.bias"]);
        }
        _fc1W = WhisperOps.EnsureF32(w[$"{p}fc1.weight"]);
        _fc1B = WhisperOps.EnsureF32(w[$"{p}fc1.bias"]);
        _fc2W = WhisperOps.EnsureF32(w[$"{p}fc2.weight"]);
        _fc2B = WhisperOps.EnsureF32(w[$"{p}fc2.bias"]);
    }

    /// <summary>Extracts the 192-dim L2-normalized embedding from a reference mel <c>[1, 80, T]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor mel)
    {
        if (_fc1W is null) throw new InvalidOperationException("CamPlusSpeakerEncoder weights not loaded.");
        Tensor x = mel;
        bool ownsX = false;
        int inCh = _melBins;
        for (int i = 0; i < 5; i++)
        {
            int t = (int)x.Shape[2];
            int outCh = _tdnnChannels[i];
            int pad = (_tdnnKernels[i] - 1) * _tdnnDilations[i] / 2;
            Tensor nx = new(new TensorShape(1, outCh, t), DType.F32);
            backend.Conv1d(nx, x, _tdnnW![i]!, _tdnnB![i], stride: 1, padLeft: pad, padRight: pad, dilation: _tdnnDilations[i], groups: 1);
            if (ownsX) x.Dispose();
            backend.LeakyRelu(nx, nx, 0f);     // ReLU (slope 0)
            x = nx;
            ownsX = true;
            inCh = outCh;
        }

        // Statistics pooling: per-channel mean + std over time → [1, 2*C].
        Tensor stats = StatsPool(x, inCh);
        if (ownsX) x.Dispose();

        Tensor e1 = WhisperOps.ProjectLinear(backend, stats, _fc1W!, _fc1B, 1, 1, 2 * inCh, _embedDim);
        stats.Dispose();
        Tensor e2 = WhisperOps.ProjectLinear(backend, e1, _fc2W!, _fc2B, 1, 1, _embedDim, _embedDim);
        e1.Dispose();

        // L2 normalize → [1, 192].
        Tensor outE = new(new TensorShape(1, _embedDim), DType.F32);
        float* sp = (float*)e2.DataPointer;
        float* op = (float*)outE.DataPointer;
        float norm = 0f;
        for (int i = 0; i < _embedDim; i++) norm += sp[i] * sp[i];
        norm = MathF.Sqrt(Math.Max(norm, 1e-12f));
        for (int i = 0; i < _embedDim; i++) op[i] = sp[i] / norm;
        e2.Dispose();
        return outE;
    }

    private static Tensor StatsPool(Tensor x, int channels)
    {
        int t = (int)x.Shape[2];
        Tensor outT = new(new TensorShape(1, 1, 2 * channels), DType.F32);
        float* ip = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            double mean = 0;
            long baseOff = (long)c * t;
            for (int j = 0; j < t; j++) mean += ip[baseOff + j];
            mean /= t;
            double var = 0;
            for (int j = 0; j < t; j++) { double d = ip[baseOff + j] - mean; var += d * d; }
            var = Math.Sqrt(var / t + 1e-8);
            op[c] = (float)mean;
            op[channels + c] = (float)var;
        }
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < 5; i++)
        {
            if (_tdnnW[i] is not null) yield return _tdnnW[i]!;
            if (_tdnnB[i] is not null) yield return _tdnnB[i]!;
        }
        Tensor?[] fc = [_fc1W, _fc1B, _fc2W, _fc2B];
        foreach (Tensor? t in fc) if (t is not null) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
