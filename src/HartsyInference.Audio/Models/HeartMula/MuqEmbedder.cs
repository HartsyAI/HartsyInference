using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.HeartMula;

/// <summary>MuQ-MuLan style-conditioning embedder: takes a reference-audio mel <c>[1, melBins, T]</c> and produces a 512-d MuQ embedding via a conv stem, pre-norm self-attention layers, mean-pooling over time, then a final projection.</summary>
// ProjectToLmHidden then applies muq_linear to lift the result into the LM hidden size for conditioning.
// Weight keys: conv_stem.{i}.{weight,bias}, layers.{i}.{ln1,ln2,q,k,v,o,fc1,fc2}.*, norm.*,
// proj.{weight,bias}, and muq_linear.{weight,bias} (LM-side projection).
public sealed unsafe class MuqEmbedder : IDisposable
{
    private readonly int _melBins, _muqDim, _hidden, _layers, _heads, _headDim, _ffn, _lmHidden;
    private readonly int[] _stemStrides;
    private readonly float _eps = 1e-5f;

    private readonly Tensor?[] _stemW, _stemB;
    private readonly Tensor?[] _ln1W, _ln1B, _ln2W, _ln2B;
    private readonly Tensor?[] _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB;
    private readonly Tensor?[] _fc1W, _fc1B, _fc2W, _fc2B;
    private Tensor? _normW, _normB, _projW, _projB, _muqLinW, _muqLinB;
    private int _disposed;

    public MuqEmbedder(int melBins, int muqDim, int lmHidden, int hidden = 256, int layers = 4,
        int heads = 8, int ffn = 0, int[]? stemStrides = null)
    {
        _melBins = melBins;
        _muqDim = muqDim;
        _lmHidden = lmHidden;
        _hidden = hidden;
        _layers = layers;
        _heads = heads;
        if (hidden % heads != 0) throw new ArgumentException($"hidden ({hidden}) must be divisible by heads ({heads}).");
        _headDim = hidden / heads;
        _ffn = ffn > 0 ? ffn : hidden * 4;
        _stemStrides = stemStrides ?? [2, 2];
        _stemW = new Tensor?[_stemStrides.Length]; _stemB = new Tensor?[_stemStrides.Length];
        _ln1W = new Tensor?[layers]; _ln1B = new Tensor?[layers]; _ln2W = new Tensor?[layers]; _ln2B = new Tensor?[layers];
        _qW = new Tensor?[layers]; _qB = new Tensor?[layers]; _kW = new Tensor?[layers]; _kB = new Tensor?[layers];
        _vW = new Tensor?[layers]; _vB = new Tensor?[layers]; _oW = new Tensor?[layers]; _oB = new Tensor?[layers];
        _fc1W = new Tensor?[layers]; _fc1B = new Tensor?[layers]; _fc2W = new Tensor?[layers]; _fc2B = new Tensor?[layers];
    }

    public int MuqDim => _muqDim;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        for (int i = 0; i < _stemStrides.Length; i++)
        {
            _stemW[i] = WhisperOps.EnsureF32(w[$"{p}conv_stem.{i}.weight"]);
            _stemB[i] = Bias(w, $"{p}conv_stem.{i}.bias");
        }
        for (int i = 0; i < _layers; i++)
        {
            string lp = $"{p}layers.{i}";
            _ln1W[i] = WhisperOps.EnsureF32(w[$"{lp}.ln1.weight"]); _ln1B[i] = WhisperOps.EnsureF32(w[$"{lp}.ln1.bias"]);
            _ln2W[i] = WhisperOps.EnsureF32(w[$"{lp}.ln2.weight"]); _ln2B[i] = WhisperOps.EnsureF32(w[$"{lp}.ln2.bias"]);
            _qW[i] = WhisperOps.EnsureF32(w[$"{lp}.q.weight"]); _qB[i] = Bias(w, $"{lp}.q.bias");
            _kW[i] = WhisperOps.EnsureF32(w[$"{lp}.k.weight"]); _kB[i] = Bias(w, $"{lp}.k.bias");
            _vW[i] = WhisperOps.EnsureF32(w[$"{lp}.v.weight"]); _vB[i] = Bias(w, $"{lp}.v.bias");
            _oW[i] = WhisperOps.EnsureF32(w[$"{lp}.o.weight"]); _oB[i] = Bias(w, $"{lp}.o.bias");
            _fc1W[i] = WhisperOps.EnsureF32(w[$"{lp}.fc1.weight"]); _fc1B[i] = Bias(w, $"{lp}.fc1.bias");
            _fc2W[i] = WhisperOps.EnsureF32(w[$"{lp}.fc2.weight"]); _fc2B[i] = Bias(w, $"{lp}.fc2.bias");
        }
        _normW = WhisperOps.EnsureF32(w[$"{p}norm.weight"]); _normB = WhisperOps.EnsureF32(w[$"{p}norm.bias"]);
        _projW = WhisperOps.EnsureF32(w[$"{p}proj.weight"]); _projB = Bias(w, $"{p}proj.bias");
        _muqLinW = WhisperOps.EnsureF32(w[$"{p}muq_linear.weight"]); _muqLinB = Bias(w, $"{p}muq_linear.bias");
    }

    /// <summary>Embeds a reference mel into the 512-d MuQ style vector <c>[1, muqDim]</c>.</summary>
    public Tensor Embed(IBackend backend, Tensor refMel, int t)
    {
        ThrowIfDisposed();
        if (_normW is null) throw new InvalidOperationException("MuqEmbedder weights not loaded.");

        // Conv stem: [1, melBins, T] → [1, hidden, T'].
        Tensor feat = refMel;
        int curT = t;
        bool ownFeat = false;
        for (int i = 0; i < _stemStrides.Length; i++)
        {
            int stride = _stemStrides[i];
            int k = (int)_stemW[i]!.Shape[2];
            int outCh = (int)_stemW[i]!.Shape[0];
            int pad = (k - 1) / 2;
            int outT = (curT + 2 * pad - (k - 1) - 1) / stride + 1;
            Tensor next = new(new TensorShape(1, outCh, outT), DType.F32);
            backend.Conv1d(next, feat, _stemW[i]!, _stemB[i], stride, pad, pad, 1, 1);
            backend.Silu(next, next);
            if (ownFeat) feat.Dispose();
            feat = next; curT = outT; ownFeat = true;
        }

        // [1, hidden, T'] → tokens [1, T', hidden] (channels-last for the transformer).
        int seq = curT;
        Tensor seqT = new(new TensorShape(1, seq, _hidden), DType.F32);
        float* fp = (float*)feat.DataPointer; float* sp = (float*)seqT.DataPointer;
        for (int c = 0; c < _hidden; c++)
            for (int j = 0; j < seq; j++) sp[(long)j * _hidden + c] = fp[(long)c * seq + j];
        if (ownFeat) feat.Dispose();

        float scale = 1f / MathF.Sqrt(_headDim);
        for (int i = 0; i < _layers; i++)
        {
            // Pre-norm self-attention.
            Tensor n1 = new(new TensorShape(1, seq, _hidden), DType.F32);
            backend.LayerNorm(n1, seqT, _ln1W[i]!, _ln1B[i]!, _eps);
            Tensor q = WhisperOps.ProjectLinear(backend, n1, _qW[i]!, _qB[i], 1, seq, _hidden, _hidden);
            Tensor k = WhisperOps.ProjectLinear(backend, n1, _kW[i]!, _kB[i], 1, seq, _hidden, _hidden);
            Tensor v = WhisperOps.ProjectLinear(backend, n1, _vW[i]!, _vB[i], 1, seq, _hidden, _hidden);
            n1.Dispose();

            Tensor q4 = new(new TensorShape(1, _heads, seq, _headDim), DType.F32);
            Tensor k4 = new(new TensorShape(1, _heads, seq, _headDim), DType.F32);
            Tensor v4 = new(new TensorShape(1, _heads, seq, _headDim), DType.F32);
            WhisperOps.ReshapeToMultiHead4D(q4, q, 1, seq, _heads, _headDim);
            WhisperOps.ReshapeToMultiHead4D(k4, k, 1, seq, _heads, _headDim);
            WhisperOps.ReshapeToMultiHead4D(v4, v, 1, seq, _heads, _headDim);
            q.Dispose(); k.Dispose(); v.Dispose();

            Tensor a4 = new(new TensorShape(1, _heads, seq, _headDim), DType.F32);
            backend.ScaledDotProductAttention(a4, q4, k4, v4, null, scale);
            q4.Dispose(); k4.Dispose(); v4.Dispose();

            Tensor attn = new(new TensorShape(1, seq, _hidden), DType.F32);
            WhisperOps.ReshapeFromMultiHead4D(attn, a4, 1, seq, _heads, _headDim);
            a4.Dispose();

            Tensor o = WhisperOps.ProjectLinear(backend, attn, _oW[i]!, _oB[i], 1, seq, _hidden, _hidden);
            attn.Dispose();
            AddInPlace(seqT, o); o.Dispose();

            // Pre-norm FFN (SiLU-gated MLP collapsed to fc1 → SiLU → fc2).
            Tensor n2 = new(new TensorShape(1, seq, _hidden), DType.F32);
            backend.LayerNorm(n2, seqT, _ln2W[i]!, _ln2B[i]!, _eps);
            Tensor f1 = WhisperOps.ProjectLinear(backend, n2, _fc1W[i]!, _fc1B[i], 1, seq, _hidden, _ffn);
            n2.Dispose();
            backend.Silu(f1, f1);
            Tensor f2 = WhisperOps.ProjectLinear(backend, f1, _fc2W[i]!, _fc2B[i], 1, seq, _ffn, _hidden);
            f1.Dispose();
            AddInPlace(seqT, f2); f2.Dispose();
        }

        // Final LayerNorm + mean-pool over time → [1, 1, hidden].
        Tensor normed = new(new TensorShape(1, seq, _hidden), DType.F32);
        backend.LayerNorm(normed, seqT, _normW!, _normB!, _eps);
        seqT.Dispose();
        Tensor pooled = new(new TensorShape(1, 1, _hidden), DType.F32);
        float* np = (float*)normed.DataPointer; float* pp = (float*)pooled.DataPointer;
        for (int c = 0; c < _hidden; c++)
        {
            double acc = 0d;
            for (int j = 0; j < seq; j++) acc += np[(long)j * _hidden + c];
            pp[c] = (float)(acc / seq);
        }
        normed.Dispose();

        // Projection → muqDim, returned as [1, muqDim].
        Tensor muq3 = WhisperOps.ProjectLinear(backend, pooled, _projW!, _projB, 1, 1, _hidden, _muqDim);
        pooled.Dispose();
        Tensor muq = new(new TensorShape(1, _muqDim), DType.F32);
        long mbytes = (long)_muqDim * sizeof(float);
        Buffer.MemoryCopy((void*)muq3.DataPointer, (void*)muq.DataPointer, mbytes, mbytes);
        muq3.Dispose();
        return muq;
    }

    /// <summary>Applies <c>muq_linear</c> to lift a MuQ embedding <c>[1, muqDim]</c> into the LM hidden <c>[1, lmHidden]</c> for conditioning.</summary>
    public Tensor ProjectToLmHidden(IBackend backend, Tensor muq)
    {
        ThrowIfDisposed();
        if (_muqLinW is null) throw new InvalidOperationException("MuqEmbedder weights not loaded.");
        Tensor muq3 = new(new TensorShape(1, 1, _muqDim), DType.F32);
        long bytes = (long)_muqDim * sizeof(float);
        Buffer.MemoryCopy((void*)muq.DataPointer, (void*)muq3.DataPointer, bytes, bytes);
        Tensor proj3 = WhisperOps.ProjectLinear(backend, muq3, _muqLinW!, _muqLinB, 1, 1, _muqDim, _lmHidden);
        muq3.Dispose();
        Tensor proj = new(new TensorShape(1, _lmHidden), DType.F32);
        long obytes = (long)_lmHidden * sizeof(float);
        Buffer.MemoryCopy((void*)proj3.DataPointer, (void*)proj.DataPointer, obytes, obytes);
        proj3.Dispose();
        return proj;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[][] arrs = [_stemW, _stemB, _ln1W, _ln1B, _ln2W, _ln2B, _qW, _qB, _kW, _kB, _vW, _vB,
            _oW, _oB, _fc1W, _fc1B, _fc2W, _fc2B];
        foreach (Tensor?[] arr in arrs) foreach (Tensor? t in arr) if (t is not null) yield return t;
        Tensor?[] own = [_normW, _normB, _projW, _projB, _muqLinW, _muqLinB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(MuqEmbedder));
    }

    private static void AddInPlace(Tensor dst, Tensor src)
    {
        float* dp = (float*)dst.DataPointer; float* sp = (float*)src.DataPointer;
        long n = dst.ElementCount;
        for (long i = 0; i < n; i++) dp[i] += sp[i];
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
}
