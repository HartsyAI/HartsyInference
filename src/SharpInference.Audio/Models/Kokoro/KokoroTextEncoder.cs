using SharpInference.Audio.Layers;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Kokoro;

/// <summary>Kokoro's <c>TextEncoder</c> — the second text path that feeds the decoder's
/// alignment-then-asr_res stage. Pipeline:
///
/// <code>
///   token_ids [1, T] int
///       │
///       ▼  text_encoder.embedding [178, 512] (channels-last [1, T, 512])
///       │
///       ▼  (transpose to channels-first [1, 512, T])
///       │
///   for i in 0..2:
///     Conv1d(512 → 512, k=5, pad=2)   weight = WeightNorm.Compose(cnn.{i}.0)
///     LayerNorm1d(channels)            cnn.{i}.1.gamma / .beta over the channel axis
///     LeakyReLU(0.2)
///       │
///       ▼  (transpose back to channels-last [1, T, 512])
///       │
///       ▼  BiLSTM(input=512, hidden=256)  → [1, T, 512]
///       ▼
///   text_features (channels-last). The pipeline then transposes to channels-first
///   [1, 512, T] and length-regulates via repeat-by-duration so it becomes
///   [1, 512, T_total] before the decoder's <c>asr_res</c> path.
/// </code>
///
/// <para>The "LayerNorm1d" between conv and LeakyReLU is a custom channel-axis LN with
/// parameters named <c>gamma</c> / <c>beta</c> (vs. the standard <c>weight</c> /
/// <c>bias</c>). We implement it as transpose → standard LayerNorm → transpose-back.</para></summary>
internal sealed unsafe class KokoroTextEncoder
{
    private readonly KokoroConfig _cfg;

    private Tensor? _emb;                     // [178, 512]
    private readonly Tensor?[] _convW = new Tensor[3];     // composed [512, 512, 5]
    private readonly Tensor?[] _convB = new Tensor[3];     // [512]
    private readonly Tensor?[] _lnGamma = new Tensor[3];   // [512]
    private readonly Tensor?[] _lnBeta = new Tensor[3];    // [512]
    private readonly BiLstm _lstm;

    private const float LeakySlope = 0.2f;
    private const float LayerNormEps = 1e-5f;

    public KokoroTextEncoder(KokoroConfig cfg)
    {
        _cfg = cfg;
        // BiLSTM input=512, hidden=256, output=512 (2*256).
        _lstm = new BiLstm(inputDim: cfg.HiddenDim, hiddenDim: cfg.HiddenDim / 2);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _emb = WhisperOps.EnsureF32(w["text_encoder.embedding.weight"]);
        for (int i = 0; i < _cfg.TextEncoderNumLayers; i++)
        {
            _convW[i] = WeightNorm.Compose(w, $"text_encoder.cnn.{i}.0");
            _convB[i] = WhisperOps.EnsureF32(w[$"text_encoder.cnn.{i}.0.bias"]);
            _lnGamma[i] = WhisperOps.EnsureF32(w[$"text_encoder.cnn.{i}.1.gamma"]);
            _lnBeta[i] = WhisperOps.EnsureF32(w[$"text_encoder.cnn.{i}.1.beta"]);
        }
        _lstm.LoadWeights(w, "text_encoder.lstm");
    }

    /// <summary>Forward pass. Returns <c>[1, T, 512]</c> channels-last F32.</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> tokenIds)
    {
        int t = tokenIds.Length;
        int c = _cfg.HiddenDim;
        int k = _cfg.TextEncoderKernelSize;
        int pad = k / 2;

        // 1. Embedding lookup → channels-last [1, T, 512], then transpose to [1, 512, T].
        Tensor embCL = new(new TensorShape(1, t, c), DType.F32);
        float* embPtr = (float*)_emb!.DataPointer;
        float* ep = (float*)embCL.DataPointer;
        for (int i = 0; i < t; i++)
        {
            int id = tokenIds[i];
            for (int d = 0; d < c; d++) ep[i * c + d] = embPtr[(long)id * c + d];
        }

        Tensor xCF = new(new TensorShape(1, c, t), DType.F32);
        backend.Transpose2D(xCF, embCL, t, c);
        embCL.Dispose();

        // 2. 3× (Conv1d → ChannelLN → LeakyReLU). All shapes preserved.
        for (int i = 0; i < _cfg.TextEncoderNumLayers; i++)
        {
            Tensor convOut = new(xCF.Shape, DType.F32);
            backend.Conv1d(convOut, xCF, _convW[i]!, _convB[i], stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
            xCF.Dispose();

            // ChannelLN: transpose [1,C,T] → [1,T,C], LN over last dim, transpose back.
            Tensor convCL = new(new TensorShape(1, t, c), DType.F32);
            backend.Transpose2D(convCL, convOut, c, t);
            convOut.Dispose();

            Tensor lnCL = new(convCL.Shape, DType.F32);
            backend.LayerNorm(lnCL, convCL, _lnGamma[i]!, _lnBeta[i]!, LayerNormEps);
            convCL.Dispose();

            // LeakyReLU in channels-last (elementwise — layout doesn't matter).
            backend.LeakyRelu(lnCL, lnCL, LeakySlope);

            // Back to channels-first for the next conv.
            Tensor nextCF = new(new TensorShape(1, c, t), DType.F32);
            backend.Transpose2D(nextCF, lnCL, t, c);
            lnCL.Dispose();
            xCF = nextCF;
        }

        // 3. To channels-last for the BiLSTM ([B, T, C]).
        Tensor xCL = new(new TensorShape(1, t, c), DType.F32);
        backend.Transpose2D(xCL, xCF, c, t);
        xCF.Dispose();

        // 4. BiLSTM 512 → 512 (256 hidden × 2 directions).
        Tensor lstmOut = _lstm.Forward(backend, xCL, batch: 1, t: t);
        xCL.Dispose();
        return lstmOut;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_emb is not null) yield return _emb;
        for (int i = 0; i < _cfg.TextEncoderNumLayers; i++)
        {
            if (_convW[i] is not null) yield return _convW[i]!;
            if (_convB[i] is not null) yield return _convB[i]!;
            if (_lnGamma[i] is not null) yield return _lnGamma[i]!;
            if (_lnBeta[i] is not null) yield return _lnBeta[i]!;
        }
        foreach (Tensor x in _lstm.EnumerateWeights()) yield return x;
    }
}
