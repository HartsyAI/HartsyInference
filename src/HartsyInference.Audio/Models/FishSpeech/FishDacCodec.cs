using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>OpenAudio S1 modded-DAC codec decoder (Fish-Speech's second generation). The quantizer is a
/// learned <c>DownsampleResidualVectorVectorQuantize</c> — 1 semantic codebook (4096 entries) plus 8 residual
/// codebooks (1024 entries each), all at <c>codebook_dim 8</c> with per-book 1×1 in/out projections; the decode
/// path looks each index up in its codebook, projects to the latent dim, and <b>sums</b> the stages. That latent
/// then runs a standard DAC decoder (Snake1d, <c>decoder_rates [8,8,4,2]</c>, <c>dim 1536</c>, DAC residual
/// units, ConvTranspose1d upsampling, final tanh) to 44.1 kHz mono PCM.
///
/// <para>The DAC residual unit (<c>Snake → dilated WNConv1d k7 → Snake → pointwise WNConv1d k1</c>) is inlined
/// here rather than reusing the HiFiGAN <c>SnakeResBlock</c>: that block's second conv shares the first conv's
/// kernel and (k-1)/2 padding, whereas DAC's second conv is pointwise (kernel 1, pad 0), so SAME-pad sizing must
/// be computed per conv. Unlike FSQ, this RVQ stores explicit learned codebooks, so a missing codebook tensor is a
/// load error.</para></summary>
public sealed unsafe class FishDacCodec : IDisposable
{
    private const int SemanticCodebookSize = 4_096;
    private const int ResidualCodebookSize = 1_024;
    private readonly int _numResidual;          // residual codebooks (8); total books = 1 + numResidual
    private readonly int _codebookDim;          // RVQ inner dim (8)
    private readonly int _latentDim;            // DAC decoder input channels (1536)
    private readonly int _decoderDim;
    private readonly int[] _decoderRates;
    private readonly int[] _resDilations;
    private readonly int _sampleRate;

    // Quantizer: per-book learned codebook + project_out (codebook_dim → latent).
    private readonly Tensor?[] _codebook;       // [codebookSize, codebookDim]
    private readonly Tensor?[] _projOutW;       // [latentDim, codebookDim] (1×1 conv as Linear)
    private readonly Tensor?[] _projOutB;

    // DAC decoder.
    private Tensor? _stemW, _stemB;             // WNConv1d latent → decoderDim, k=7
    private readonly Tensor?[] _stageAlpha;     // per-stage leading Snake1d alpha [dim]
    private readonly Tensor?[] _stageUpW, _stageUpB;
    // DAC residual units [stage][dilation]: Snake → conv k7 dilated → Snake → conv k1.
    // These differ from the HiFiGAN ResBlock1 (SnakeResBlock): the second conv is pointwise
    // (kernel 1, pad 0), so the shared block's (k-1)/2 convs2 padding does not apply here.
    private readonly Tensor?[][] _unitConvs1W, _unitConvs1B;
    private readonly Tensor?[][] _unitConvs2W, _unitConvs2B;
    private readonly Tensor?[][] _unitAlpha1, _unitAlpha2;
    private Tensor? _finalAlpha;
    private Tensor? _finalW, _finalB;
    private int _disposed;

    public FishDacCodec(int numResidualCodebooks = 8, int codebookDim = 8, int latentDim = 1_024,
        int decoderDim = 1_536, int[]? decoderRates = null, int[]? residualDilations = null, int sampleRate = 44_100)
    {
        _numResidual = numResidualCodebooks;
        _codebookDim = codebookDim;
        _latentDim = latentDim;
        _decoderDim = decoderDim;
        _decoderRates = decoderRates ?? [8, 8, 4, 2];
        _resDilations = residualDilations ?? [1, 3, 9];
        _sampleRate = sampleRate;

        int books = 1 + _numResidual;
        _codebook = new Tensor?[books];
        _projOutW = new Tensor?[books];
        _projOutB = new Tensor?[books];

        int stages = _decoderRates.Length;
        _stageAlpha = new Tensor?[stages];
        _stageUpW = new Tensor?[stages];
        _stageUpB = new Tensor?[stages];
        int units = _resDilations.Length;
        _unitConvs1W = new Tensor?[stages][];
        _unitConvs1B = new Tensor?[stages][];
        _unitConvs2W = new Tensor?[stages][];
        _unitConvs2B = new Tensor?[stages][];
        _unitAlpha1 = new Tensor?[stages][];
        _unitAlpha2 = new Tensor?[stages][];
        for (int i = 0; i < stages; i++)
        {
            _unitConvs1W[i] = new Tensor?[units];
            _unitConvs1B[i] = new Tensor?[units];
            _unitConvs2W[i] = new Tensor?[units];
            _unitConvs2B[i] = new Tensor?[units];
            _unitAlpha1[i] = new Tensor?[units];
            _unitAlpha2[i] = new Tensor?[units];
        }
    }

    public int SampleRate => _sampleRate;

    /// <summary>Total codebooks (1 semantic + N residual).</summary>
    public int NumCodebooks => 1 + _numResidual;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string quantizerPrefix = "quantizer",
        string decoderPrefix = "decoder")
    {
        int books = 1 + _numResidual;
        for (int i = 0; i < books; i++)
        {
            string p = $"{quantizerPrefix}.quantizers.{i}";
            _codebook[i] = WhisperOps.EnsureF32(w[$"{p}.codebook.weight"]);
            _projOutW[i] = WhisperOps.EnsureF32(w[$"{p}.out_proj.weight"]);
            if (w.TryGetValue($"{p}.out_proj.bias", out Tensor? b)) _projOutB[i] = WhisperOps.EnsureF32(b);
        }

        _stemW = WhisperOps.EnsureF32(w[$"{decoderPrefix}.model.0.weight"]);
        if (w.TryGetValue($"{decoderPrefix}.model.0.bias", out Tensor? sb)) _stemB = WhisperOps.EnsureF32(sb);

        int dim = _decoderDim;
        for (int i = 0; i < _decoderRates.Length; i++)
        {
            _stageAlpha[i] = WhisperOps.EnsureF32(w[$"{decoderPrefix}.model.{i + 1}.block.0.alpha"]).Reshape(new TensorShape(dim));
            _stageUpW[i] = WhisperOps.EnsureF32(w[$"{decoderPrefix}.model.{i + 1}.block.1.weight"]);
            if (w.TryGetValue($"{decoderPrefix}.model.{i + 1}.block.1.bias", out Tensor? ub)) _stageUpB[i] = WhisperOps.EnsureF32(ub);
            for (int j = 0; j < _resDilations.Length; j++)
            {
                string p = $"{decoderPrefix}.model.{i + 1}.block.{j + 2}";
                _unitConvs1W[i][j] = WeightNorm.Compose(w, $"{p}.convs1.0");
                _unitConvs1B[i][j] = WhisperOps.EnsureF32(w[$"{p}.convs1.0.bias"]);
                _unitConvs2W[i][j] = WeightNorm.Compose(w, $"{p}.convs2.0");
                _unitConvs2B[i][j] = WhisperOps.EnsureF32(w[$"{p}.convs2.0.bias"]);
                _unitAlpha1[i][j] = WhisperOps.EnsureF32(w[$"{p}.activations1.0.alpha"]);
                _unitAlpha2[i][j] = WhisperOps.EnsureF32(w[$"{p}.activations2.0.alpha"]);
            }
            dim /= 2;
        }
        _finalAlpha = WhisperOps.EnsureF32(w[$"{decoderPrefix}.model.{_decoderRates.Length + 1}.alpha"]).Reshape(new TensorShape(dim));
        _finalW = WhisperOps.EnsureF32(w[$"{decoderPrefix}.model.{_decoderRates.Length + 2}.weight"]);
        if (w.TryGetValue($"{decoderPrefix}.model.{_decoderRates.Length + 2}.bias", out Tensor? fb)) _finalB = WhisperOps.EnsureF32(fb);
    }

    /// <summary>Decodes <c>[1 + numResidual, T]</c> codebook indices (row 0 = semantic) to 44.1 kHz mono PCM.</summary>
    public float[] Decode(IBackend backend, int[,] codes, int t)
    {
        int books = 1 + _numResidual;
        if (codes.GetLength(0) != books)
            throw new ArgumentException($"codes must have {books} rows (1 semantic + {_numResidual} residual), got {codes.GetLength(0)}.");
        if (t <= 0) throw new ArgumentException("t must be positive.", nameof(t));

        // RVQ get_output_from_indices: sum per-book project_out(codebook[index]) → [1, latentDim, T].
        Tensor latent = new(new TensorShape(1, _latentDim, t), DType.F32);
        float* lp = (float*)latent.DataPointer;
        for (long n = 0; n < latent.ElementCount; n++) lp[n] = 0f;

        for (int i = 0; i < books; i++)
        {
            int maxIdx = i == 0 ? SemanticCodebookSize : ResidualCodebookSize;
            float* cb = (float*)_codebook[i]!.DataPointer;
            float* pw = (float*)_projOutW[i]!.DataPointer;     // [latentDim, codebookDim]
            float* pb = _projOutB[i] is null ? null : (float*)_projOutB[i]!.DataPointer;
            int cbDim = (int)_codebook[i]!.Shape[1];
            for (int j = 0; j < t; j++)
            {
                int idx = codes[i, j];
                if (idx < 0 || idx >= maxIdx)
                    throw new ArgumentException($"code [{i},{j}]={idx} out of range [0,{maxIdx}).");
                float* vec = cb + (long)idx * cbDim;
                for (int c = 0; c < _latentDim; c++)
                {
                    float acc = pb is null ? 0f : pb[c];
                    float* row = pw + (long)c * cbDim;
                    for (int k = 0; k < cbDim; k++) acc += row[k] * vec[k];
                    lp[(long)c * t + j] += acc;
                }
            }
        }

        // DAC decoder: stem conv → stages (Snake → upsample → residual units) → final Snake → conv → tanh.
        int stemK = (int)_stemW!.Shape[2];
        int stemPad = (stemK - 1) / 2;
        int stemT = t + 2 * stemPad - (stemK - 1);
        Tensor x = new(new TensorShape(1, _decoderDim, stemT), DType.F32);
        backend.Conv1d(x, latent, _stemW!, _stemB, 1, stemPad, stemPad, 1, 1);
        latent.Dispose();

        int curT = stemT;
        int dim = _decoderDim;
        for (int i = 0; i < _decoderRates.Length; i++)
        {
            Tensor snk = new(x.Shape, DType.F32);
            backend.Snake(snk, x, _stageAlpha[i]!, null);
            x.Dispose();

            int stride = _decoderRates[i];
            int kUp = (int)_stageUpW[i]!.Shape[2];
            int padding = (stride + 1) / 2;
            int outputPadding = stride % 2;
            int padLeft = padding;
            int padRight = padding - outputPadding;
            int outDim = dim / 2;
            int upT = (curT - 1) * stride + kUp - padLeft - padRight;
            Tensor up = new(new TensorShape(1, outDim, upT), DType.F32);
            backend.ConvTranspose1d(up, snk, _stageUpW[i]!, _stageUpB[i], stride, padLeft, padRight, 1, 1);
            snk.Dispose();
            x = up; curT = upT; dim = outDim;

            for (int j = 0; j < _resDilations.Length; j++)
            {
                x = ResidualUnit(backend, x, i, j, curT);
                curT = (int)x.Shape[2];
            }
        }

        Tensor act = new(x.Shape, DType.F32);
        backend.Snake(act, x, _finalAlpha!, null);
        x.Dispose();
        int fK = (int)_finalW!.Shape[2];
        int fPad = (fK - 1) / 2;
        int fT = curT + 2 * fPad - (fK - 1);
        Tensor post = new(new TensorShape(1, 1, fT), DType.F32);
        backend.Conv1d(post, act, _finalW!, _finalB, 1, fPad, fPad, 1, 1);
        act.Dispose();

        float* postP = (float*)post.DataPointer;
        float[] audio = new float[fT];
        for (int j = 0; j < fT; j++) audio[j] = MathF.Tanh(postP[j]);
        post.Dispose();
        return audio;
    }

    /// <summary>DAC residual unit: <c>Snake → dilated conv (k7, SAME) → Snake → pointwise conv (k1) → +input</c>.
    /// Both convs are stride-1 SAME-padded, so every intermediate keeps the running length <paramref name="curT"/>;
    /// the pointwise conv uses zero padding because its kernel is 1, unlike the dilated branch.</summary>
    private Tensor ResidualUnit(IBackend backend, Tensor x, int stage, int unit, int curT)
    {
        int channels = (int)x.Shape[1];
        int dilation = _resDilations[unit];
        int k1 = (int)_unitConvs1W[stage][unit]!.Shape[2];
        int pad1 = (k1 - 1) * dilation / 2;
        int k2 = (int)_unitConvs2W[stage][unit]!.Shape[2];
        int pad2 = (k2 - 1) / 2;

        Tensor a1 = new(x.Shape, DType.F32);
        backend.Snake(a1, x, _unitAlpha1[stage][unit]!, null);
        Tensor c1 = new(new TensorShape(1, channels, curT), DType.F32);
        backend.Conv1d(c1, a1, _unitConvs1W[stage][unit]!, _unitConvs1B[stage][unit], 1, pad1, pad1, dilation, 1);
        a1.Dispose();

        Tensor a2 = new(c1.Shape, DType.F32);
        backend.Snake(a2, c1, _unitAlpha2[stage][unit]!, null);
        c1.Dispose();
        Tensor c2 = new(new TensorShape(1, channels, curT), DType.F32);
        backend.Conv1d(c2, a2, _unitConvs2W[stage][unit]!, _unitConvs2B[stage][unit], 1, pad2, pad2, 1, 1);
        a2.Dispose();

        float* xp = (float*)x.DataPointer;
        float* c2p = (float*)c2.DataPointer;
        long n = c2.ElementCount;
        for (long m = 0; m < n; m++) c2p[m] += xp[m];
        x.Dispose();
        return c2;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[][] q = [_codebook, _projOutW, _projOutB];
        foreach (Tensor?[] g in q) foreach (Tensor? t in g) if (t is not null) yield return t;
        if (_stemW is not null) yield return _stemW;
        if (_stemB is not null) yield return _stemB;
        for (int i = 0; i < _decoderRates.Length; i++)
        {
            if (_stageAlpha[i] is not null) yield return _stageAlpha[i]!;
            if (_stageUpW[i] is not null) yield return _stageUpW[i]!;
            if (_stageUpB[i] is not null) yield return _stageUpB[i]!;
            for (int j = 0; j < _resDilations.Length; j++)
            {
                Tensor?[] unit = [_unitConvs1W[i][j], _unitConvs1B[i][j], _unitConvs2W[i][j], _unitConvs2B[i][j],
                    _unitAlpha1[i][j], _unitAlpha2[i][j]];
                foreach (Tensor? t in unit) if (t is not null) yield return t;
            }
        }
        if (_finalAlpha is not null) yield return _finalAlpha;
        if (_finalW is not null) yield return _finalW;
        if (_finalB is not null) yield return _finalB;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
