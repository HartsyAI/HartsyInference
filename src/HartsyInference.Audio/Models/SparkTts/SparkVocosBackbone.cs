using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.SparkTts;

/// <summary>Vocos ConvNeXt backbone used by the Spark-TTS BiCodec PreNet (mirrors
/// <c>sparktts.modules.blocks.vocos.VocosBackbone</c>). An input <c>embed</c> conv, a (plain or adaptive)
/// LayerNorm, a stack of <see cref="ConvNeXtBlock"/>s, then a final plain LayerNorm. With
/// <paramref name="conditionDim"/> &gt; 0 every norm is an AdaLayerNorm conditioned on the speaker
/// d-vector (scale/shift produced by per-block linears); otherwise norms are plain. Channels-first
/// <c>[1, C, T]</c> in, channels-last <c>[1, T, C]</c> out (matching the reference, which transposes
/// before the final norm).</summary>
internal sealed unsafe class VocosBackbone
{
    private const float Eps = 1e-6f;
    private readonly int _dim, _intermediate, _numLayers, _conditionDim;
    private readonly bool _adaNorm;
    private readonly ConvNeXtBlock[] _blocks;

    private Tensor? _embedW, _embedB;
    private AdaOrPlainNorm _norm = default!;
    private Tensor? _finalW, _finalB;

    public VocosBackbone(int dim, int intermediate, int numLayers, int conditionDim)
    {
        _dim = dim; _intermediate = intermediate; _numLayers = numLayers; _conditionDim = conditionDim;
        _adaNorm = conditionDim > 0;
        _blocks = new ConvNeXtBlock[numLayers];
        for (int i = 0; i < numLayers; i++) _blocks[i] = new ConvNeXtBlock(dim, intermediate, conditionDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _embedW = WhisperOps.EnsureF32(w[$"{prefix}.embed.weight"]);
        _embedB = WhisperOps.EnsureF32(w[$"{prefix}.embed.bias"]);
        _norm = AdaOrPlainNorm.Load(w, $"{prefix}.norm", _adaNorm);
        for (int i = 0; i < _numLayers; i++) _blocks[i].LoadWeights(w, $"{prefix}.convnext.{i}");
        _finalW = WhisperOps.EnsureF32(w[$"{prefix}.final_layer_norm.weight"]);
        _finalB = WhisperOps.EnsureF32(w[$"{prefix}.final_layer_norm.bias"]);
    }

    /// <summary>Runs the backbone. <paramref name="x"/> is channels-first <c>[1, C, T]</c>; returns
    /// channels-last <c>[1, T, C]</c>. <paramref name="cond"/> (the d-vector <c>[1, conditionDim]</c>) is
    /// required iff this backbone was built with conditioning.</summary>
    public Tensor Run(IBackend backend, Tensor x, int t, Tensor? cond)
    {
        int inC = (int)_embedW!.Shape[1];
        Tensor e = new(new TensorShape(1, _dim, t), DType.F32);
        backend.Conv1d(e, x, _embedW!, _embedB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);
        Tensor eTc = BiCodecDecoder.Transpose(e, _dim, t);  // [1, T, dim]
        e.Dispose();

        Tensor normed = _norm.Apply(backend, eTc, t, _dim, cond);
        eTc.Dispose();
        Tensor xcf = BiCodecDecoder.Transpose(normed, t, _dim);  // [1, dim, T]
        normed.Dispose();

        foreach (ConvNeXtBlock block in _blocks)
        {
            Tensor next = block.Run(backend, xcf, t, cond);
            xcf.Dispose();
            xcf = next;
        }

        Tensor xtc = BiCodecDecoder.Transpose(xcf, _dim, t);  // [1, T, dim]
        xcf.Dispose();
        Tensor outTc = new(new TensorShape(1, t, _dim), DType.F32);
        backend.LayerNorm(outTc, xtc, _finalW!, _finalB!, Eps);
        xtc.Dispose();
        return outTc;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedW is not null) yield return _embedW;
        if (_embedB is not null) yield return _embedB;
        foreach (Tensor t in _norm.EnumerateWeights()) yield return t;
        foreach (ConvNeXtBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        if (_finalW is not null) yield return _finalW;
        if (_finalB is not null) yield return _finalB;
    }
}

/// <summary>One ConvNeXt block (1D): depthwise conv → (Ada)LayerNorm → pointwise(→GELU→pointwise) → layer
/// scale γ → residual. Channels-first <c>[1, C, T]</c> in/out.</summary>
internal sealed unsafe class ConvNeXtBlock
{
    private const float Eps = 1e-6f;
    private readonly int _dim, _intermediate;
    private readonly bool _adaNorm;

    private Tensor? _dwW, _dwB, _pw1W, _pw1B, _pw2W, _pw2B, _gamma;
    private AdaOrPlainNorm _norm = default!;

    public ConvNeXtBlock(int dim, int intermediate, int conditionDim)
    {
        _dim = dim; _intermediate = intermediate; _adaNorm = conditionDim > 0;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _dwW = WhisperOps.EnsureF32(w[$"{prefix}.dwconv.weight"]);
        _dwB = WhisperOps.EnsureF32(w[$"{prefix}.dwconv.bias"]);
        _norm = AdaOrPlainNorm.Load(w, $"{prefix}.norm", _adaNorm);
        _pw1W = WhisperOps.EnsureF32(w[$"{prefix}.pwconv1.weight"]);
        _pw1B = WhisperOps.EnsureF32(w[$"{prefix}.pwconv1.bias"]);
        _pw2W = WhisperOps.EnsureF32(w[$"{prefix}.pwconv2.weight"]);
        _pw2B = WhisperOps.EnsureF32(w[$"{prefix}.pwconv2.bias"]);
        _gamma = WhisperOps.EnsureF32(w[$"{prefix}.gamma"]);
    }

    public Tensor Run(IBackend backend, Tensor x, int t, Tensor? cond)
    {
        // depthwise conv (groups = dim), k=7, pad=3.
        Tensor dw = new(new TensorShape(1, _dim, t), DType.F32);
        backend.Conv1d(dw, x, _dwW!, _dwB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: _dim);
        Tensor dwTc = BiCodecDecoder.Transpose(dw, _dim, t);  // [1, T, dim]
        dw.Dispose();

        Tensor normed = _norm.Apply(backend, dwTc, t, _dim, cond);
        dwTc.Dispose();

        Tensor h = WhisperOps.ProjectLinear(backend, normed, _pw1W!, _pw1B, 1, t, _dim, _intermediate);
        normed.Dispose();
        Activations.ErfGelu(h);
        Tensor h2 = WhisperOps.ProjectLinear(backend, h, _pw2W!, _pw2B, 1, t, _intermediate, _dim);
        h.Dispose();

        // layer-scale γ (per channel, last dim).
        float* hp = (float*)h2.DataPointer;
        float* gp = (float*)_gamma!.DataPointer;
        for (int ti = 0; ti < t; ti++)
            for (int c = 0; c < _dim; c++) hp[(long)ti * _dim + c] *= gp[c];

        Tensor yCf = BiCodecDecoder.Transpose(h2, t, _dim);  // [1, dim, T]
        h2.Dispose();

        // residual.
        float* yp = (float*)yCf.DataPointer;
        float* xp = (float*)x.DataPointer;
        long n = (long)_dim * t;
        for (long i = 0; i < n; i++) yp[i] += xp[i];
        return yCf;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_dwW, _dwB, _pw1W, _pw1B, _pw2W, _pw2B, _gamma];
        foreach (Tensor? x in own) if (x is not null) yield return x;
        foreach (Tensor t in _norm.EnumerateWeights()) yield return t;
    }
}

/// <summary>A LayerNorm that is either plain (learned weight/bias) or adaptive — an AdaLayerNorm whose
/// scale/shift are linear projections of a conditioning vector (Spark BiCodec PreNet). Operates on
/// channels-last <c>[1, T, C]</c>.</summary>
internal unsafe struct AdaOrPlainNorm
{
    private bool _ada;
    private Tensor? _w, _b;                     // plain LayerNorm affine
    private Tensor? _scaleW, _scaleB, _shiftW, _shiftB;   // AdaLayerNorm projections

    public static AdaOrPlainNorm Load(IReadOnlyDictionary<string, Tensor> w, string prefix, bool ada)
    {
        AdaOrPlainNorm n = new() { _ada = ada };
        if (ada)
        {
            n._scaleW = WhisperOps.EnsureF32(w[$"{prefix}.scale.weight"]);
            n._scaleB = WhisperOps.EnsureF32(w[$"{prefix}.scale.bias"]);
            n._shiftW = WhisperOps.EnsureF32(w[$"{prefix}.shift.weight"]);
            n._shiftB = WhisperOps.EnsureF32(w[$"{prefix}.shift.bias"]);
        }
        else
        {
            n._w = WhisperOps.EnsureF32(w[$"{prefix}.weight"]);
            n._b = WhisperOps.EnsureF32(w[$"{prefix}.bias"]);
        }
        return n;
    }

    /// <summary>Normalizes <paramref name="x"/> <c>[1, T, C]</c> → <c>[1, T, C]</c>.</summary>
    public readonly Tensor Apply(IBackend backend, Tensor x, int t, int c, Tensor? cond)
    {
        Tensor o = new(new TensorShape(1, t, c), DType.F32);
        if (!_ada)
        {
            backend.LayerNorm(o, x, _w!, _b!, 1e-6f);
            return o;
        }
        backend.LayerNormNoAffine(o, x, 1e-6f);
        int condDim = (int)_scaleW!.Shape[1];
        Tensor condR = cond!.Reshape(new TensorShape(1, 1, condDim));
        Tensor scale = WhisperOps.ProjectLinear(backend, condR, _scaleW!, _scaleB, 1, 1, condDim, c);  // [1,1,c]
        Tensor shift = WhisperOps.ProjectLinear(backend, condR, _shiftW!, _shiftB, 1, 1, condDim, c);
        float* op = (float*)o.DataPointer;
        float* sp = (float*)scale.DataPointer;
        float* hp = (float*)shift.DataPointer;
        for (int ti = 0; ti < t; ti++)
            for (int ci = 0; ci < c; ci++) op[(long)ti * c + ci] = op[(long)ti * c + ci] * sp[ci] + hp[ci];
        scale.Dispose(); shift.Dispose();
        return o;
    }

    public readonly IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_w, _b, _scaleW, _scaleB, _shiftW, _shiftB];
        foreach (Tensor? x in own) if (x is not null) yield return x;
    }
}
