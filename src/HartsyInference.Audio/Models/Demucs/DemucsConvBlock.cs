using HartsyInference.Audio.Layers;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Demucs;

/// <summary>HTDemucs HEncLayer / HDecLayer, parameterized for the time branch (1D) and the spectrogram branch
/// (2D, stride only on the frequency axis). Matches demucs exactly (norm1/norm2 = Identity at the released
/// depth-4 config since norm_starts=4):
/// <code>
///   Encoder: y = GELU(conv(x))            # conv: Conv2d/Conv1d, stride, pad=(k-stride)/2
///            y = dconv(y)                  # DConv residual branch (per freq-row in the 2D case)
///            z = GLU(rewrite(y))           # rewrite: 1x1
///   Decoder: y = GLU(rewrite(x+skip))     # rewrite: 3x3 (pad 1)  [skip added by caller]
///            y = dconv(y)
///            z = conv_tr(y)                # ConvTranspose, stride; trim (k-stride)/2 (freq) or [pad:pad+len] (time)
///            z = GELU(z)                   # except the last layer
/// </code></summary>
public sealed unsafe class DemucsConvBlock : IDisposable
{
    private readonly DemucsCastOwner _casts = new();
    /// <summary>Parity-debug only: one-shot probe fired with the next encoder conv output (pre-GELU). Not used in production.</summary>
    internal static Action<Tensor>? ConvProbe;
    /// <summary>Parity-debug only: one-shot probe fired with the next encoder post-DConv output. Not used in production.</summary>
    internal static Action<Tensor>? DConvProbe;
    private readonly bool _is2d, _decoder, _last;
    private readonly int _inCh, _outCh, _kernel, _stride;
    private Tensor? _convW, _convB, _rewriteW, _rewriteB;
    private readonly DemucsDConv _dconv;

    public DemucsConvBlock(int inCh, int outCh, int kernel, int stride, bool is2d, bool decoder, bool last = false)
    {
        _inCh = inCh; _outCh = outCh; _kernel = kernel; _stride = stride; _is2d = is2d; _decoder = decoder; _last = last;
        // DConv operates after GELU(conv) (encoder → outCh) or after rewrite→GLU (decoder → inCh, before conv_tr).
        _dconv = new DemucsDConv(decoder ? inCh : outCh);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        string convKey = _decoder ? "conv_tr" : "conv";
        _convW = _casts.F32(w, $"{prefix}.{convKey}.weight"); _convB = _casts.Optional(w, $"{prefix}.{convKey}.bias");
        _rewriteW = _casts.F32(w, $"{prefix}.rewrite.weight"); _rewriteB = _casts.Optional(w, $"{prefix}.rewrite.bias");
        _dconv.LoadWeights(w, $"{prefix}.dconv");
    }

    /// <summary>Encoder forward. Channels-first <c>[1, inCh, F, T]</c> (2D) or <c>[1, inCh, T]</c> (1D);
    /// returns the downsampled features. dconv runs on the conv output (per freq-row in 2D).</summary>
    public Tensor EncodeForward(IBackend backend, Tensor x, int f, int t)
    {
        int pad = (_kernel - _stride) / 2;
        // Time branch (demucs HEncLayer.forward): pad the input length up to a multiple of stride first.
        Tensor xin = x; bool ownXin = false; int tIn = t;
        if (!_is2d && t % _stride != 0)
        {
            tIn = t + (_stride - t % _stride);
            xin = new(new TensorShape(1, _inCh, tIn), DType.F32);
            ownXin = true;
            float* xp = (float*)xin.DataPointer; float* xsrc = (float*)x.DataPointer;
            for (int c = 0; c < _inCh; c++)
            {
                for (int j = 0; j < t; j++) xp[(long)c * tIn + j] = xsrc[(long)c * t + j];
                for (int j = t; j < tIn; j++) xp[(long)c * tIn + j] = 0f;
            }
        }
        int outF = _is2d ? (f + 2 * pad - _kernel) / _stride + 1 : f;
        int outT = _is2d ? t : (tIn + 2 * pad - _kernel) / _stride + 1;
        // conv: freq branch strides freq only (kernel (k,1), stride (s,1), pad (pad,0)); time branch strides time.
        Tensor conv = _is2d ? Conv2(backend, xin, _convW!, _convB, _outCh, outF, outT, _stride, 1, pad, 0)
            : Conv1(backend, xin, _convW!, _convB, _outCh, outT, _stride, pad);
        if (ownXin) xin.Dispose();
        if (ConvProbe is not null) { Action<Tensor> p = ConvProbe; ConvProbe = null; p(conv); }   // one-shot: enc0 conv
        GeluInPlace(backend, conv);
        DConvApply(backend, conv, _outCh, _is2d ? outF : 1, _is2d ? outT : outT);
        if (DConvProbe is not null) { Action<Tensor> p = DConvProbe; DConvProbe = null; p(conv); }   // one-shot: enc0 post-dconv
        int n = _is2d ? outF * outT : outT;
        Tensor rew = _is2d ? Conv2(backend, conv, _rewriteW!, _rewriteB, 2 * _outCh, outF, outT, 1, 1, 0, 0)
            : Conv1(backend, conv, _rewriteW!, _rewriteB, 2 * _outCh, outT, 1, 0);
        conv.Dispose();
        Tensor outT2 = Glu(rew, _outCh, n); rew.Dispose();
        return outT2;
    }

    /// <summary>Decoder forward over <c>[1, inCh, F, T]</c>/<c>[1, inCh, T]</c> (skip already added) → upsampled
    /// features. <paramref name="length"/> is the target time length for the 1D branch trim.</summary>
    public Tensor DecodeForward(IBackend backend, Tensor x, int f, int t, int length)
    {
        // rewrite is 3x3 (pad 1) in the decoder → GLU.
        int n = _is2d ? f * t : t;
        Tensor rew = _is2d ? Conv2(backend, x, _rewriteW!, _rewriteB, 2 * _inCh, f, t, 1, 1, 1, 1)
            : Conv1(backend, x, _rewriteW!, _rewriteB, 2 * _inCh, t, 1, 1);
        Tensor y = Glu(rew, _inCh, n); rew.Dispose();
        DConvApply(backend, y, _inCh, _is2d ? f : 1, _is2d ? t : t);

        int pad = (_kernel - _stride) / 2;
        if (_is2d)
        {
            // ConvTranspose2d on freq (stride (s,1)); produces (f-1)*s + k, then trim `pad` freq rows each side.
            int rawF = (f - 1) * _stride + _kernel;
            Tensor raw = ConvT2(backend, y, _convW!, _convB, _outCh, rawF, t, _stride, 1, 0, 0); y.Dispose();
            Tensor z = TrimFreq(raw, _outCh, rawF, t, pad); raw.Dispose();
            if (!_last) GeluInPlace(backend, z);
            return z;
        }
        else
        {
            // ConvTranspose1d on time; produces (t-1)*s + k, then crop [pad : pad+length].
            int rawT = (t - 1) * _stride + _kernel;
            Tensor raw = ConvT1(backend, y, _convW!, _convB, _outCh, rawT, _stride, 0); y.Dispose();
            Tensor z = CropTime(raw, _outCh, rawT, pad, length); raw.Dispose();
            if (!_last) GeluInPlace(backend, z);
            return z;
        }
    }

    /// <summary>Applies the DConv residual branch. 1D: in place on [1,C,T]. 2D: per freq-row (each [1,C,T]
    /// slice independently, matching demucs's permute(0,2,1,3).reshape(B*Fr,C,T)).</summary>
    private void DConvApply(IBackend backend, Tensor x, int c, int f, int t)
    {
        if (!_is2d) { _dconv.Forward(backend, x, t); return; }
        float* xp = (float*)x.DataPointer;
        Tensor row = new(new TensorShape(1, c, t), DType.F32);
        float* rp = (float*)row.DataPointer;
        for (int fr = 0; fr < f; fr++)
        {
            for (int ch = 0; ch < c; ch++)
                for (int j = 0; j < t; j++) rp[(long)ch * t + j] = xp[(((long)ch * f + fr) * t) + j];
            _dconv.Forward(backend, row, t);
            for (int ch = 0; ch < c; ch++)
                for (int j = 0; j < t; j++) xp[(((long)ch * f + fr) * t) + j] = rp[(long)ch * t + j];
        }
        row.Dispose();
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] a = [_convW, _convB, _rewriteW, _rewriteB];
        foreach (Tensor? t in a) if (t is not null) yield return t;
        foreach (Tensor t in _dconv.EnumerateWeights()) yield return t;
    }

    private static Tensor Conv1(IBackend b, Tensor x, Tensor w, Tensor? bias, int outCh, int outT, int stride, int pad)
    {
        Tensor o = new(new TensorShape(1, outCh, outT), DType.F32);
        b.Conv1d(o, x, w, bias, stride, pad, pad, 1, 1);
        return o;
    }

    private static Tensor ConvT1(IBackend b, Tensor x, Tensor w, Tensor? bias, int outCh, int outT, int stride, int pad)
    {
        Tensor o = new(new TensorShape(1, outCh, outT), DType.F32);
        b.ConvTranspose1d(o, x, w, bias, stride, pad, pad, 1, 1);
        return o;
    }

    private static Tensor Conv2(IBackend b, Tensor x, Tensor w, Tensor? bias, int outCh, int outF, int outT, int sH, int sW, int pH, int pW)
    {
        Tensor o = new(new TensorShape(1, outCh, outF, outT), DType.F32);
        b.Conv2D(o, x, w, bias, sH, sW, pH, pW);
        return o;
    }

    private static Tensor ConvT2(IBackend b, Tensor x, Tensor w, Tensor? bias, int outCh, int outF, int outT, int sH, int sW, int pH, int pW)
    {
        Tensor o = new(new TensorShape(1, outCh, outF, outT), DType.F32);
        b.ConvTranspose2d(o, x, w, bias, sH, sW, pH, pW);
        return o;
    }

    private static Tensor TrimFreq(Tensor x, int c, int f, int t, int pad)
    {
        int of = f - 2 * pad;
        Tensor o = new(new TensorShape(1, c, of, t), DType.F32);
        float* xp = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        for (int ch = 0; ch < c; ch++)
            for (int i = 0; i < of; i++)
                for (int j = 0; j < t; j++)
                    op[(((long)ch * of + i) * t) + j] = xp[(((long)ch * f + (i + pad)) * t) + j];
        return o;
    }

    private static Tensor CropTime(Tensor x, int c, int rawT, int pad, int length)
    {
        Tensor o = new(new TensorShape(1, c, length), DType.F32);
        float* xp = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        for (int ch = 0; ch < c; ch++)
            for (int j = 0; j < length; j++) op[(long)ch * length + j] = xp[(long)ch * rawT + (pad + j)];
        return o;
    }

    private static void GeluInPlace(IBackend b, Tensor x) => Activations.ErfGelu(x);   // demucs uses F.gelu (exact erf)

    /// <summary>Plain GLU: split channels in half → <c>a · σ(b)</c>. Input <c>[1, 2·outCh, …]</c> (n spatial).</summary>
    private Tensor Glu(Tensor x, int outCh, int n)
    {
        Tensor o = new(_is2d ? new TensorShape(1, outCh, (int)x.Shape[2], (int)x.Shape[3]) : new TensorShape(1, outCh, n), DType.F32);
        float* xp = (float*)x.DataPointer; float* op = (float*)o.DataPointer;
        for (int c = 0; c < outCh; c++)
            for (int j = 0; j < n; j++)
            {
                float a = xp[(long)c * n + j];
                float bb = xp[(long)(outCh + c) * n + j];
                op[(long)c * n + j] = a * (1f / (1f + MathF.Exp(-bb)));
            }
        return o;
    }

    /// <summary>Frees the F32 casts this block and its DConv allocated at load time.</summary>
    public void Dispose()
    {
        _dconv.Dispose();
        _casts.Dispose();
    }
}
