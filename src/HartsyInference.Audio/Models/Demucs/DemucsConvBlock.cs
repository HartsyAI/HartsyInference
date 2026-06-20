using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Demucs;

/// <summary>HTDemucs HEncLayer/HDecLayer conv block, parameterized for the time branch (1D) and the spectrogram
/// branch (2D, stride only on the frequency axis). Encoder: main conv → GELU → 1×1 rewrite + plain GLU
/// (<c>a·σ(b)</c>). Decoder: 1×1 rewrite + GLU → transposed conv → GELU (skipped on the last layer). GroupNorm
/// and the DConv residual branch are off at the released depth-4 config (staged). Reuses <c>IBackend</c>
/// Conv1d/Conv2D/ConvTranspose1d/ConvTranspose2d + Gelu.</summary>
public sealed unsafe class DemucsConvBlock
{
    private readonly bool _is2d, _decoder, _last;
    private readonly int _inCh, _outCh, _kernel, _stride;
    private Tensor? _convW, _convB, _rewriteW, _rewriteB;

    public DemucsConvBlock(int inCh, int outCh, int kernel, int stride, bool is2d, bool decoder, bool last = false)
    {
        _inCh = inCh; _outCh = outCh; _kernel = kernel; _stride = stride; _is2d = is2d; _decoder = decoder; _last = last;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _convW = WhisperOps.EnsureF32(w[$"{prefix}.conv.weight"]); _convB = Bias(w, $"{prefix}.conv.bias");
        _rewriteW = WhisperOps.EnsureF32(w[$"{prefix}.rewrite.weight"]); _rewriteB = Bias(w, $"{prefix}.rewrite.bias");
    }

    /// <summary>Encoder forward. Channels-first <c>[1, inCh, F, T]</c> (2D) or <c>[1, inCh, T]</c> (1D);
    /// returns the downsampled features.</summary>
    public Tensor EncodeForward(IBackend backend, Tensor x, int f, int t)
    {
        int pad = (_kernel - _stride) / 2;
        int outF = _is2d ? (f + 2 * pad - _kernel) / _stride + 1 : f;
        int outT = _is2d ? t : (t + 2 * pad - _kernel) / _stride + 1;
        Tensor conv = _is2d
            ? Conv2(backend, x, _convW!, _convB, _outCh, outF, outT, _stride, 1, pad, 0)
            : Conv1(backend, x, _convW!, _convB, _outCh, outT, _stride, pad);
        GeluInPlace(backend, conv);
        int n = _is2d ? outF * outT : outT;
        Tensor rew = _is2d
            ? Conv2(backend, conv, _rewriteW!, _rewriteB, 2 * _outCh, outF, outT, 1, 1, 0, 0)
            : Conv1(backend, conv, _rewriteW!, _rewriteB, 2 * _outCh, outT, 1, 0);
        conv.Dispose();
        Tensor outT2 = Glu(rew, _outCh, n); rew.Dispose();
        return outT2;
    }

    /// <summary>Decoder forward over <c>[1, inCh, F, T]</c>/<c>[1, inCh, T]</c> → upsampled features.</summary>
    public Tensor DecodeForward(IBackend backend, Tensor x, int f, int t)
    {
        int n = _is2d ? f * t : t;
        Tensor rew = _is2d
            ? Conv2(backend, x, _rewriteW!, _rewriteB, 2 * _outCh, f, t, 1, 1, 0, 0)
            : Conv1(backend, x, _rewriteW!, _rewriteB, 2 * _outCh, t, 1, 0);
        Tensor g = Glu(rew, _outCh, n); rew.Dispose();
        int pad = (_kernel - _stride) / 2;
        int outF = _is2d ? (f - 1) * _stride + _kernel - 2 * pad : f;
        int outT = _is2d ? t : (t - 1) * _stride + _kernel - 2 * pad;
        Tensor outX = _is2d
            ? ConvT2(backend, g, _convW!, _convB, _outCh, outF, outT, _stride, 1, pad, 0)
            : ConvT1(backend, g, _convW!, _convB, _outCh, outT, _stride, pad);
        g.Dispose();
        if (!_last) GeluInPlace(backend, outX);
        return outX;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] a = [_convW, _convB, _rewriteW, _rewriteB];
        foreach (Tensor? t in a) if (t is not null) yield return t;
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

    private static void GeluInPlace(IBackend b, Tensor x) { Tensor t = new(x.Shape, DType.F32); b.Gelu(t, x); Buffer.MemoryCopy((void*)t.DataPointer, (void*)x.DataPointer, x.ElementCount * 4, x.ElementCount * 4); t.Dispose(); }

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

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) => w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
}
