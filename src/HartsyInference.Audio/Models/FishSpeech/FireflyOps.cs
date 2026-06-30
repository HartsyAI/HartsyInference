using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;

namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Shared building blocks for the Fish-Speech firefly-gan-vq codec (fish-speech 1.5
/// <c>firefly.py</c>/<c>fsq.py</c>). Reproduces the upstream <c>FishConvNet</c> (causal constant
/// left-pad), <c>FishTransConvNet</c> (transpose conv + right-trim), and <c>ConvNeXtBlock</c>
/// (depthwise conv → channels-last LayerNorm → GELU MLP → LayerScale → residual) exactly.</summary>
internal static unsafe class FireflyOps
{
    /// <summary>FishConvNet (stride 1): causal constant left-pad of <c>(kernel-1)*dilation</c>, no right pad.
    /// Supports depthwise (<paramref name="groups"/> = channels). Returns <c>[1, cOut, t]</c>.</summary>
    public static Tensor CausalConv(IBackend backend, Tensor x, Tensor w, Tensor? b,
        int cIn, int t, int cOut, int kernel, int dilation, int groups)
    {
        int padLeft = (kernel - 1) * dilation;
        Tensor outp = new(new TensorShape(1, cOut, t), DType.F32);
        backend.Conv1d(outp, x, w, b, stride: 1, padLeft: padLeft, padRight: 0, dilation: dilation, groups: groups);
        return outp;
    }

    /// <summary>FishTransConvNet: ConvTranspose1d(stride) then trim <c>(kernel-stride)</c> samples off the right
    /// (causal). Returns <c>[1, cOut, t*stride]</c>.</summary>
    public static Tensor TransConv(IBackend backend, Tensor x, Tensor w, Tensor? b,
        int cIn, int t, int cOut, int kernel, int stride)
    {
        int padRight = kernel - stride;
        int outT = (t - 1) * stride + kernel - padRight;        // = t * stride
        Tensor outp = new(new TensorShape(1, cOut, outT), DType.F32);
        backend.ConvTranspose1d(outp, x, w, b, stride, padLeft: 0, padRight: padRight, dilation: 1, groups: 1);
        return outp;
    }

    /// <summary>Loads a Conv1d weight from <paramref name="baseKey"/> (the <c>...conv</c> module), fusing weight-norm
    /// when present (<c>parametrizations.weight.original0/1</c> or legacy <c>weight_g/weight_v</c>, both dim=0).</summary>
    public static Tensor LoadConvWeight(IReadOnlyDictionary<string, Tensor> w, string baseKey)
    {
        if (w.TryGetValue($"{baseKey}.weight_g", out Tensor? g) && w.TryGetValue($"{baseKey}.weight_v", out Tensor? v))
            return WeightNormFusion.Fuse(WhisperOps.EnsureF32(g), WhisperOps.EnsureF32(v));
        if (w.TryGetValue($"{baseKey}.parametrizations.weight.original0", out Tensor? g2)
            && w.TryGetValue($"{baseKey}.parametrizations.weight.original1", out Tensor? v2))
            return WeightNormFusion.Fuse(WhisperOps.EnsureF32(g2), WhisperOps.EnsureF32(v2));
        return WhisperOps.EnsureF32(w[$"{baseKey}.weight"]);
    }

    /// <summary>Loads a ConvTranspose1d weight (<c>[C_in, C_out, K]</c>), fusing weight-norm along dim=0
    /// (per input channel) when present.</summary>
    public static Tensor LoadTransConvWeight(IReadOnlyDictionary<string, Tensor> w, string baseKey)
    {
        if (w.TryGetValue($"{baseKey}.weight_g", out Tensor? g) && w.TryGetValue($"{baseKey}.weight_v", out Tensor? v))
            return WeightNormFusionT.Fuse(WhisperOps.EnsureF32(g), WhisperOps.EnsureF32(v));
        if (w.TryGetValue($"{baseKey}.parametrizations.weight.original0", out Tensor? g2)
            && w.TryGetValue($"{baseKey}.parametrizations.weight.original1", out Tensor? v2))
            return WeightNormFusionT.Fuse(WhisperOps.EnsureF32(g2), WhisperOps.EnsureF32(v2));
        return WhisperOps.EnsureF32(w[$"{baseKey}.weight"]);
    }
}

/// <summary>ConvNeXt block (fish-speech <c>firefly.ConvNeXtBlock</c>, channels-last variant): depthwise FishConvNet
/// (k=7, causal) → permute to <c>[B,L,C]</c> → LayerNorm(eps 1e-6) → Linear(C→4C) → GELU(erf) → Linear(4C→C) →
/// per-channel LayerScale (<c>gamma</c>) → permute back → residual add. Used in the firefly quantizer up/down stacks.</summary>
internal sealed unsafe class FireflyConvNeXtBlock : IDisposable
{
    private readonly int _dim;
    private readonly int _hidden;
    private readonly int _kernel;
    private Tensor? _dwW, _dwB, _normW, _normB, _pw1W, _pw1B, _pw2W, _pw2B, _gamma;
    private int _disposed;

    public FireflyConvNeXtBlock(int dim, int kernelSize = 7, float mlpRatio = 4f)
    {
        _dim = dim;
        _hidden = (int)(mlpRatio * dim);
        _kernel = kernelSize;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _dwW = FireflyOps.LoadConvWeight(w, $"{prefix}.dwconv.conv");
        _dwB = WhisperOps.EnsureF32(w[$"{prefix}.dwconv.conv.bias"]);
        _normW = WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]);
        _normB = WhisperOps.EnsureF32(w[$"{prefix}.norm.bias"]);
        _pw1W = WhisperOps.EnsureF32(w[$"{prefix}.pwconv1.weight"]);
        _pw1B = WhisperOps.EnsureF32(w[$"{prefix}.pwconv1.bias"]);
        _pw2W = WhisperOps.EnsureF32(w[$"{prefix}.pwconv2.weight"]);
        _pw2B = WhisperOps.EnsureF32(w[$"{prefix}.pwconv2.bias"]);
        _gamma = WhisperOps.EnsureF32(w[$"{prefix}.gamma"]);
    }

    /// <summary>Forward — <paramref name="x"/> channels-first <c>[1, dim, t]</c>. Returns a fresh <c>[1, dim, t]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int t)
    {
        if (_dwW is null) throw new InvalidOperationException("FireflyConvNeXtBlock weights not loaded.");

        // Depthwise causal conv (groups = dim).
        Tensor dw = FireflyOps.CausalConv(backend, x, _dwW!, _dwB, _dim, t, _dim, _kernel, dilation: 1, groups: _dim);

        // Permute [1,dim,t] -> [1,t,dim], LayerNorm over the channel dim.
        Tensor cl = new(new TensorShape(1, t, _dim), DType.F32);
        backend.Transpose2D(cl, dw, _dim, t);
        dw.Dispose();
        Tensor ln = new(cl.Shape, DType.F32);
        backend.LayerNorm(ln, cl, _normW!, _normB!, 1e-6f);
        cl.Dispose();

        // MLP: Linear dim->4dim, GELU(erf), Linear 4dim->dim.
        Tensor h1 = WhisperOps.ProjectLinear(backend, ln, _pw1W!, _pw1B, 1, t, _dim, _hidden);
        ln.Dispose();
        Activations.ErfGelu(h1);
        Tensor h2 = WhisperOps.ProjectLinear(backend, h1, _pw2W!, _pw2B, 1, t, _hidden, _dim);
        h1.Dispose();

        // Per-channel LayerScale (gamma) over the last dim.
        float* hp = (float*)h2.DataPointer;
        float* gp = (float*)_gamma!.DataPointer;
        for (int ti = 0; ti < t; ti++)
            for (int c = 0; c < _dim; c++) hp[(long)ti * _dim + c] *= gp[c];

        // Permute back [1,t,dim] -> [1,dim,t] and residual add.
        Tensor cf = new(new TensorShape(1, _dim, t), DType.F32);
        backend.Transpose2D(cf, h2, t, _dim);
        h2.Dispose();
        Tensor result = new(new TensorShape(1, _dim, t), DType.F32);
        backend.Add(result, x, cf);
        cf.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_dwW, _dwB, _normW, _normB, _pw1W, _pw1B, _pw2W, _pw2B, _gamma];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
