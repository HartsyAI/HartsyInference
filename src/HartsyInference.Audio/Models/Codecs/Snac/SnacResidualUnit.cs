using HartsyInference.Audio.Models.Codecs.Dac;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Snac;

/// <summary>SNAC residual unit. Structurally identical to <see cref="DacResidualUnit"/>: <c>Snake → Conv1d(k=7, dilation) → Snake → Conv1d(k=1) → residual add</c>. Kept as a SNAC-namespaced class so the codecs stay independently maintainable; the math is identical and weights load with the same key conventions.</summary>
internal sealed unsafe class SnacResidualUnit
{
    private readonly string _prefix;
    private readonly int _dim;
    private readonly int _kernel;
    private readonly int _dilation;
    private readonly int _groups;

    private Tensor? _snake1Alpha;
    private Tensor? _snake2Alpha;
    private Tensor? _conv1W;
    private Tensor? _conv1B;
    private Tensor? _conv2W;
    private Tensor? _conv2B;

    public SnacResidualUnit(string prefix, int dim, int kernel, int dilation, int groups = 1)
    {
        _prefix = prefix;
        _dim = dim;
        _kernel = kernel;
        _dilation = dilation;
        _groups = groups;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _snake1Alpha = WhisperOps.EnsureF32(w[$"{_prefix}.block.0.alpha"]).Reshape(new TensorShape(_dim));
        _conv1W = LoadFusedWeight(w, $"{_prefix}.block.1");
        _conv1B = WhisperOps.EnsureF32(w[$"{_prefix}.block.1.bias"]);
        _snake2Alpha = WhisperOps.EnsureF32(w[$"{_prefix}.block.2.alpha"]).Reshape(new TensorShape(_dim));
        _conv2W = LoadFusedWeight(w, $"{_prefix}.block.3");
        _conv2B = WhisperOps.EnsureF32(w[$"{_prefix}.block.3.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        if (_conv1W is null) throw new InvalidOperationException($"SnacResidualUnit '{_prefix}' weights not loaded.");

        Tensor a1 = new(x.Shape, DType.F32);
        backend.Snake(a1, x, _snake1Alpha!, null);

        int pad = (_kernel - 1) * _dilation / 2;
        int tConv1 = t + 2 * pad - _dilation * (_kernel - 1);
        Tensor mid = new(new TensorShape(batch, _dim, tConv1), DType.F32);
        backend.Conv1d(mid, a1, _conv1W!, _conv1B,
            stride: 1, padLeft: pad, padRight: pad, dilation: _dilation, groups: _groups);
        a1.Dispose();

        Tensor a2 = new(mid.Shape, DType.F32);
        backend.Snake(a2, mid, _snake2Alpha!, null);
        mid.Dispose();

        Tensor proj = new(new TensorShape(batch, _dim, tConv1), DType.F32);
        backend.Conv1d(proj, a2, _conv2W!, _conv2B,
            stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        a2.Dispose();

        int tProj = (int)proj.Shape[2];
        int diff = t - tProj;
        int cropLeft = diff / 2;
        Tensor result = new(proj.Shape, DType.F32);

        if (diff > 0)
        {
            float* xp = (float*)x.DataPointer;
            float* pp = (float*)proj.DataPointer;
            float* rp = (float*)result.DataPointer;
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < _dim; c++)
                {
                    int srcXBase = (b * _dim + c) * t + cropLeft;
                    int srcPBase = (b * _dim + c) * tProj;
                    int dstBase = (b * _dim + c) * tProj;
                    for (int j = 0; j < tProj; j++) rp[dstBase + j] = xp[srcXBase + j] + pp[srcPBase + j];
                }
        }
        else
        {
            backend.Add(result, x, proj);
        }
        proj.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_snake1Alpha, _snake2Alpha, _conv1W, _conv1B, _conv2W, _conv2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    private static Tensor LoadFusedWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        return WeightNormFusion.Fuse(g, v);
    }
}
