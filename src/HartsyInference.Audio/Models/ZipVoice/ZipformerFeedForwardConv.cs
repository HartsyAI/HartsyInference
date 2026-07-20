using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>FeedforwardModule: <c>in_proj (Linear) → SwooshL → out_proj (Linear)</c>. Three independent
/// instances per Zipformer layer (<c>feed_forward1/2/3</c>, macaron-style) with hidden dims
/// <c>3/4·ff</c>, <c>1·ff</c>, <c>5/4·ff</c> respectively.</summary>
internal sealed unsafe class ZipformerFeedForward
{
    private readonly int _dim, _hidden;
    private Tensor? _inW, _inB, _outW, _outB;

    public ZipformerFeedForward(int dim, int hidden) { _dim = dim; _hidden = hidden; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _inW = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.weight"]);
        _inB = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.bias"]);
        _outW = WhisperOps.EnsureF32(w[$"{prefix}.out_proj.weight"]);
        _outB = WhisperOps.EnsureF32(w[$"{prefix}.out_proj.bias"]);
    }

    /// <summary><paramref name="x"/> is <c>[1, T, dim]</c>. Returns <c>[1, T, dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int t)
    {
        Tensor h = WhisperOps.ProjectLinear(backend, x, _inW!, _inB, 1, t, _dim, _hidden);
        ZipformerScaling.SwooshLInPlace(h);
        Tensor output = WhisperOps.ProjectLinear(backend, h, _outW!, _outB, 1, t, _hidden, _dim);
        h.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_inW, _inB, _outW, _outB];
        foreach (Tensor? tt in all) if (tt is not null) yield return tt;
    }
}

/// <summary>ConvolutionModule: <c>in_proj (dim→2dim) → chunk[content, gate] → content·sigmoid(gate) →
/// depthwise Conv1d (kernel, SAME padding, non-causal) → SwooshR → out_proj (dim→dim)</c>. Two independent
/// instances per Zipformer layer (<c>conv_module1</c>, <c>conv_module2</c>), each with its own full weight
/// set (not tied).</summary>
internal sealed unsafe class ZipformerConvModule
{
    private readonly int _dim, _kernel;
    private Tensor? _inW, _inB, _dwW, _dwB, _outW, _outB;

    public ZipformerConvModule(int dim, int kernel) { _dim = dim; _kernel = kernel; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _inW = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.weight"]);
        _inB = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.bias"]);
        _dwW = WhisperOps.EnsureF32(w[$"{prefix}.depthwise_conv.weight"]);
        _dwB = WhisperOps.EnsureF32(w[$"{prefix}.depthwise_conv.bias"]);
        _outW = WhisperOps.EnsureF32(w[$"{prefix}.out_proj.weight"]);
        _outB = WhisperOps.EnsureF32(w[$"{prefix}.out_proj.bias"]);
    }

    /// <summary><paramref name="x"/> is <c>[1, T, dim]</c>. Returns <c>[1, T, dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int t)
    {
        int dim = _dim;
        Tensor proj = WhisperOps.ProjectLinear(backend, x, _inW!, _inB, 1, t, dim, 2 * dim);
        float* pp = (float*)proj.DataPointer;

        Tensor gated = new(new TensorShape(1, t, dim), DType.F32);
        float* gp = (float*)gated.DataPointer;
        for (int i = 0; i < t; i++)
        {
            float* row = pp + (long)i * 2 * dim;
            float* gRow = gp + (long)i * dim;
            for (int d = 0; d < dim; d++)
            {
                float gate = 1f / (1f + MathF.Exp(-row[dim + d]));
                gRow[d] = row[d] * gate;
            }
        }
        proj.Dispose();

        Tensor chFirst = new(new TensorShape(1, dim, t), DType.F32);
        backend.Transpose2D(chFirst, gated, t, dim);
        gated.Dispose();

        int pad = _kernel / 2;
        Tensor conv = new(new TensorShape(1, dim, t), DType.F32);
        backend.Conv1d(conv, chFirst, _dwW!, _dwB, stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: dim);
        chFirst.Dispose();

        Tensor back = new(new TensorShape(1, t, dim), DType.F32);
        backend.Transpose2D(back, conv, dim, t);
        conv.Dispose();

        ZipformerScaling.SwooshRInPlace(back);
        Tensor output = WhisperOps.ProjectLinear(backend, back, _outW!, _outB, 1, t, dim, dim);
        back.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_inW, _inB, _dwW, _dwB, _outW, _outB];
        foreach (Tensor? tt in all) if (tt is not null) yield return tt;
    }
}
