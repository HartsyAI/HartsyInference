using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan-Animate face encoder (<c>FaceEncoder</c> in ComfyUI <c>comfy/ldm/wan/model_animate.py</c>): turns
/// per-frame motion vectors <c>[T, in_dim]</c> (B=1) into temporally-aligned face features
/// <c>[T', num_heads+1, out_dim]</c>. <c>conv1_local</c> (CausalConv1d, stride 1) expands to <c>num_heads</c>
/// parallel 1024-wide streams (the head axis folds into the batch, so each stream runs the rest independently);
/// <c>conv2</c>/<c>conv3</c> (CausalConv1d, stride 2 each) downsample 4× temporally; each conv is followed by
/// no-affine LayerNorm + SiLU; <c>out_proj</c> maps 1024 → out_dim and the learnable <c>padding_tokens</c> row is
/// appended along the head axis. CausalConv1d = replicate-pad <c>kernel−1</c> on the left, then a plain Conv1d.
/// All dims derive from the loaded weights (real: in 512, hidden 1024, heads 4).</summary>
public sealed unsafe class WanAnimateFaceEncoder
{
    private readonly float _eps;
    private int _inDim, _outDim, _hiddenDim, _numHeads, _kernel;

    private Tensor? _conv1W, _conv1B, _conv2W, _conv2B, _conv3W, _conv3B;
    private Tensor? _outW, _outB, _paddingTokens;   // padding_tokens [1,1,1,out_dim]

    public WanAnimateFaceEncoder(float eps = 1e-6f)
    {
        _eps = eps;
    }

    /// <summary>Number of parallel head streams (+1 padding token in the output). Valid after <see cref="LoadWeights"/>.</summary>
    public int NumHeads => _numHeads;

    /// <summary>Loads the <c>face_encoder.*</c> subtree — CausalConv1d wraps its conv as <c>.conv</c>, so the weight
    /// keys are <c>conv1_local.conv.weight</c> / <c>conv2.conv.weight</c> / <c>conv3.conv.weight</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _conv1W = LoadF32(w, $"{p}.conv1_local.conv.weight"); w.TryGetValue($"{p}.conv1_local.conv.bias", out _conv1B);
        _conv2W = LoadF32(w, $"{p}.conv2.conv.weight"); w.TryGetValue($"{p}.conv2.conv.bias", out _conv2B);
        _conv3W = LoadF32(w, $"{p}.conv3.conv.weight"); w.TryGetValue($"{p}.conv3.conv.bias", out _conv3B);
        _outW = w[$"{p}.out_proj.weight"]; w.TryGetValue($"{p}.out_proj.bias", out _outB);
        _paddingTokens = LoadF32(w, $"{p}.padding_tokens");
        _inDim = (int)_conv1W.Shape[1];
        _hiddenDim = (int)_conv2W.Shape[1];
        _numHeads = (int)(_conv1W.Shape[0] / _hiddenDim);
        _kernel = (int)_conv1W.Shape[2];
        _outDim = (int)_outW.Shape[0];
        if (_numHeads * _hiddenDim != (int)_conv1W.Shape[0])
            throw new ArgumentException($"conv1_local out channels {_conv1W.Shape[0]} not a multiple of hidden {_hiddenDim}.", nameof(w));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _conv1W, _conv1B, _conv2W, _conv2B, _conv3W, _conv3B, _outW, _outB, _paddingTokens })
            if (t is not null) yield return t;
    }

    /// <summary>Encodes motion vectors <c>[T, in_dim]</c> (single sample) → face features <c>[T', num_heads+1, out_dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor motion)
    {
        int tIn = (int)motion.Shape[0];

        // conv1_local: [in_dim, T] → [hidden·num_heads, T] (stride 1, causal replicate pad). Channel layout is
        // head-major ("b (n c) t"), so head h owns rows [h·hidden, (h+1)·hidden) and runs independently below.
        Tensor c1 = Conv1dCausal(backend, Transpose(motion, tIn, _inDim), _inDim, _hiddenDim * _numHeads, _conv1W!, _conv1B, 1, tIn);
        int t1 = (int)c1.Shape[1];
        Tensor[] headOut = new Tensor[_numHeads];
        int tFinal = -1;
        for (int h = 0; h < _numHeads; h++)
        {
            Tensor headSeq = SliceRowsBlock(c1, h * _hiddenDim, _hiddenDim, t1);   // [hidden, T]
            Tensor x = ChannelsToTokens(headSeq, _hiddenDim, t1); headSeq.Dispose();   // [T, hidden]
            LayerNormSilu(x, t1, _hiddenDim);
            Tensor c2 = Conv1dCausal(backend, Transpose(x, t1, _hiddenDim), _hiddenDim, _hiddenDim, _conv2W!, _conv2B, 2, t1); x.Dispose();
            int t2 = (int)c2.Shape[1];
            Tensor x2 = ChannelsToTokens(c2, _hiddenDim, t2); c2.Dispose();
            LayerNormSilu(x2, t2, _hiddenDim);
            Tensor c3 = Conv1dCausal(backend, Transpose(x2, t2, _hiddenDim), _hiddenDim, _hiddenDim, _conv3W!, _conv3B, 2, t2); x2.Dispose();
            int t3 = (int)c3.Shape[1];
            Tensor x3 = ChannelsToTokens(c3, _hiddenDim, t3); c3.Dispose();
            LayerNormSilu(x3, t3, _hiddenDim);
            Tensor o = new Tensor(new TensorShape(t3, _outDim), DType.F32);
            backend.Linear(o, x3, _outW!, _outB); x3.Dispose();
            headOut[h] = o;
            tFinal = t3;
        }
        c1.Dispose();

        // Assemble [T', num_heads+1, out_dim]: per frame, the num_heads features + the learnable padding token.
        Tensor outT = new Tensor(new TensorShape(tFinal, _numHeads + 1, _outDim), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int t = 0; t < tFinal; t++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                float* hp = (float*)headOut[h].DataPointer + (long)t * _outDim;
                Buffer.MemoryCopy(hp, op + ((long)t * (_numHeads + 1) + h) * _outDim, (long)_outDim * 4, (long)_outDim * 4);
            }
            Buffer.MemoryCopy((float*)_paddingTokens!.DataPointer, op + ((long)t * (_numHeads + 1) + _numHeads) * _outDim,
                (long)_outDim * 4, (long)_outDim * 4);
        }
        foreach (Tensor h in headOut) h.Dispose();
        return outT;
    }

    /// <summary>Causal Conv1d over the frame axis: replicate-pads the left by <c>kernel-1</c>, then a strided Conv1d.
    /// <paramref name="input"/> is <c>[1, Cin, T]</c> (consumed), returns <c>[Cout, Tout]</c>.</summary>
    private Tensor Conv1dCausal(IBackend backend, Tensor input, int cin, int cout, Tensor weight, Tensor? bias, int stride, int tIn)
    {
        int pad = _kernel - 1;
        Tensor padded = new Tensor(new TensorShape(1, cin, tIn + pad), DType.F32);
        float* pp = (float*)padded.DataPointer, ip = (float*)input.DataPointer;
        for (int c = 0; c < cin; c++)
        {
            float first = ip[(long)c * tIn];   // replicate the first frame into the causal pad
            for (int t = 0; t < pad; t++) pp[(long)c * (tIn + pad) + t] = first;
            Buffer.MemoryCopy(ip + (long)c * tIn, pp + (long)c * (tIn + pad) + pad, (long)tIn * 4, (long)tIn * 4);
        }
        input.Dispose();
        int tOut = (tIn + pad - _kernel) / stride + 1;
        Tensor o = new Tensor(new TensorShape(1, cout, tOut), DType.F32);
        backend.Conv1d(o, padded, weight, bias, stride, 0, 0, 1, 1);
        padded.Dispose();
        // Drop the batch dim → [cout, tOut].
        Tensor flat = new Tensor(new TensorShape(cout, tOut), DType.F32);
        Buffer.MemoryCopy((float*)o.DataPointer, (float*)flat.DataPointer, (long)cout * tOut * 4, (long)cout * tOut * 4);
        o.Dispose();
        return flat;
    }

    /// <summary>[T, C] → [1, C, T] for Conv1d (channels-first with a batch dim).</summary>
    private static Tensor Transpose(Tensor x, int t, int c)
    {
        Tensor o = new Tensor(new TensorShape(1, c, t), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int ti = 0; ti < t; ti++)
            for (int ci = 0; ci < c; ci++) op[(long)ci * t + ti] = xp[(long)ti * c + ci];
        return o;
    }

    /// <summary>[C, T] → [T, C].</summary>
    private static Tensor ChannelsToTokens(Tensor x, int c, int t)
    {
        Tensor o = new Tensor(new TensorShape(t, c), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
            for (int ti = 0; ti < t; ti++) op[(long)ti * c + ci] = xp[(long)ci * t + ti];
        return o;
    }

    private static Tensor SliceRowsBlock(Tensor x, int startRow, int rows, int t)
    {
        Tensor o = new Tensor(new TensorShape(rows, t), DType.F32);
        Buffer.MemoryCopy((float*)x.DataPointer + (long)startRow * t, (float*)o.DataPointer, (long)rows * t * 4, (long)rows * t * 4);
        return o;
    }

    /// <summary>In-place no-affine LayerNorm over the channel dim followed by SiLU, on a <c>[T, C]</c> tensor.</summary>
    private void LayerNormSilu(Tensor x, int t, int c)
    {
        float* xp = (float*)x.DataPointer;
        for (int i = 0; i < t; i++)
        {
            long off = (long)i * c;
            double mean = 0; for (int d = 0; d < c; d++) mean += xp[off + d]; mean /= c;
            double var = 0; for (int d = 0; d < c; d++) { double dd = xp[off + d] - mean; var += dd * dd; }
            float inv = 1f / MathF.Sqrt((float)(var / c) + _eps);
            for (int d = 0; d < c; d++)
            {
                float n = (float)((xp[off + d] - mean) * inv);
                xp[off + d] = n / (1f + MathF.Exp(-n));   // SiLU
            }
        }
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
}
