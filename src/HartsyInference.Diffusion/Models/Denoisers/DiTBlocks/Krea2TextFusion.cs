using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Krea 2 text-fusion stage (<c>Krea2TextFusion</c>). Collapses a stack of tapped text-encoder hidden states
/// <c>[B, S, numTextLayers, dim]</c> into a single refined sequence <c>[B, S, dim]</c>:
/// <list type="number">
///   <item><c>numLayerwise</c> pre-norm blocks attend across the <b>layer axis</b> (each token independently) —
///         <c>[B·S, numTextLayers, dim]</c>.</item>
///   <item>a <c>Linear(numTextLayers → 1)</c> <c>projector</c> collapses the layer axis.</item>
///   <item><c>numRefiner</c> pre-norm blocks attend across the <b>token axis</b> — <c>[B, S, dim]</c>.</item>
/// </list>
/// All blocks are <see cref="Krea2TextFusionBlock"/> (no RoPE, no timestep modulation, full MHA).</summary>
public sealed unsafe class Krea2TextFusion
{
    private readonly int _numTextLayers;
    private readonly int _dim;
    private readonly Krea2TextFusionBlock[] _layerwise;
    private readonly Krea2TextFusionBlock[] _refiner;
    private Tensor? _projector; // [1, numTextLayers]

    public Krea2TextFusion(int numTextLayers, int dim, int numHeads, int numKvHeads, int intermediateSize,
        int numLayerwise, int numRefiner, float eps)
    {
        _numTextLayers = numTextLayers;
        _dim = dim;
        _layerwise = new Krea2TextFusionBlock[numLayerwise];
        _refiner = new Krea2TextFusionBlock[numRefiner];
        for (int i = 0; i < numLayerwise; i++) _layerwise[i] = new Krea2TextFusionBlock(dim, numHeads, numKvHeads, intermediateSize, eps);
        for (int i = 0; i < numRefiner; i++) _refiner[i] = new Krea2TextFusionBlock(dim, numHeads, numKvHeads, intermediateSize, eps);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        for (int i = 0; i < _layerwise.Length; i++) _layerwise[i].LoadWeights(w, $"{p}.layerwise_blocks.{i}");
        for (int i = 0; i < _refiner.Length; i++) _refiner[i].LoadWeights(w, $"{p}.refiner_blocks.{i}");
        _projector = F32(w[$"{p}.projector.weight"]); // [1, numTextLayers]
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Krea2TextFusionBlock b in _layerwise) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (Krea2TextFusionBlock b in _refiner) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        if (_projector is not null) yield return _projector;
    }

    /// <summary>Fuses <paramref name="encoderHidden"/> <c>[B, S, numTextLayers, dim]</c> → <c>[B, S, dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor encoderHidden, int batch, int seqLen)
    {
        int bs = batch * seqLen;
        // [B, S, L, dim] is contiguous as [B·S, L, dim]; run layerwise blocks with that as the (batch, seq).
        Tensor layerStack = new Tensor(new TensorShape(bs, _numTextLayers, _dim), DType.F32);
        long bytes = (long)bs * _numTextLayers * _dim * sizeof(float);
        Buffer.MemoryCopy((void*)encoderHidden.DataPointer, (void*)layerStack.DataPointer, bytes, bytes);

        for (int i = 0; i < _layerwise.Length; i++)
        {
            Tensor next = _layerwise[i].Forward(backend, layerStack, bs, _numTextLayers);
            layerStack.Dispose();
            layerStack = next;
        }

        // projector: out[b,s,d] = Σ_l proj[0,l] · layerStack[(b·s), l, d]
        Tensor fused = new Tensor(new TensorShape(batch, seqLen, _dim), DType.F32);
        float* src = (float*)layerStack.DataPointer;
        float* pj = (float*)_projector!.DataPointer;
        float* dst = (float*)fused.DataPointer;
        for (int t = 0; t < bs; t++)
            for (int d = 0; d < _dim; d++)
            {
                float acc = 0f;
                for (int l = 0; l < _numTextLayers; l++)
                    acc += pj[l] * src[((long)t * _numTextLayers + l) * _dim + d];
                dst[(long)t * _dim + d] = acc;
            }
        layerStack.Dispose();

        for (int i = 0; i < _refiner.Length; i++)
        {
            Tensor next = _refiner[i].Forward(backend, fused, batch, seqLen);
            fused.Dispose();
            fused = next;
        }
        return fused;
    }

    private static Tensor F32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}

/// <summary>Pre-norm transformer block used by <see cref="Krea2TextFusion"/> (no RoPE, no timestep modulation):
/// <c>h = h + attn(norm1(h)); h = h + ff(norm2(h))</c>. Attention is full MHA with the Krea 2 sigmoid output gate.</summary>
public sealed unsafe class Krea2TextFusionBlock
{
    private readonly int _dim;
    private readonly int _ffnInner;
    private readonly float _eps;
    private readonly Krea2Attention _attn;
    private Tensor? _norm1, _norm2, _ffGate, _ffUp, _ffDown;

    public Krea2TextFusionBlock(int dim, int numHeads, int numKvHeads, int intermediateSize, float eps)
    {
        _dim = dim;
        _ffnInner = intermediateSize;
        _eps = eps;
        _attn = new Krea2Attention(dim, numHeads, numKvHeads, eps);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _norm1 = Krea2Norm.LoadZeroCentered(w[$"{p}.norm1.weight"]);
        _norm2 = Krea2Norm.LoadZeroCentered(w[$"{p}.norm2.weight"]);
        _attn.LoadWeights(w, $"{p}.attn");
        _ffGate = w[$"{p}.ff.gate.weight"];
        _ffUp = w[$"{p}.ff.up.weight"];
        _ffDown = w[$"{p}.ff.down.weight"];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_norm1 is not null) yield return _norm1;
        if (_norm2 is not null) yield return _norm2;
        foreach (Tensor t in _attn.EnumerateWeights()) yield return t;
        if (_ffGate is not null) yield return _ffGate;
        if (_ffUp is not null) yield return _ffUp;
        if (_ffDown is not null) yield return _ffDown;
    }

    public Tensor Forward(IBackend backend, Tensor hidden, int batch, int seqLen)
    {
        TensorShape hShape = new TensorShape(batch, seqLen, _dim);
        Tensor n1 = new Tensor(hShape, DType.F32);
        backend.RmsNorm(n1, hidden, _norm1!, _eps);
        Tensor attnOut = _attn.Forward(backend, n1, null, batch, seqLen);
        n1.Dispose();
        Tensor h1 = new Tensor(hShape, DType.F32);
        backend.Add(h1, hidden, attnOut);
        attnOut.Dispose();

        Tensor n2 = new Tensor(hShape, DType.F32);
        backend.RmsNorm(n2, h1, _norm2!, _eps);
        Tensor ffOut = SwiGlu(backend, n2, batch, seqLen);
        n2.Dispose();
        Tensor outp = new Tensor(hShape, DType.F32);
        backend.Add(outp, h1, ffOut);
        ffOut.Dispose(); h1.Dispose();
        return outp;
    }

    private Tensor SwiGlu(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnInner);
        Tensor g = new Tensor(ffShape, DType.F32);
        Tensor u = new Tensor(ffShape, DType.F32);
        backend.Linear(g, input, _ffGate!, null);
        backend.Linear(u, input, _ffUp!, null);
        Tensor act = new Tensor(ffShape, DType.F32);
        backend.Silu(act, g);
        g.Dispose();
        Tensor gated = new Tensor(ffShape, DType.F32);
        backend.Mul(gated, act, u);
        act.Dispose(); u.Dispose();
        Tensor outp = new Tensor(new TensorShape(batch, seqLen, _dim), DType.F32);
        backend.Linear(outp, gated, _ffDown!, null);
        gated.Dispose();
        return outp;
    }
}
