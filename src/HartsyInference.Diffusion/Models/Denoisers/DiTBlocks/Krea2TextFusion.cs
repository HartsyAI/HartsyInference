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
        _projector = TensorCasts.EnsureF32(w[$"{p}.projector.weight"]); // [1, numTextLayers]
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Krea2TextFusionBlock b in _layerwise) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (Krea2TextFusionBlock b in _refiner) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        if (_projector is not null) yield return _projector;
    }

    /// <summary>Fuses <paramref name="encoderHidden"/> <c>[B, S, numTextLayers, dim]</c> → <c>[B, S, dim]</c>.</summary>
    // GPU-residency rewrite: the [B,S,L,dim]→[B·S,L,dim] reshape is a device-resident contiguous copy (SliceRows,
    // not a host Buffer.MemoryCopy over DataPointer), and the layer-axis projector collapse becomes Transpose2D
    // (move the L axis to the last dim) + Linear(L→1) — the exact Σ_l proj[0,l]·layerStack[t,l,d] contraction, run
    // on-device so the fusion stack never leaves the GPU between blocks.
    public Tensor Forward(IBackend backend, Tensor encoderHidden, int batch, int seqLen)
    {
        int bs = batch * seqLen;
        // [B, S, L, dim] is contiguous as [B·S, L, dim]; device-copy reshape, then run layerwise blocks with that
        // as the (batch, seq).
        Tensor layerStack = new Tensor(new TensorShape(bs, _numTextLayers, _dim), DType.F32);
        backend.SliceRows(layerStack, encoderHidden, 0);

        for (int i = 0; i < _layerwise.Length; i++)
        {
            Tensor next = _layerwise[i].Forward(backend, layerStack, bs, _numTextLayers);
            layerStack.Dispose();
            layerStack = next;
        }

        // projector: fused[t, d] = Σ_l proj[0, l] · layerStack[t, l, d]. Transpose the layer axis to the last dim
        // ([bs, L, dim] → [bs, dim, L]), then Linear(L → 1) contracts it → [bs, dim, 1] ≡ [B, S, dim].
        Tensor transposed = new Tensor(new TensorShape(bs, _dim, _numTextLayers), DType.F32);
        backend.Transpose2D(transposed, layerStack, _numTextLayers, _dim);
        layerStack.Dispose();
        Tensor fused = new Tensor(new TensorShape(batch, seqLen, _dim), DType.F32);
        backend.Linear(fused, transposed, _projector!, null);
        transposed.Dispose();

        for (int i = 0; i < _refiner.Length; i++)
        {
            Tensor next = _refiner[i].Forward(backend, fused, batch, seqLen);
            fused.Dispose();
            fused = next;
        }
        return fused;
    }
}

/// <summary>Pre-norm transformer block used by <see cref="Krea2TextFusion"/> (no RoPE, no timestep modulation): <c>h = h + attn(norm1(h)); h = h + ff(norm2(h))</c>, with the Krea 2 sigmoid output gate.</summary>
public sealed unsafe class Krea2TextFusionBlock
{
    private readonly int _dim;
    private readonly float _eps;
    private readonly Krea2Attention _attn;
    private readonly SwiGluFfn _ffn;
    private Tensor? _norm1, _norm2;

    public Krea2TextFusionBlock(int dim, int numHeads, int numKvHeads, int intermediateSize, float eps)
    {
        _dim = dim;
        _eps = eps;
        _attn = new Krea2Attention(dim, numHeads, numKvHeads, eps);
        _ffn = new SwiGluFfn(dim, intermediateSize);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _norm1 = Krea2Norm.LoadZeroCentered(w[$"{p}.norm1.weight"]);
        _norm2 = Krea2Norm.LoadZeroCentered(w[$"{p}.norm2.weight"]);
        _attn.LoadWeights(w, $"{p}.attn");
        _ffn.LoadSwiGluWeights(w[$"{p}.ff.gate.weight"], null, w[$"{p}.ff.up.weight"], null, w[$"{p}.ff.down.weight"], null);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_norm1 is not null) yield return _norm1;
        if (_norm2 is not null) yield return _norm2;
        foreach (Tensor t in _attn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _ffn.EnumerateWeights()) yield return t;
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
        Tensor ffOut = _ffn.Forward(backend, n2, batch, seqLen);
        n2.Dispose();
        Tensor outp = new Tensor(hShape, DType.F32);
        backend.Add(outp, h1, ffOut);
        ffOut.Dispose(); h1.Dispose();
        return outp;
    }
}
