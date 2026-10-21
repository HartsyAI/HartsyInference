using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>FLUX.3 block, used for the per-mode and the joint stacks: one no-affine LayerNorm under a single adaLN,
/// then attention (Q/K RMSNorm + RoPE) and a SwiGLU MLP in parallel off the same modulated input, summed and added
/// back through one gate: <c>x += gate·(attn_out(attn) + mlp_out(silu(gate_in)·up_in))</c>. No biases.
/// Weights use ComfyUI's names with any fused <c>qkv</c> split by <c>Flux3CheckpointConverter</c>.</summary>
public sealed class Flux3Block(int hiddenSize, int numHeads, int mlpHidden, float layerNormEps, float qkNormEps)
{
    private readonly int _hiddenSize = hiddenSize;
    private readonly int _numHeads = numHeads;
    private readonly int _headDim = hiddenSize / numHeads;
    private readonly int _mlpHidden = mlpHidden;
    private readonly float _layerNormEps = layerNormEps;
    private readonly QkNorm _normQ = new QkNorm(hiddenSize / numHeads, qkNormEps);
    private readonly QkNorm _normK = new QkNorm(hiddenSize / numHeads, qkNormEps);
    private Tensor? _qProj;
    private Tensor? _kProj;
    private Tensor? _vProj;
    private Tensor? _mlpIn;
    private Tensor? _attnOut;
    private Tensor? _mlpOut;
    private AttentionLayer? _attention;

    /// <summary>Binds this block's weights under <paramref name="prefix"/>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _qProj = weights[$"{prefix}.q_proj.weight"];
        _kProj = weights[$"{prefix}.k_proj.weight"];
        _vProj = weights[$"{prefix}.v_proj.weight"];
        _mlpIn = weights[$"{prefix}.mlp_in.weight"];
        _attnOut = weights[$"{prefix}.attn_out.weight"];
        _mlpOut = weights[$"{prefix}.mlp_out.weight"];
        _normQ.LoadWeights(weights[$"{prefix}.norm.query_norm.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.norm.key_norm.weight"]);
        _attention = new AttentionLayer(
            new AttentionLayerConfig
            {
                Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim, QkNorm = AttentionQkNorm.PerHead, AllowF16 = true,
            },
            new AttentionNormWeights(_normQ.Weight, _normK.Weight, _normQ.Eps, _normK.Eps));
    }

    /// <summary>Every weight this block reads, for preloading and freeing.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_qProj is not null) yield return _qProj;
        if (_kProj is not null) yield return _kProj;
        if (_vProj is not null) yield return _vProj;
        if (_mlpIn is not null) yield return _mlpIn;
        if (_attnOut is not null) yield return _attnOut;
        if (_mlpOut is not null) yield return _mlpOut;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
    }

    /// <summary>Runs the block on <paramref name="x"/> <c>[1, S, D]</c> in its own dtype (F32, or F16 on the damped stream)
    /// and returns a new tensor of that dtype; <paramref name="rope"/> must hold tables precomputed for exactly these
    /// <c>S</c> tokens.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Flux3Modulation modulation, FluxRope rope)
    {
        if (x.Shape.Rank != 3 || x.Shape[0] != 1 || x.Shape[2] != _hiddenSize)
            throw new ArgumentException($"Flux3Block expects [1, S, {_hiddenSize}], got {x.Shape}.", nameof(x));
        int seqLen = (int)x.Shape[1];
        DType act = x.DType;
        TensorShape hiddenShape = new TensorShape(1, seqLen, _hiddenSize);
        TensorShape headsShape = new TensorShape(1, seqLen, _numHeads, _headDim);

        Tensor modulated = modulation.NormModulate(backend, x, _layerNormEps);
        Tensor q = new Tensor(headsShape, act);
        backend.Linear(q, modulated, _qProj!, null);
        Tensor k = new Tensor(headsShape, act);
        backend.Linear(k, modulated, _kProj!, null);
        Tensor v = new Tensor(headsShape, act);
        backend.Linear(v, modulated, _vProj!, null);
        Tensor gateUp = new Tensor(new TensorShape(1, seqLen, 2 * _mlpHidden), act);
        backend.Linear(gateUp, modulated, _mlpIn!, null);
        modulated.Dispose();

        // Q/K RMSNorm bounds the logits, which keeps the fused F16 attention path safe.
        Tensor attn = _attention!.Attend(backend, q, k, v, hiddenShape, new AttentionCallContext { Rotation = rope });

        Tensor branch = new Tensor(hiddenShape, act);
        backend.Linear(branch, attn, _attnOut!, null);
        attn.Dispose();
        Tensor swiglu = new Tensor(new TensorShape(1, seqLen, _mlpHidden), act);
        backend.GluActivate(swiglu, gateUp, _mlpHidden, gelu: false);
        gateUp.Dispose();
        Tensor mlp = new Tensor(hiddenShape, act);
        backend.Linear(mlp, swiglu, _mlpOut!, null);
        swiglu.Dispose();
        Tensor summed = new Tensor(hiddenShape, act);
        backend.Add(summed, branch, mlp);
        branch.Dispose();
        mlp.Dispose();

        Tensor result = modulation.GatedResidual(backend, x, summed);
        summed.Dispose();
        return result;
    }
}