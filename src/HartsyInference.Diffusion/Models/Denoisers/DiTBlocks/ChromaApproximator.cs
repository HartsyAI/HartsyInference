using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Chroma's distilled-guidance approximator: a 5-layer SiLU + RMSNorm-residual MLP that produces
/// the per-block modulation table. Input <c>[B, mod_index_length, in_dim=64]</c> →
/// output <c>[B, mod_index_length, out_dim=3072]</c>.
///
/// Layer structure (per <c>transformer_chroma.py:184-200</c>):
/// <list type="bullet">
///   <item><c>in_proj</c>: <c>Linear(in_dim, hidden_dim, bias=True)</c></item>
///   <item>5x: <c>x = x + linear_2(silu(linear_1(rmsnorm(x))))</c> where <c>linear_1, linear_2</c> are
///         <see cref="PixArtAlphaTextProjection"/> internals (<c>hidden→hidden, bias=True</c>) and
///         <c>rmsnorm</c> has affine scale weight only.</item>
///   <item><c>out_proj</c>: <c>Linear(hidden_dim, out_dim, bias=True)</c></item>
/// </list></summary>
public sealed unsafe class ChromaApproximator : IDisposable
{
    private readonly int _inDim;
    private readonly int _hiddenDim;
    private readonly int _outDim;
    private readonly int _numLayers;

    // in_proj
    private Tensor? _inProjWeight;
    private Tensor? _inProjBias;

    // PixArtAlphaTextProjection layers: [linear_1, linear_2] x num_layers
    private readonly Tensor?[] _linear1Weight;
    private readonly Tensor?[] _linear1Bias;
    private readonly Tensor?[] _linear2Weight;
    private readonly Tensor?[] _linear2Bias;

    // RMSNorm scale (per layer)
    private readonly Tensor?[] _normWeight;

    // out_proj
    private Tensor? _outProjWeight;
    private Tensor? _outProjBias;

    private int _disposed;

    /// <summary>Creates an approximator with the given dimensions. Default Chroma v1: <c>(64, 3072, 5120, 5)</c>.</summary>
    public ChromaApproximator(int inDim, int outDim, int hiddenDim, int numLayers)
    {
        _inDim = inDim;
        _outDim = outDim;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;

        _linear1Weight = new Tensor?[numLayers];
        _linear1Bias = new Tensor?[numLayers];
        _linear2Weight = new Tensor?[numLayers];
        _linear2Bias = new Tensor?[numLayers];
        _normWeight = new Tensor?[numLayers];
    }

    /// <summary>Loads weights from named tensors using diffusers naming (post-conversion). Expected keys:
    /// <c>distilled_guidance_layer.in_proj.{weight,bias}</c>, <c>distilled_guidance_layer.out_proj.{weight,bias}</c>,
    /// <c>distilled_guidance_layer.layers.{i}.linear_{1,2}.{weight,bias}</c>,
    /// <c>distilled_guidance_layer.norms.{i}.weight</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        const string Prefix = "distilled_guidance_layer";

        _inProjWeight = weights[$"{Prefix}.in_proj.weight"];
        _inProjBias = weights[$"{Prefix}.in_proj.bias"];
        _outProjWeight = weights[$"{Prefix}.out_proj.weight"];
        _outProjBias = weights[$"{Prefix}.out_proj.bias"];

        for (int i = 0; i < _numLayers; i++)
        {
            _linear1Weight[i] = weights[$"{Prefix}.layers.{i}.linear_1.weight"];
            _linear1Bias[i] = weights[$"{Prefix}.layers.{i}.linear_1.bias"];
            _linear2Weight[i] = weights[$"{Prefix}.layers.{i}.linear_2.weight"];
            _linear2Bias[i] = weights[$"{Prefix}.layers.{i}.linear_2.bias"];

            // RMSNorm scale is read via raw float* in Forward; cast to F32 if it isn't already.
            // (PHASE_3_DEVIATIONS #30: norms touched via raw pointer must be F32.)
            Tensor norm = weights[$"{Prefix}.norms.{i}.weight"];
            _normWeight[i] = norm.DType != DType.F32 ? norm.CastTo(DType.F32) : norm;
        }
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_inProjWeight is not null) yield return _inProjWeight;
        if (_inProjBias is not null) yield return _inProjBias;
        for (int i = 0; i < _numLayers; i++)
        {
            if (_linear1Weight[i] is not null) yield return _linear1Weight[i]!;
            if (_linear1Bias[i] is not null) yield return _linear1Bias[i]!;
            if (_linear2Weight[i] is not null) yield return _linear2Weight[i]!;
            if (_linear2Bias[i] is not null) yield return _linear2Bias[i]!;
            if (_normWeight[i] is not null) yield return _normWeight[i]!;
        }
        if (_outProjWeight is not null) yield return _outProjWeight;
        if (_outProjBias is not null) yield return _outProjBias;
    }

    /// <summary>Forward pass. Input: <c>[B, mod_index_length, in_dim]</c> → Output: <c>[B, mod_index_length, out_dim]</c>.
    /// The internal hidden tensor stays at <c>[B, mod_index_length, hidden_dim]</c> across the residual stack.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int seqLen = (int)input.Shape[1];

        // ── 1. in_proj: [B, S, in_dim] → [B, S, hidden_dim] ──
        TensorShape hiddenShape = new TensorShape(batch, seqLen, _hiddenDim);
        Tensor hidden = new Tensor(hiddenShape, DType.F32);
        backend.Linear(hidden, input, _inProjWeight!, _inProjBias);

        // ── 2. Residual stack (5x): hidden = hidden + linear_2(silu(linear_1(rmsnorm(hidden)))) ──
        for (int i = 0; i < _numLayers; i++)
        {
            // Device RMSNorm (affine scale, last dim): the old host RmsNormAffine read the device-produced
            // hidden's DataPointer — a full pipeline drain per layer (5×/table build), and a capture-illegal
            // op inside the Chroma step graph. Identical formula (mean-of-squares + eps, F32 weight).
            Tensor normed = new Tensor(hiddenShape, DType.F32);
            backend.RmsNorm(normed, hidden, _normWeight[i]!, 1e-6f);

            Tensor lin1 = new Tensor(hiddenShape, DType.F32);
            backend.Linear(lin1, normed, _linear1Weight[i]!, _linear1Bias[i]);
            normed.Dispose();

            Tensor activated = new Tensor(hiddenShape, DType.F32);
            backend.Silu(activated, lin1);
            lin1.Dispose();

            Tensor lin2 = new Tensor(hiddenShape, DType.F32);
            backend.Linear(lin2, activated, _linear2Weight[i]!, _linear2Bias[i]);
            activated.Dispose();

            Tensor next = new Tensor(hiddenShape, DType.F32);
            backend.Add(next, hidden, lin2);
            lin2.Dispose();
            hidden.Dispose();
            hidden = next;
        }

        // ── 3. out_proj: [B, S, hidden_dim] → [B, S, out_dim] ──
        TensorShape outShape = new TensorShape(batch, seqLen, _outDim);
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Linear(output, hidden, _outProjWeight!, _outProjBias);
        hidden.Dispose();

        return output;
    }

    /// <summary>Releases tensor references.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _inProjWeight = null;
            _inProjBias = null;
            for (int i = 0; i < _numLayers; i++)
            {
                _linear1Weight[i] = null;
                _linear1Bias[i] = null;
                _linear2Weight[i] = null;
                _linear2Bias[i] = null;
                _normWeight[i] = null;
            }
            _outProjWeight = null;
            _outProjBias = null;
        }
        GC.SuppressFinalize(this);
    }
}
