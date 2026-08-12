using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Pre-norm neighborhood-attention + SwiGLU block of the LTX-2.5 video decoder, covering both variants: the
/// deterministic stage-1..4 <c>NABlock</c> (<c>x += attn(norm1(x))</c>, <c>x += mlp(norm2(x))</c>) and the stage-5
/// <c>DiffusionNABlock</c>, which additionally adds a projection of the stage-1..4 context and modulates each norm by
/// AdaLN scale/shift. Both residual adds are ungated in either variant — the AdaLN gate slots (2, 5, 6) are unused,
/// folded away at export.</summary>
internal sealed class LtxVideo25NaBlock
{
    private const int ModulationChunks = 7;
    private const int ScaleMsaChunk = 0, ShiftMsaChunk = 1, ScaleMlpChunk = 3, ShiftMlpChunk = 4;

    private readonly int _dim;
    private readonly int _hidden;
    private readonly float _eps;
    private readonly bool _isDiffusion;
    private readonly LtxVideo25NeighborhoodAttention3d _attention;

    private Tensor? _norm1Weight, _norm2Weight;
    private Tensor? _gateWeight, _upWeight, _downWeight;
    private Tensor? _contextProjWeight, _contextProjBias, _scaleShiftTable;

    public LtxVideo25NaBlock(int dim, (int T, int H, int W) kernel, LtxVideo25DiffusionDecoderConfig config, bool isDiffusion)
    {
        _dim = dim;
        _hidden = LtxVideo25DiffusionDecoderConfig.SwiGluHidden(dim);
        _eps = config.NormEps;
        _isDiffusion = isDiffusion;
        _attention = new LtxVideo25NeighborhoodAttention3d(dim, kernel, config);
    }

    public void LoadWeights(LtxVideo25WeightScope scope, string prefix)
    {
        _norm1Weight = scope.F32($"{prefix}.norm1.weight");
        _norm2Weight = scope.F32($"{prefix}.norm2.weight");
        _attention.LoadWeights(scope, $"{prefix}.attn");
        _gateWeight = scope.Raw($"{prefix}.mlp.w_gate.weight");
        _upWeight = scope.Raw($"{prefix}.mlp.w_up.weight");
        _downWeight = scope.Raw($"{prefix}.mlp.w_down.weight");
        if (_gateWeight.Shape[0] != _hidden || _gateWeight.Shape[1] != _dim)
            throw new InvalidOperationException($"'{prefix}.mlp.w_gate.weight' is {_gateWeight.Shape}, expected [{_hidden}, {_dim}].");
        if (!_isDiffusion) return;
        _contextProjWeight = scope.Raw($"{prefix}.context_proj.weight");
        _contextProjBias = scope.OptionalF32($"{prefix}.context_proj.bias");
        _scaleShiftTable = scope.F32($"{prefix}.scale_shift_table");
        if (_scaleShiftTable.ElementCount != (long)ModulationChunks * _dim)
            throw new InvalidOperationException($"'{prefix}.scale_shift_table' has {_scaleShiftTable.ElementCount} entries, expected {ModulationChunks * _dim}.");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _attention.EnumerateWeights()) yield return t;
        foreach (Tensor? t in new[] { _norm1Weight, _norm2Weight, _gateWeight, _upWeight, _downWeight,
                                      _contextProjWeight, _contextProjBias, _scaleShiftTable })
        {
            if (t is not null) yield return t;
        }
    }

    /// <summary>Updates <paramref name="x"/> <c>[t·h·w, dim]</c> in place. <paramref name="context"/> and
    /// <paramref name="modulation"/> (<c>[7, dim]</c> AdaLN chunks, before this block's own table is added) are
    /// required for a diffusion block and ignored by a deterministic one.</summary>
    public unsafe void Forward(IBackend backend, Tensor x, Tensor? context, Tensor? modulation, int t, int h, int w)
    {
        long tokens = (long)t * h * w;
        if (_isDiffusion)
        {
            if (context is null || modulation is null)
                throw new ArgumentNullException(nameof(context), "A diffusion block needs both the latent context and the AdaLN modulation.");
            using Tensor projected = new Tensor(new TensorShape(tokens, _dim), DType.F32);
            backend.Linear(projected, context, _contextProjWeight!, _contextProjBias);
            backend.Add(x, x, projected);
        }

        using (Tensor normed = new Tensor(new TensorShape(tokens, _dim), DType.F32))
        {
            backend.RmsNorm(normed, x, _norm1Weight!, _eps);
            if (_isDiffusion) Modulate(normed, modulation!, ScaleMsaChunk, ShiftMsaChunk, tokens);
            using Tensor attended = _attention.Forward(backend, normed, t, h, w);
            backend.Add(x, x, attended);
        }

        using Tensor normed2 = new Tensor(new TensorShape(tokens, _dim), DType.F32);
        backend.RmsNorm(normed2, x, _norm2Weight!, _eps);
        if (_isDiffusion) Modulate(normed2, modulation!, ScaleMlpChunk, ShiftMlpChunk, tokens);
        using Tensor mlp = Feedforward(backend, normed2, tokens);
        backend.Add(x, x, mlp);
    }

    private Tensor Feedforward(IBackend backend, Tensor x, long tokens)
    {
        using Tensor gate = new Tensor(new TensorShape(tokens, _hidden), DType.F32);
        backend.Linear(gate, x, _gateWeight!, null);
        backend.Silu(gate, gate);
        using (Tensor up = new Tensor(new TensorShape(tokens, _hidden), DType.F32))
        {
            backend.Linear(up, x, _upWeight!, null);
            backend.Mul(gate, gate, up);
        }
        Tensor result = new Tensor(new TensorShape(tokens, _dim), DType.F32);
        backend.Linear(result, gate, _downWeight!, null);
        return result;
    }

    /// <summary><c>x = x·(1 + scale) + shift</c> where scale/shift are the per-block sums of the shared AdaLN chunk
    /// and this block's <c>scale_shift_table</c> row, broadcast over every token.</summary>
    private unsafe void Modulate(Tensor x, Tensor modulation, int scaleChunk, int shiftChunk, long tokens)
    {
        float* rows = (float*)x.DataPointer;
        float* mod = (float*)modulation.DataPointer;
        float* table = (float*)_scaleShiftTable!.DataPointer;
        for (long token = 0; token < tokens; token++)
        {
            float* row = rows + token * _dim;
            for (int i = 0; i < _dim; i++)
            {
                float scale = mod[scaleChunk * _dim + i] + table[scaleChunk * _dim + i];
                float shift = mod[shiftChunk * _dim + i] + table[shiftChunk * _dim + i];
                row[i] = row[i] * (1f + scale) + shift;
            }
        }
    }
}
