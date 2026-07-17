using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Dia;

/// <summary>Dia SwiGLU MLP with a <b>fused</b> gate+up projection (<c>gate_up_proj</c> → 2·ffn, split into
/// gate|up, <c>silu(gate)*up</c>, then <c>down_proj</c>). No biases. When <paramref name="siluOnFirstHalf"/> is
/// <c>false</c> the halves are <c>up|gate</c> and the activation is <c>up·silu(gate)</c> — Zonos's
/// <c>FeedForward</c> (<c>y, gate = fc1.chunk(2); fc2(y·silu(gate))</c>).</summary>
public sealed unsafe class DiaMlp
{
    private readonly int _dim, _ffn;
    private readonly bool _siluOnFirstHalf;
    private Tensor? _gateUpW, _downW;

    public DiaMlp(int dim, int ffn, bool siluOnFirstHalf = true)
    {
        _dim = dim; _ffn = ffn; _siluOnFirstHalf = siluOnFirstHalf;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _gateUpW = WhisperOps.EnsureF32(w[$"{prefix}.gate_up_proj.weight"]);
        _downW = WhisperOps.EnsureF32(w[$"{prefix}.down_proj.weight"]);
    }

    public Tensor Forward(IBackend backend, Tensor x, int t)
    {
        Tensor gu = WhisperOps.ProjectLinear(backend, x, _gateUpW!, null, 1, t, _dim, 2 * _ffn);
        Tensor act = new(new TensorShape(1, t, _ffn), DType.F32);
        float* gup = (float*)gu.DataPointer;
        float* ap = (float*)act.DataPointer;
        for (int s = 0; s < t; s++)
        {
            long gbase = (long)s * 2 * _ffn;
            long abase = (long)s * _ffn;
            for (int i = 0; i < _ffn; i++)
            {
                // first|second halves; silu is applied to the gate half (first when _siluOnFirstHalf).
                float first = gup[gbase + i];
                float second = gup[gbase + _ffn + i];
                float gate = _siluOnFirstHalf ? first : second;
                float up = _siluOnFirstHalf ? second : first;
                float silu = gate / (1f + MathF.Exp(-gate));
                ap[abase + i] = silu * up;
            }
        }
        gu.Dispose();
        Tensor outT = WhisperOps.ProjectLinear(backend, act, _downW!, null, 1, t, _ffn, _dim);
        act.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_gateUpW is not null) yield return _gateUpW;
        if (_downW is not null) yield return _downW;
    }
}
