using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Dia;

/// <summary>Dia SwiGLU MLP with a <b>fused</b> gate+up projection (<c>gate_up_proj</c> → 2·ffn, split into
/// gate|up, <c>silu(gate)*up</c>, then <c>down_proj</c>). No biases.</summary>
public sealed unsafe class DiaMlp
{
    private readonly int _dim, _ffn;
    private Tensor? _gateUpW, _downW;

    public DiaMlp(int dim, int ffn) { _dim = dim; _ffn = ffn; }

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
                float g = gup[gbase + i];
                float u = gup[gbase + _ffn + i];
                float silu = g / (1f + MathF.Exp(-g));
                ap[abase + i] = silu * u;
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
