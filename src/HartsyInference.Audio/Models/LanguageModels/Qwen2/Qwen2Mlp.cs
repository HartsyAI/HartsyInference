using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.LanguageModels.Qwen2;

/// <summary>Qwen2's SwiGLU MLP. <c>down(silu(gate(x)) * up(x))</c> with no bias on any of
/// the three Linear projections. Mirrors HF <c>Qwen2MLP</c>.</summary>
internal sealed unsafe class Qwen2Mlp
{
    private readonly Qwen2Config _cfg;
    private Tensor? _gateW, _upW, _downW;

    public Qwen2Mlp(Qwen2Config cfg) { _cfg = cfg; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _gateW = WhisperOps.EnsureF32(w[$"{prefix}.gate_proj.weight"]);
        _upW = WhisperOps.EnsureF32(w[$"{prefix}.up_proj.weight"]);
        _downW = WhisperOps.EnsureF32(w[$"{prefix}.down_proj.weight"]);
    }

    public Tensor Forward(IBackend backend, Tensor input, int batch, int t)
    {
        int h = _cfg.HiddenSize;
        int inter = _cfg.IntermediateSize;
        Tensor gate = WhisperOps.ProjectLinear(backend, input, _gateW!, bias: null, batch, t, h, inter);
        Tensor up = WhisperOps.ProjectLinear(backend, input, _upW!, bias: null, batch, t, h, inter);
        backend.Silu(gate, gate);     // in-place
        Tensor gated = new(gate.Shape, DType.F32);
        backend.Mul(gated, gate, up);
        gate.Dispose();
        up.Dispose();
        Tensor output = WhisperOps.ProjectLinear(backend, gated, _downW!, bias: null, batch, t, inter, h);
        gated.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_gateW is not null) yield return _gateW;
        if (_upW is not null) yield return _upW;
        if (_downW is not null) yield return _downW;
    }
}
