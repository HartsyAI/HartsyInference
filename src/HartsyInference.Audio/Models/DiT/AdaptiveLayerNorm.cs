using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.DiT;

/// <summary>"AdaLN" (not AdaLN-zero): <c>weight · RMSNorm(x) + bias</c>, where <c>weight</c>/<c>bias</c> come
/// from a per-step conditioning embedding via a single bare Linear (no SiLU before it — unlike the
/// SiLU-gated AdaLN-zero convention <c>HartsyInference.Diffusion</c>'s own <c>AdaLNModulation</c> implements
/// for SD3). Ported here rather than referenced across the Audio/Diffusion package boundary (Audio has no
/// dependency on Diffusion) because the math genuinely differs, not just the namespace. Matches IndexTTS-2's
/// real S2Mel DiT (<c>indextts/s2mel/modules/gpt_fast/model.py:AdaptiveLayerNorm</c>), used for every
/// transformer block's <c>attention_norm</c>/<c>ffn_norm</c> and the stack's own final norm:
/// <code>
/// class AdaptiveLayerNorm(nn.Module):
///     def __init__(self, d_model, norm):           # norm: RMSNorm(d_model), its own trained gamma
///         self.project_layer = nn.Linear(d_model, 2 * d_model)
///     def forward(self, input, embedding):
///         weight, bias = self.project_layer(embedding).chunk(2, dim=-1)
///         return weight * self.norm(input) + bias
/// </code>
/// Built generic (constructor takes only <c>hiddenSize</c>) so any future "weight·RMSNorm(x)+bias" AdaLN
/// architecture can reuse it without an IndexTTS-2-specific name in the way.</summary>
public sealed unsafe class AdaptiveLayerNorm(int hiddenSize)
{
    private readonly int _hidden = hiddenSize;
    private Tensor? _normWeight;              // the RMSNorm's own trained gamma, [hidden]
    private Tensor? _projWeight, _projBias;   // project_layer: [2*hidden, hidden] / [2*hidden]

    /// <param name="normWeight">The wrapped RMSNorm's own trained gamma, <c>[hidden]</c>.</param>
    /// <param name="projWeight">The AdaLN projection's weight, <c>[2*hidden, hidden]</c>.</param>
    /// <param name="projBias">The AdaLN projection's bias, <c>[2*hidden]</c>.</param>
    public void LoadWeights(Tensor normWeight, Tensor projWeight, Tensor projBias)
    {
        _normWeight = normWeight;
        _projWeight = projWeight;
        _projBias = projBias;
    }

    /// <summary><paramref name="input"/> is <c>[1, T, hidden]</c>; <paramref name="embedding"/> is the shared
    /// per-step conditioning vector, <c>[1, hidden]</c> (every block receives the SAME embedding — IndexTTS-2's
    /// S2Mel DiT conditions every block on one timestep embedding, not a per-block one). Returns a new
    /// <c>[1, T, hidden]</c> tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor input, Tensor embedding, float eps = 1e-5f)
    {
        int t = (int)input.Shape[1];
        Tensor normed = new(input.Shape, DType.F32);
        backend.RmsNorm(normed, input, _normWeight!, eps);

        Tensor modulation = new(new TensorShape(1, 2 * _hidden), DType.F32);
        backend.Linear(modulation, embedding, _projWeight!, _projBias);

        Tensor output = new(input.Shape, DType.F32);
        float* np_ = (float*)normed.DataPointer;
        float* mp = (float*)modulation.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int ti = 0; ti < t; ti++)
        {
            float* row = np_ + (long)ti * _hidden;
            float* outRow = op + (long)ti * _hidden;
            for (int c = 0; c < _hidden; c++)
                outRow[c] = mp[c] * row[c] + mp[_hidden + c];
        }
        normed.Dispose();
        modulation.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_normWeight is not null) yield return _normWeight;
        if (_projWeight is not null) yield return _projWeight;
        if (_projBias is not null) yield return _projBias;
    }
}
