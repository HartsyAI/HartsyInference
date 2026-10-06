using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.LengthRegulation;

/// <summary>Generic explicit-target-length regulator: projects continuous content features to a working
/// channel width, nearest-neighbor-resamples along time to an explicit target length, then refines with N
/// stages of (Conv1d k3 s1 p1 → GroupNorm → Mish) followed by a final 1×1 Conv1d. Distinct from
/// <see cref="HartsyInference.Audio.Models.Vits.VitsLengthRegulator"/>'s duration-predictor shape — this one
/// takes the target length directly rather than predicting it, matching IndexTTS-2's real
/// <c>indextts/s2mel/modules/length_regulator.py:InterpolateRegulator</c> (the <c>is_discrete=False</c>
/// branch; the discrete-codebook branch is a dead module for IndexTTS-2's real checkpoint — the real
/// <c>infer_v2_5.py</c> only ever calls it with continuous semantic-codec features). Built generic (no
/// IndexTTS-2-specific naming) so a future FastSpeech/StableTTS-style model can reuse it.
/// <code>
/// class InterpolateRegulator(nn.Module):
///     def forward(self, x, ylens):
///         x = self.content_in_proj(x)                                    # Linear(in_channels, channels)
///         x = F.interpolate(x.transpose(1,2), size=ylens.max(), mode='nearest')
///         return self.model(x).transpose(1, 2)                           # N×(Conv1d k3→GroupNorm→Mish), then Conv1d k1
/// </code></summary>
public sealed unsafe class InterpolateLengthRegulator
{
    private readonly int _channels, _inChannels, _numStages, _groups;
    private Tensor? _inProjW, _inProjB;
    private readonly Tensor?[] _convW, _convB, _normW, _normB;
    private Tensor? _outW, _outB;

    public InterpolateLengthRegulator(int channels, int inChannels, int numStages, int groups = 1)
    {
        _channels = channels;
        _inChannels = inChannels;
        _numStages = numStages;
        _groups = groups;
        _convW = new Tensor?[numStages];
        _convB = new Tensor?[numStages];
        _normW = new Tensor?[numStages];
        _normB = new Tensor?[numStages];
    }

    /// <param name="prefix">e.g. <c>net.length_regulator</c> — the real checkpoint's own module name, no
    /// further nesting.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _inProjW = WhisperOps.EnsureF32(w[$"{prefix}.content_in_proj.weight"]);
        _inProjB = WhisperOps.EnsureF32(w[$"{prefix}.content_in_proj.bias"]);
        for (int i = 0; i < _numStages; i++)
        {
            int convIdx = i * 3, normIdx = i * 3 + 1;   // index 2 of each triple is the parameter-free Mish.
            _convW[i] = WhisperOps.EnsureF32(w[$"{prefix}.model.{convIdx}.weight"]);
            _convB[i] = WhisperOps.EnsureF32(w[$"{prefix}.model.{convIdx}.bias"]);
            _normW[i] = WhisperOps.EnsureF32(w[$"{prefix}.model.{normIdx}.weight"]);
            _normB[i] = WhisperOps.EnsureF32(w[$"{prefix}.model.{normIdx}.bias"]);
        }
        int outIdx = _numStages * 3;
        _outW = WhisperOps.EnsureF32(w[$"{prefix}.model.{outIdx}.weight"]);
        _outB = WhisperOps.EnsureF32(w[$"{prefix}.model.{outIdx}.bias"]);
    }

    /// <summary><paramref name="content"/> is <c>[1, sourceT, inChannels]</c> (continuous, channel-last).
    /// Returns <c>[1, targetT, channels]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor content, int sourceT, int targetT)
    {
        Tensor projected = WhisperOps.ProjectLinear(backend, content, _inProjW!, _inProjB, 1, sourceT, _inChannels, _channels);
        Tensor channelFirst = TransposeLastTwo(projected, sourceT, _channels);
        projected.Dispose();

        Tensor cur = NearestInterpolate(channelFirst, sourceT, targetT, _channels);
        channelFirst.Dispose();

        for (int i = 0; i < _numStages; i++)
        {
            Tensor conv = new(new TensorShape(1, _channels, targetT), DType.F32);
            backend.Conv1d(conv, cur, _convW[i]!, _convB[i], 1, 1, 1, 1, 1);
            cur.Dispose();

            Tensor normed = new(conv.Shape, DType.F32);
            backend.GroupNorm(normed, conv, _normW[i]!, _normB[i]!, _groups, 1e-5f);
            conv.Dispose();

            Tensor act = new(normed.Shape, DType.F32);
            backend.Mish(act, normed);
            normed.Dispose();
            cur = act;
        }

        Tensor outConv = new(new TensorShape(1, _channels, targetT), DType.F32);
        backend.Conv1d(outConv, cur, _outW!, _outB, 1, 0, 0, 1, 1);
        cur.Dispose();

        Tensor result = TransposeLastTwo(outConv, _channels, targetT);
        outConv.Dispose();
        return result;
    }

    /// <summary>[1, d1, d2] → [1, d2, d1].</summary>
    private static Tensor TransposeLastTwo(Tensor t, int d1, int d2)
    {
        Tensor result = new(new TensorShape(1, d2, d1), DType.F32);
        float* sp = (float*)t.DataPointer;
        float* dp = (float*)result.DataPointer;
        for (int a = 0; a < d1; a++)
            for (int b = 0; b < d2; b++)
                dp[(long)b * d1 + a] = sp[(long)a * d2 + b];
        return result;
    }

    /// <summary>Real <c>F.interpolate(mode='nearest')</c>'s index mapping on a channel-first <c>[1, C, sourceT]</c> input.</summary>
    private static Tensor NearestInterpolate(Tensor x, int sourceT, int targetT, int channels)
    {
        Tensor result = new(new TensorShape(1, channels, targetT), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* dp = (float*)result.DataPointer;
        for (int j = 0; j < targetT; j++)
        {
            int srcIdx = (int)((long)j * sourceT / targetT);
            if (srcIdx >= sourceT) srcIdx = sourceT - 1;
            for (int c = 0; c < channels; c++)
                dp[(long)c * targetT + j] = sp[(long)c * sourceT + srcIdx];
        }
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_inProjW, _inProjB, _outW, _outB];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        foreach (Tensor?[] g in new[] { _convW, _convB, _normW, _normB })
            foreach (Tensor? t in g) if (t is not null) yield return t;
    }
}
