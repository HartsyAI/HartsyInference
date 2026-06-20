using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Zonos;

/// <summary>Zonos <c>FourierConditioner</c> — Gaussian random-feature embedding of a scalar/vector control.
/// Normalizes the input to <c>[0,1]</c> over <c>[min,max]</c>, projects through a fixed random weight buffer,
/// and emits <c>[cos(2π·f), sin(2π·f)]</c> as the <see cref="OutputDim"/>-d conditioning token. A learned
/// <c>uncond_vector</c> replaces the output when the control is unconditional.</summary>
public sealed unsafe class ZonosFourierConditioner
{
    private readonly int _inputDim;
    private readonly float _min, _max;
    private Tensor? _weight;        // [OutputDim/2, inputDim]
    private Tensor? _uncond;        // [OutputDim]

    public int OutputDim { get; }

    public ZonosFourierConditioner(int inputDim, int outputDim, float min, float max)
    {
        _inputDim = inputDim; OutputDim = outputDim; _min = min; _max = max;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _weight = WhisperOps.EnsureF32(w[$"{prefix}.weight"]);
        _uncond = w.TryGetValue($"{prefix}.uncond_vector", out Tensor? u) ? WhisperOps.EnsureF32(u) : null;
    }

    /// <summary>Embeds <paramref name="value"/> (length <see cref="_inputDim"/>) → <c>[OutputDim]</c>.</summary>
    public float[] Forward(ReadOnlySpan<float> value)
    {
        int half = OutputDim / 2;
        float[] outArr = new float[OutputDim];
        float* wp = (float*)_weight!.DataPointer;
        float range = _max - _min;
        for (int j = 0; j < half; j++)
        {
            double f = 0;
            for (int d = 0; d < _inputDim; d++)
            {
                float xNorm = range != 0 ? (value[d] - _min) / range : value[d];
                f += xNorm * wp[(long)j * _inputDim + d];
            }
            f *= 2 * Math.PI;
            outArr[j] = (float)Math.Cos(f);
            outArr[half + j] = (float)Math.Sin(f);
        }
        return outArr;
    }

    /// <summary>The learned unconditional vector (or zeros if absent).</summary>
    public float[] Uncond()
    {
        float[] outArr = new float[OutputDim];
        if (_uncond is not null)
            new Span<float>((void*)_uncond.DataPointer, OutputDim).CopyTo(outArr);
        return outArr;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_weight is not null) yield return _weight;
        if (_uncond is not null) yield return _uncond;
    }
}
