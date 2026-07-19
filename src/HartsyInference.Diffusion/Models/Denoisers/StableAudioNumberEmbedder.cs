using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Stable Audio's timing conditioner (<c>stable_audio_tools.models.conditioners.NumberConditioner</c> +
/// <c>adp.NumberEmbedder</c>/<c>TimePositionalEmbedding</c>/<c>LearnedPositionalEmbedding</c>): encodes a scalar
/// (<c>seconds_total</c>, and <c>seconds_start</c> for Open 1.0) into a <see cref="StableAudioDitConfig.CondTokenDim"/>-
/// wide Fourier token — <c>StableAudioDit.Forward</c>'s raw (pre-<c>to_cond_embed</c>/<c>to_global_embed</c>)
/// cross-attention/global conditioning input. Math: <c>norm = clamp(value, min, max) / max</c> (min is always 0
/// on the released checkpoints) → <c>fourier = [norm, sin(norm·w·2π), cos(norm·w·2π)]</c> (257-wide: the scalar
/// itself prepended to 128 sin ‖ 128 cos features) → <c>Linear(257 → CondTokenDim)</c>. Checkpoint keys:
/// <c>conditioners.{name}.embedder.embedding.0.weights</c> [128], <c>conditioners.{name}.embedder.embedding.1.
/// {weight,bias}</c> [768,257]/[768].</summary>
public sealed unsafe class StableAudioNumberEmbedder
{
    private readonly float _minVal, _maxVal;
    private Tensor? _fourierWeights, _linearWeight, _linearBias;

    /// <param name="minVal">Clamp/normalize range minimum (0 on both released variants).</param>
    /// <param name="maxVal">Clamp/normalize range maximum — <see cref="StableAudioDitConfig.TimingMaxSeconds"/>.</param>
    public StableAudioNumberEmbedder(float minVal, float maxVal)
    {
        _minVal = minVal;
        _maxVal = maxVal;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _fourierWeights = PickF32(w, $"{prefix}.embedder.embedding.0.weights");
        _linearWeight = Pick(w, $"{prefix}.embedder.embedding.1.weight");
        _linearBias = PickOpt(w, $"{prefix}.embedder.embedding.1.bias");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _fourierWeights, _linearWeight, _linearBias })
            if (t is not null) yield return t;
    }

    /// <summary>Encodes one scalar into a <c>[1, 1, CondTokenDim]</c> token.</summary>
    public Tensor Embed(IBackend backend, float value)
    {
        int half = (int)_fourierWeights!.Shape[0];
        float norm = (Math.Clamp(value, _minVal, _maxVal) - _minVal) / (_maxVal - _minVal);

        Tensor fourier = new Tensor(new TensorShape(1, 1, 2 * half + 1), DType.F32);
        float* fp = (float*)fourier.DataPointer;
        float* wp = (float*)_fourierWeights!.DataPointer;
        fp[0] = norm;
        for (int i = 0; i < half; i++)
        {
            float freq = norm * wp[i] * 2f * MathF.PI;
            fp[1 + i] = MathF.Sin(freq);
            fp[1 + half + i] = MathF.Cos(freq);
        }

        int outDim = (int)_linearWeight!.Shape[0];
        Tensor embed = new Tensor(new TensorShape(1, 1, outDim), DType.F32);
        backend.Linear(embed, fourier, _linearWeight!, _linearBias);
        fourier.Dispose();
        return embed;
    }

    private static Tensor Pick(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? t : throw new KeyNotFoundException($"'{key}' not found in checkpoint.");

    private static Tensor? PickOpt(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? t : null;

    private static Tensor PickF32(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        Tensor t = Pick(w, key);
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
