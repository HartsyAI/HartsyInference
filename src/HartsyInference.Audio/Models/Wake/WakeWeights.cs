using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Wake;

/// <summary>Weight-loading helpers shared by the wake models.
///
/// <para>These models take <b>owned copies</b> of their weights rather than borrowing the loader's tensors.
/// A wake model is resident for the life of the process while its <c>OnnxWeightLoader</c> is transient, and a
/// loader disposes the tensors it handed out — so borrowing turns into a use-after-dispose the moment the
/// loading scope closes. The weights are ~3 MB in total, so the copy is not worth avoiding.</para></summary>
public static class WakeWeights
{
    /// <summary>Returns an owned F32 copy of <paramref name="source"/>, independent of the loader's lifetime.</summary>
    public static Tensor Own(Tensor source)
    {
        Tensor f32 = WhisperOps.EnsureF32(source);
        Tensor owned = new(f32.Shape, DType.F32);
        f32.AsSpan<float>().CopyTo(owned.AsSpan<float>());
        if (!ReferenceEquals(f32, source)) f32.Dispose();
        return owned;
    }

    /// <summary>Looks up <paramref name="name"/> and returns an owned copy, or throws naming the model that
    /// should have supplied it.</summary>
    public static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string name, string expectedSource) =>
        weights.TryGetValue(name, out Tensor? t) ? Own(t)
            : throw new HartsyInferenceException($"Missing weight '{name}'. Expected {expectedSource}.");
}
