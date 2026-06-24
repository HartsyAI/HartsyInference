using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Layers;

/// <summary>Helpers for reconstructing PyTorch <c>weight_norm</c>-decoupled weights at
/// load time. <c>torch.nn.utils.weight_norm</c> reparametrizes a weight tensor <c>W</c>
/// (shape <c>[out, ...]</c>) into a magnitude <c>weight_g</c> (shape <c>[out, 1, ...]</c>
/// with all dims-after-0 of size 1) and a direction <c>weight_v</c> (same shape as the
/// original <c>W</c>), where the effective forward-pass weight is:
///
/// <code>
///   W = weight_v * (weight_g / ||weight_v||_dim0)
/// </code>
///
/// with the norm taken over every axis except <c>dim 0</c> (the output-channel axis).
///
/// <para>Used by Kokoro / StyleTTS 2 / iSTFTNet — almost every Conv1D in the prosody
/// predictor and decoder is stored as a <c>(weight_g, weight_v)</c> pair. We compose
/// them into the dense weight tensor once at <c>LoadWeights</c> time so the forward
/// pass uses the normal <see cref="Core.Backends.IBackend.Conv1d"/> path.</para></summary>
internal static unsafe class WeightNorm
{
    /// <summary>Composes <c>weight_g</c> and <c>weight_v</c> into a dense weight tensor.
    /// Both inputs are F32; the returned tensor is F32 with shape equal to
    /// <paramref name="weightV"/>.
    ///
    /// <para><paramref name="weightG"/> may be shape <c>[out, 1, 1]</c> (Conv1D),
    /// <c>[out, 1, 1, 1]</c> (Conv2D), or a flat <c>[out]</c> vector — any of these is
    /// accepted as long as the element count equals the output-channel count.</para></summary>
    public static Tensor Compose(Tensor weightG, Tensor weightV)
    {
        Tensor vF = WhisperOps.EnsureF32(weightV);
        Tensor gF = WhisperOps.EnsureF32(weightG);
        int outChannels = (int)vF.Shape[0];
        if ((int)gF.ElementCount != outChannels)
            throw new ArgumentException($"WeightNorm: weight_g has {gF.ElementCount} elements but weight_v has {outChannels} output channels.");

        long perOut = vF.ElementCount / outChannels;
        Tensor result = new(vF.Shape, DType.F32);

        float* gp = (float*)gF.DataPointer;
        float* vp = (float*)vF.DataPointer;
        float* op = (float*)result.DataPointer;

        for (int oc = 0; oc < outChannels; oc++)
        {
            long baseIdx = oc * perOut;
            // L2 norm of v along the trailing axes for this output channel.
            double sumSq = 0d;
            for (long k = 0; k < perOut; k++)
            {
                float val = vp[baseIdx + k];
                sumSq += (double)val * val;
            }
            float norm = (float)Math.Sqrt(sumSq);
            // Avoid div-by-zero. weight_norm explicitly does not regularize so a true
            // zero direction would already be a degenerate model — guard rather than
            // mask the issue.
            float invNorm = norm > 1e-12f ? 1.0f / norm : 0f;
            float scale = gp[oc] * invNorm;
            for (long k = 0; k < perOut; k++) op[baseIdx + k] = vp[baseIdx + k] * scale;
        }

        if (!ReferenceEquals(vF, weightV)) vF.Dispose();
        if (!ReferenceEquals(gF, weightG)) gF.Dispose();
        return result;
    }

    /// <summary>Convenience: pulls the magnitude/direction pair from <paramref name="w"/> and composes them.
    /// Accepts both serialization formats: the legacy <c>torch.nn.utils.weight_norm</c> names
    /// (<c>{prefix}.weight_g</c> / <c>{prefix}.weight_v</c>) and the newer (PyTorch &gt;= 2.1)
    /// <c>torch.nn.utils.parametrizations.weight_norm</c> names
    /// (<c>{prefix}.parametrizations.weight.original0</c> = g / <c>.original1</c> = v). Throws if neither
    /// pair is present.</summary>
    public static Tensor Compose(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        // Legacy format.
        if (w.TryGetValue($"{prefix}.weight_g", out Tensor? g) && w.TryGetValue($"{prefix}.weight_v", out Tensor? v))
            return Compose(g, v);
        // New parametrizations format (original0 = g/magnitude, original1 = v/direction).
        if (w.TryGetValue($"{prefix}.parametrizations.weight.original0", out Tensor? g2)
            && w.TryGetValue($"{prefix}.parametrizations.weight.original1", out Tensor? v2))
            return Compose(g2, v2);
        throw new KeyNotFoundException(
            $"WeightNorm.Compose: missing weight-norm pair for '{prefix}' (looked for '.weight_g'/'.weight_v' and '.parametrizations.weight.original0'/'.original1').");
    }
}
