using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Mert2;

/// <summary>Pieces of MERT-v2 that no <see cref="IBackend"/> op covers directly: the checkpoint's BF16-to-F32
/// load with explicit ownership, ConvNeXt-V2's GlobalResponseNorm, and the split-half RoPE tables.</summary>
internal static unsafe class Mert2Ops
{
    /// <summary>Fetches a weight as F32. A converted tensor is appended to <paramref name="owned"/> because the
    /// model must free it; an already-F32 tensor still belongs to the checkpoint loader and is not added.</summary>
    public static Tensor Load(IReadOnlyDictionary<string, Tensor> weights, string key, List<Tensor> owned)
    {
        if (!weights.TryGetValue(key, out Tensor? source))
            throw new HartsyInferenceException($"MERT2 checkpoint is missing '{key}'.");
        if (source.DType == DType.F32)
        {
            return source;
        }
        Tensor converted = source.CastTo(DType.F32);
        owned.Add(converted);
        return converted;
    }

    /// <summary>ConvNeXt-V2 GlobalResponseNorm over <c>x [frames, channels]</c>:
    /// <c>out = weight·(x·n) + bias + x</c>, where <c>n</c> is each channel's L2 norm across the WHOLE time axis
    /// divided by the mean of those norms. Folding the residual into the broadcast makes it one pass:
    /// <c>out = x·(weight·n + 1) + bias</c>.
    ///
    /// <para>The reduction is expressed as a <c>[1, frames] × [frames, channels]</c> GEMM so it runs wherever the
    /// activations already live instead of dragging them back to the host.</para></summary>
    /// <param name="weight">Host copy of the learned gain; it is folded into a scalar per channel here rather than
    /// handed to a backend op, so the model never reads a possibly device-resident weight's host pointer.</param>
    /// <param name="ones">A <c>[1, frames]</c> tensor of ones, reused across calls.</param>
    /// <param name="squared">Scratch shaped like <paramref name="x"/>.</param>
    public static void GlobalResponseNorm(IBackend backend, Tensor output, Tensor x, ReadOnlySpan<float> weight,
        Tensor bias, Tensor ones, Tensor squared)
    {
        int channels = (int)x.Shape[x.Shape.Rank - 1];
        backend.Mul(squared, x, x);
        // Freshly allocated rather than pooled: the scale below is written on the host, and a backend that caches a
        // device copy per tensor object would serve the previous layer's values for a reused buffer.
        using Tensor sums = new(new TensorShape(1, channels), DType.F32);
        backend.MatMul(sums, ones, squared);

        float* sumValues = (float*)sums.DataPointer;
        using Tensor scale = new(new TensorShape(1, channels), DType.F32);
        float* scaleValues = (float*)scale.DataPointer;
        float mean = 0f;
        for (int c = 0; c < channels; c++)
        {
            scaleValues[c] = MathF.Sqrt(sumValues[c]);
            mean += scaleValues[c];
        }
        float divisor = mean / channels + 1e-6f;
        for (int c = 0; c < channels; c++)
        {
            scaleValues[c] = weight[c] * (scaleValues[c] / divisor) + 1f;
        }
        backend.AffineBroadcastLastDim(output, x, scale, bias);
    }

    /// <summary>Fills the split-half RoPE tables <c>[1, tokens, headDim]</c>, each half holding the same
    /// <c>headDim/2</c> values so <see cref="IBackend.ApplyRope"/>'s rotate-half indexing lines up.
    ///
    /// <para>The angle is accumulated in single precision on purpose. The released model builds it as a float32
    /// product of position and inverse frequency, and by token 7499 an exact angle has drifted a few times 1e-4
    /// radians from that rounded one — enough to show up in the attention scores.</para></summary>
    public static void BuildRopeTables(Tensor cos, Tensor sin, int tokens, int headDim, float theta)
    {
        int half = headDim / 2;
        float* cosValues = (float*)cos.DataPointer;
        float* sinValues = (float*)sin.DataPointer;
        for (int j = 0; j < half; j++)
        {
            float inverse = 1f / MathF.Pow(theta, (2f * j) / headDim);
            for (int t = 0; t < tokens; t++)
            {
                float angle = t * inverse;
                float cosine = MathF.Cos(angle);
                float sine = MathF.Sin(angle);
                long row = (long)t * headDim;
                cosValues[row + j] = cosine;
                cosValues[row + j + half] = cosine;
                sinValues[row + j] = sine;
                sinValues[row + j + half] = sine;
            }
        }
    }
}
