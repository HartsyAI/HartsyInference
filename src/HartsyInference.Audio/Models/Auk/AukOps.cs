using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>Shared AuK DiT helpers: shape-validated weight lookup, adaLN modulation, joint/self attention core and the final adaLN head.</summary>
public static class AukOps
{
    /// <summary>float32 machine epsilon, the value <c>nn.RMSNorm(eps=None)</c> uses; upstream runs these norms in the model dtype, the engine in F32.</summary>
    public const float RmsEps = 1.1920929e-7f;

    /// <summary>Throws <see cref="InvalidDataException"/> unless <paramref name="key"/> exists with exactly <paramref name="shape"/>.</summary>
    public static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] shape)
    {
        if (!weights.TryGetValue(key, out Tensor? t)) throw new InvalidDataException($"AuK checkpoint is missing tensor '{key}'.");
        bool ok = t.Shape.Rank == shape.Length;
        for (int i = 0; ok && i < shape.Length; i++) ok = t.Shape[i] == shape[i];
        if (!ok) throw new InvalidDataException($"AuK tensor '{key}' has shape {t.Shape}, expected [{string.Join(", ", shape)}].");
        return t;
    }

    /// <summary>Validated lookup that returns the tensor as F32 (the input itself when already F32, so the dictionary keeps ownership).</summary>
    public static Tensor Take(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] shape) =>
        WhisperOps.EnsureF32(Require(weights, key, shape));

    /// <summary><c>LayerNorm(x) * scale + shift</c>; <paramref name="scale"/> must already include the +1.</summary>
    public static Tensor AdaNorm(IBackend backend, Tensor x, Tensor scale, Tensor shift, float eps)
    {
        Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNormNoAffine(normed, x, eps);
        Tensor result = new(x.Shape, DType.F32);
        backend.AffineBroadcastLastDim(result, normed, scale, shift);
        normed.Dispose();
        return result;
    }

    /// <summary>Non-causal unmasked attention over q/k/v <c>[1, n, heads, headDim]</c>; returns <c>[1, n, heads*headDim]</c> and leaves the inputs to the caller.</summary>
    public static Tensor Attend(IBackend backend, Tensor q, Tensor k, Tensor v, int n, int heads, int headDim)
    {
        TensorShape mh = new(1, heads, n, headDim);
        Tensor qMh = new(mh, DType.F32);
        Tensor kMh = new(mh, DType.F32);
        Tensor vMh = new(mh, DType.F32);
        backend.Permute0213(qMh, q, n, heads, headDim);
        backend.Permute0213(kMh, k, n, heads, headDim);
        backend.Permute0213(vMh, v, n, heads, headDim);
        Tensor attn = new(mh, DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, null, 1f / MathF.Sqrt(headDim));
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();
        Tensor merged = new(new TensorShape(1, n, heads * headDim), DType.F32);
        backend.Permute0213(merged, attn, heads, n, headDim);
        attn.Dispose();
        return merged;
    }

    /// <summary>AdaLayerNorm_Final: <c>LN(x) * (1 + scale) + shift</c> with chunk order (scale, shift) from <c>Linear(silu(t))</c>.</summary>
    public static Tensor FinalAdaLn(IBackend backend, Tensor x, Tensor linearW, Tensor linearB, Tensor siluTime, int dim, float eps)
    {
        Tensor mods = WhisperOps.ProjectLinear(backend, siluTime, linearW, linearB, 1, 1, dim, 2 * dim);
        Tensor scale = new(new TensorShape(1, 1, dim), DType.F32);
        Tensor shift = new(new TensorShape(1, 1, dim), DType.F32);
        backend.SliceLastDim(scale, mods, 0);
        backend.SliceLastDim(shift, mods, dim);
        mods.Dispose();
        backend.AddScalar(scale, scale, 1f);
        Tensor result = AdaNorm(backend, x, scale, shift, eps);
        scale.Dispose(); shift.Dispose();
        return result;
    }

    /// <summary><c>residual + gate * value</c> into a new tensor.</summary>
    public static Tensor GatedResidual(IBackend backend, Tensor residual, Tensor value, Tensor gate)
    {
        Tensor result = new(residual.Shape, DType.F32);
        backend.GatedResidualLastDim(result, residual, value, gate);
        return result;
    }
}
