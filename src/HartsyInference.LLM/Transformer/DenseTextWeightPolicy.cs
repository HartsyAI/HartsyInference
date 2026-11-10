using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Transformer;

/// <summary>The dtype a dense text decoder loads each Hugging Face weight at, whatever the checkpoint stores: plain float, bitsandbytes
/// NF4, FP8 (block or per-channel), GPTQ, MXFP4/NVFP4 or MLX affine. A quantized projection becomes the GGUF block type of the same
/// width that the backend keeps packed (4-bit sources → Q4_K, else Q4_0; 8-bit → Q8_0), so it stays about as small as on disk; a
/// backend without that kernel gets F16, and one with no packed kernels at all (CPU) F32.</summary>
/// <remarks>Re-quantizing NF4 is a second quantization (the NF4 values are decoded, then rounded onto the GGUF grid), not a
/// bit-exact repack. Measured on MOSS-VL: per weight, NF4 is 9.3% relative RMS from BF16, 12.9% after Q4_0 and 12.2% after Q4_K,
/// but through the stack Q4_0 let a few outlier activations run away (the reply was garbage) where Q4_K held. FP8 to Q8_0 loses
/// less than one FP8 step. Plain float weights keep their dtype unless the caller asks for
/// <paramref name="plainProjections"/>, which re-quantizes the plain projections too (a BF16 checkpoint, or the layers a 4-bit
/// checkpoint left in BF16) to fit a smaller card.</remarks>
public static class DenseTextWeightPolicy
{
    /// <summary>The load dtype of <paramref name="weight"/>.</summary>
    /// <param name="projection">Whether the weight is a rank-2 projection (attention, MLP, <c>lm_head</c>); others stay float.</param>
    /// <param name="backend">The backend that will run the model; null is a host load (F32 for anything quantized).</param>
    /// <param name="plainProjections">A GGUF block type for plain float projections, or null to keep them as stored.</param>
    /// <param name="quantizedProjections">A GGUF block type for quantized projections instead of the width-matched default; a wider
    /// one (Q5_K, Q6_K, Q8_0) keeps a 4-bit source closer to its own values, since re-quantizing to Q4_0 adds a second 4-bit rounding.</param>
    public static DType Target(HfQuantWeight weight, bool projection, IBackend? backend, DType? plainProjections = null,
        DType? quantizedProjections = null)
    {
        ArgumentNullException.ThrowIfNull(weight);
        if (weight.Storage == HfWeightStorage.Plain)
        {
            return projection && plainProjections is { } requested && Packs(backend, weight, requested) ? requested : weight.StoredDType;
        }
        if (!projection || weight.Shape.Rank != 2)
            return DType.F32;
        if (quantizedProjections is { } chosen && Packs(backend, weight, chosen))
            return chosen;
        // A 4-bit source becomes Q4_K where it fits: its per-sub-block scale and minimum follow the source's own block scales more
        // closely than Q4_0's single scale, which matters for the few large weights that drive a model's outlier activations.
        if (IsFourBit(weight) && Packs(backend, weight, DType.Q4_K)) return DType.Q4_K;
        DType packed = IsFourBit(weight) ? DType.Q4_0 : DType.Q8_0;
        if (Packs(backend, weight, packed)) return packed;
        // A 4-bit source on a backend that has Q8_0 but not Q4_0 still stays packed at twice the size, rather than four times as F16.
        if (packed == DType.Q4_0 && Packs(backend, weight, DType.Q8_0)) return DType.Q8_0;
        return backend is null or CpuBackend ? DType.F32 : DType.F16;
    }

    /// <summary>Parses a precision name for plain projections: <c>native</c> (or empty) keeps them; <c>q8_0</c>, <c>q6_k</c>,
    /// <c>q5_k</c>, <c>q4_k</c>, <c>q4_0</c> re-quantize them.</summary>
    /// <exception cref="ArgumentException">Any other name.</exception>
    public static DType? ParsePlainPrecision(string? name)
    {
        string? mode = name?.Trim().ToLowerInvariant();
        if (string.IsNullOrEmpty(mode) || mode is "native" or "default" or "bf16") return null;
        // The engine's one quant-name table, so a name means the same type here as for every other model (a bare q4 is Q4_K).
        return GgufQuantCache.ResolveQuant(mode)
            ?? throw new ArgumentException($"Unknown weight precision '{name}'; expected native, q8_0, q6_k, q5_k, q4_k or q4_0.", nameof(name));
    }

    private static bool Packs(IBackend? backend, HfQuantWeight weight, DType packed) =>
        backend is not null && backend.SupportsResidentQuant(packed) && weight.Shape.Rank == 2 && weight.Shape[1] % packed.BlockElementCount == 0;

    private static bool IsFourBit(HfQuantWeight weight) => weight.Storage switch
    {
        HfWeightStorage.Nf4 or HfWeightStorage.GptqInt4 => true,
        HfWeightStorage.BlockRecipe => weight.Binding?.Encoding is QuantEncoding.Mxfp4E8M0 or QuantEncoding.Nvfp4 or QuantEncoding.AffineInt4,
        _ => false,
    };
}
