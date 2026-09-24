using HartsyInference.Core.Backends;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>How a <see cref="CheckpointSource"/> presents a checkpoint's weights to the converter that consumes them.</summary>
/// <remarks>The defaults are what an architecture converter wants: keys and shapes normalized so a GGUF file is
/// indistinguishable from the safetensors build of the same model, and quantization companions already folded onto the
/// weights they belong to. Only a caller that needs the file exactly as written — a quantizer reading a source, a
/// diagnostic — turns them off.</remarks>
public sealed record CheckpointOpenOptions
{
    /// <summary>Move <c>.weight_scale</c> / <c>.scale_weight</c> / <c>.comfy_quant</c> and the bitsandbytes NF4 companions onto the weights they describe, and drop the companion keys.</summary>
    /// <remarks>This runs <b>before</b> the architecture converter, which is the point: a converter renames
    /// <c>.weight</c> without renaming <c>.weight_scale</c>, so folding afterwards loses the pairing silently and the
    /// model runs at 1/scale. Idempotent, so a converter that still folds internally is harmless.</remarks>
    public bool FoldQuantCompanions { get; init; } = true;

    /// <summary>Dequantize NVFP4 weights to fp8 (1 byte/param) rather than F16 (2), halving an all-nvfp4 DiT's resident size.</summary>
    public bool Nvfp4ToFp8 { get; init; }

    /// <summary>Keep NVFP4 weights packed with their scales on <see cref="Core.Tensors.Tensor.QuantInfo"/> instead of unpacking them, for a caller that knows a CUDA backend will consume them.</summary>
    public bool ResidentNvfp4 { get; init; }

    /// <summary>Keep NVFP4 weights packed with their scales as COMPANION KEYS, for a caller whose own layer reads that
    /// form — the opposite trade to <see cref="ResidentNvfp4"/>, which moves the scales onto the tensor.</summary>
    /// <remarks>Set this only for a dictionary that reaches such a consumer directly, never one an architecture
    /// converter renames: surviving companions are exactly what folding exists to prevent a converter from splitting
    /// from their weight. It covers AWQ layers, which <see cref="ResidentNvfp4"/> cannot — <c>QuantInfo</c> has no
    /// <c>pre_quant_scale</c> field, so those layers take the eager path there and widen. Ignored unless
    /// <see cref="FoldQuantCompanions"/> is on, which is the pass it modifies.</remarks>
    public bool KeepNvfp4Companions { get; init; }

    /// <summary>The options for a checkpoint whose nvfp4 groups <paramref name="backend"/> will consume: kept packed where it multiplies them natively, folded to fp8 (half the F16 footprint, the same fp8 GEMM path) everywhere else. One decision, so the native path is reachable on exactly the hardware it targets and nothing else changes.</summary>
    public static CheckpointOpenOptions ForNvfp4Consumer(IBackend backend)
        => backend.Capabilities.NativeBlockScaledGemm ? new() { ResidentNvfp4 = true } : new() { Nvfp4ToFp8 = true };

    /// <summary>Relabel rank-2 GGUF tensors from ggml's <c>[in, out]</c> order to the <c>[out, in]</c> order the engine assumes for a matrix weight.</summary>
    /// <remarks>Only a caller that wants GGUF's own layout — a re-quantizer rewriting a GGUF — sets this false. With it
    /// off, every Linear is transposed and the first matmul derives a degenerate <c>M=0</c>.</remarks>
    public bool RelabelGgufRank2 { get; init; } = true;
}
