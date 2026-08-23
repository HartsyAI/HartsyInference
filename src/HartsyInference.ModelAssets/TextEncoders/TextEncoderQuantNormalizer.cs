using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Quant;

namespace HartsyInference.ModelAssets.TextEncoders;

/// <summary>Normalizes a HuggingFace-style text-encoder weight dictionary that may carry ComfyUI quantization companions into a form the runtime <c>Linear</c> can consume directly. Concretely it:
/// <list type="bullet">
/// <item>Folds per-tensor fp8 <c>.weight_scale</c> / <c>.scale_weight</c> scalars into <see cref="Tensor.Fp8ScaleFactor"/> (applied for free as the cuBLAS GEMM alpha), via <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/>.</item>
/// <item>Folds <c>int8_tensorwise</c>'s per-output-row <c>.weight_scale</c> and its <c>.comfy_quant</c> descriptor onto <see cref="Tensor.QuantInfo"/>, leaving the weight packed at 1 byte/param (the Gemma 4 12B LTX 2.5 encoder is 15.4 GB int8 against 26 GB BF16), via <see cref="CheckpointConvertUtils.AttachInt8QuantInfo"/>.</item>
/// <item>Drops the <c>.comfy_quant</c> / <c>*_scale</c> companion tensors so they never reach the model.</item>
/// <item>Leaves plain BF16/F16/F32 checkpoints untouched (no copy when there are no companions).</item>
/// </list>
/// <para>If a weight is still in a U8-packed quant format we don't dequantize yet (NVFP4 / MXFP4 / svdquant), this throws a clear, format-named error <b>at load time</b> instead of letting a raw U8 weight reach the GPU and surface as the opaque <c>"GPU cast from U8 to F32 not supported"</c> kernel error mid-generation. The <c>.comfy_quant</c> blob's declared <c>format</c> is included so the message is actionable.</para> Why this exists: text encoders like Qwen3-4B are loaded straight from a raw safetensors file and handed to <see cref="HartsyInference.Diffusion.Models.TextEncoders.LlamaStyleEncoder"/> with no converter in the path, so the per-tensor scale handling every diffusion-backbone converter already does was being skipped for the encoder. Running every encoder load through here closes that gap for all callers at once.</summary>
public static unsafe class TextEncoderQuantNormalizer
{
    /// <summary>Normalizes <paramref name="weights"/> as described on the type. Returns a dictionary safe to pass to a text-encoder's <c>LoadWeights</c>. The input is not mutated structurally (a new dict is returned), though <see cref="Tensor.Fp8ScaleFactor"/> may be set on shared fp8 weight tensors.</summary>
    public static Dictionary<string, Tensor> Normalize(IReadOnlyDictionary<string, Tensor> weights)
    {
        // Capture the comfy_quant format declarations before ApplyFp8ScaledDequant drops them, so an
        // unsupported-format error below can name the actual format (e.g. "mxfp4", "svdquant").
        Dictionary<string, string> formats = new();
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            if (!kvp.Key.EndsWith(ComfyQuantDescriptor.Suffix, StringComparison.Ordinal))
                continue;
            ComfyQuantDescriptor? descriptor = CheckpointConvertUtils.TryReadComfyQuant(kvp.Value);
            if (descriptor is not null)
                formats[kvp.Key[..^ComfyQuantDescriptor.Suffix.Length]] = descriptor.Format;
        }

        // int8_tensorwise: move the row scale + rotation onto the weight (it stays packed). fp8_scaled: fold
        // weight_scale into Fp8ScaleFactor. Both then drop their comfy_quant / *_scale companions. Plain
        // checkpoints (no companions) are returned as-is by the util.
        Dictionary<string, Tensor> normalized = CheckpointConvertUtils.ApplyFp8ScaledDequant(new Dictionary<string, Tensor>(weights));

        // Anything still U8-packed is a quant format we don't dequantize yet. Fail clearly here rather than
        // deep in a Linear GEMM cast.
        foreach (KeyValuePair<string, Tensor> kvp in normalized)
        {
            if (kvp.Value.DType != DType.U8 || !kvp.Key.EndsWith(".weight", StringComparison.Ordinal))
                continue;
            string baseKey = kvp.Key[..^".weight".Length];
            string format = formats.TryGetValue(baseKey, out string? f) ? f : "unknown (no comfy_quant tag)";
            throw new NotSupportedException(
                $"Text-encoder weight '{kvp.Key}' is {format}-quantized (U8-packed), which HartsyInference " +
                "does not dequantize yet (nvfp4 IS supported and handled upstream — this weight lacks nvfp4's " +
                "block-scale companions). Use a BF16/F16, fp8_scaled (float8_e4m3fn), or nvfp4 text-encoder checkpoint instead.");
        }

        return normalized;
    }
}
