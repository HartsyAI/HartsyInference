using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;

namespace HartsyInference.ModelHandler.TextEncoders;

/// <summary>Normalizes a HuggingFace-style text-encoder weight dictionary that may carry ComfyUI
/// quantization companions into a form the runtime <c>Linear</c> can consume directly. Concretely it:
/// <list type="bullet">
///   <item>Folds per-tensor fp8 <c>.weight_scale</c> / <c>.scale_weight</c> scalars into
///   <see cref="Tensor.Fp8ScaleFactor"/> (applied for free as the cuBLAS GEMM alpha), via
///   <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/>.</item>
///   <item>Drops the <c>.comfy_quant</c> / <c>*_scale</c> companion tensors so they never reach the model.</item>
///   <item>Leaves plain BF16/F16/F32 checkpoints untouched (no copy when there are no companions).</item>
/// </list>
/// <para>If a weight is still in a U8-packed quant format we don't dequantize yet (NVFP4 / MXFP4 /
/// svdquant), this throws a clear, format-named error <b>at load time</b> instead of letting a raw U8
/// weight reach the GPU and surface as the opaque <c>"GPU cast from U8 to F32 not supported"</c> kernel
/// error mid-generation. The <c>.comfy_quant</c> blob's declared <c>format</c> is included so the message
/// is actionable.</para>
/// Why this exists: text encoders like Qwen3-4B are loaded straight from a raw safetensors file and handed
/// to <see cref="HartsyInference.Diffusion.Models.TextEncoders.LlamaStyleEncoder"/> with no converter in the
/// path, so the per-tensor scale handling every diffusion-backbone converter already does was being skipped
/// for the encoder. Running every encoder load through here closes that gap for all callers at once.</summary>
public static unsafe class TextEncoderQuantNormalizer
{
    /// <summary>Normalizes <paramref name="weights"/> as described on the type. Returns a dictionary safe to
    /// pass to a text-encoder's <c>LoadWeights</c>. The input is not mutated structurally (a new dict is
    /// returned), though <see cref="Tensor.Fp8ScaleFactor"/> may be set on shared fp8 weight tensors.</summary>
    public static Dictionary<string, Tensor> Normalize(IReadOnlyDictionary<string, Tensor> weights)
    {
        // Capture the comfy_quant format declarations before ApplyFp8ScaledDequant drops them, so an
        // unsupported-format error below can name the actual format (e.g. "nvfp4", "mxfp4").
        Dictionary<string, string> formats = new();
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            if (!kvp.Key.EndsWith(".comfy_quant", StringComparison.Ordinal))
                continue;
            string baseKey = kvp.Key[..^".comfy_quant".Length];
            string? format = TryReadComfyQuantFormat(kvp.Value);
            if (format is not null)
                formats[baseKey] = format;
        }

        // fp8_scaled: fold weight_scale into Fp8ScaleFactor and drop comfy_quant / *_scale companions.
        // Plain checkpoints (no companions) are returned as-is by the util.
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
                "does not dequantize yet. Use a BF16/F16 or fp8_scaled (float8_e4m3fn) text-encoder checkpoint instead.");
        }

        return normalized;
    }

    /// <summary>Reads the <c>format</c> string out of a ComfyUI <c>.comfy_quant</c> blob (a tiny U8 tensor
    /// holding UTF-8 JSON like <c>{"format": "float8_e4m3fn"}</c>). Returns null if it can't be parsed.</summary>
    private static string? TryReadComfyQuantFormat(Tensor blob)
    {
        if (blob.DType != DType.U8 || blob.ElementCount <= 0 || blob.ElementCount > 4096)
            return null;
        try
        {
            ReadOnlySpan<byte> json = new(blob.DataPointer, (int)blob.ElementCount);
            using JsonDocument doc = JsonDocument.Parse(System.Text.Encoding.UTF8.GetString(json));
            return doc.RootElement.TryGetProperty("format", out JsonElement fmt) ? fmt.GetString() : null;
        }
        catch
        {
            return null;
        }
    }
}
