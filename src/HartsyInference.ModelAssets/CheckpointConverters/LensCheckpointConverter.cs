using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Mxfp4;
using HartsyInference.ModelAssets.Mxfp8;
using HartsyInference.ModelAssets.Nvfp4;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converts a Microsoft Lens checkpoint (`microsoft/Lens`, `microsoft/Lens-Turbo`, `microsoft/Lens-Base`) into per-component diffusers-format dictionaries. The upstream weights ship in diffusers-native naming (the model is built from <c>diffusers.ConfigMixin</c>), so most keys pass through unchanged; the only structural transforms are: (1) splitting the fused <c>transformer_blocks.{i}.attn.img_qkv.{weight,bias}</c> into <c>to_q / to_k / to_v</c> and the fused <c>txt_qkv</c> into <c>add_q_proj / add_k_proj / add_v_proj</c> — mirrors <see cref="QwenImageCheckpointConverter"/> and <see cref="ChromaCheckpointConverter"/>; (2) bucketing by <c>transformer.</c> / <c>text_encoder.</c> / <c>vae.</c> prefix (the HF repo lives in three subdirectories, but combined single-file dumps also work). Folds FP8 scale-companion entries via <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/>. MXFP4-packed GPT-OSS MoE expert weights are dequantized to forward-ready F32 here via <see cref="Mxfp4Codec.DequantGptOssExpertsInPlace"/> (the transposed <c>[E, hidden, 2·intermediate]</c> / <c>[E, intermediate, hidden]</c> runtime layout) — no-op for already-dequantized / BF16 checkpoints.</summary>
public sealed class LensCheckpointConverter
{
    /// <summary>Result of converting a Lens checkpoint.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Diffusers-format transformer weights consumed by <see cref="HartsyInference.Diffusion.Models.Denoisers.LensTransformer.LoadWeights"/>.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>GPT-OSS text encoder weights (HuggingFace naming under <c>model.*</c>). Diffusers MXFP4-packed MoE experts are dequantized to F32 by <see cref="Convert"/>; ComfyUI NVFP4-packed experts pass through PACKED from <see cref="ConvertComfyTextEncoder"/> (forward-time streaming dequant). Either way, ready to hand straight to the encoder loader.</summary>
        public required Dictionary<string, Tensor> TextEncoder { get; init; }

        /// <summary>Flux.2 semantic VAE weights. Same layout as Flux.2 Klein — handed off to the existing VAE loader.</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }
    }

    /// <summary>Converts a raw weights dict into per-component dicts.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new(3000);
        Dictionary<string, Tensor> textEncoder = new(500);
        Dictionary<string, Tensor> vae = new(300);

        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            if (key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || key == "scaled_fp8")
                continue;

            if (key.StartsWith("transformer.", StringComparison.Ordinal))
            {
                ConvertTransformerKey(key["transformer.".Length..], tensor, transformer);
                continue;
            }

            if (key.StartsWith("text_encoder.", StringComparison.Ordinal))
            {
                textEncoder[key["text_encoder.".Length..]] = tensor;
                continue;
            }

            if (key.StartsWith("vae.", StringComparison.Ordinal))
            {
                vae[key["vae.".Length..]] = tensor;
                continue;
            }

            ConvertTransformerKey(key, tensor, transformer);
        }

        // Dequantize the GPT-OSS MoE expert weights from their MXFP4 on-disk packing to forward-ready
        // F32 (the transposed [E, hidden, 2·intermediate] / [E, intermediate, hidden] layout). No-op for
        // already-dequantized / BF16 checkpoints.
        int dequanted = Mxfp4Codec.DequantGptOssExpertsInPlace(textEncoder);
        if (dequanted > 0)
            Logs.Verbose($"Lens: dequantized {dequanted} MXFP4 GPT-OSS expert tensors to F32.");

        return new ConvertedWeights
        {
            Transformer = transformer,
            TextEncoder = textEncoder,
            Vae = vae,
        };
    }

    /// <summary>Loads a single-file Lens checkpoint and converts it in one call.</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }

    // ──────────────────────────────────────────────────────────────────────
    // ComfyUI split-file layout (Comfy-Org/Lens): the DiT, GPT-OSS text encoder, and VAE ship as three
    // SEPARATE single-component safetensors with NO bucket prefix. The DiT uses diffusers-native naming
    // (BF16 or MXFP8-quantized); the text encoder uses HuggingFace GPT-OSS naming WITHOUT the `model.`
    // prefix and ComfyUI's `gate_up_proj.weight`/`.bias`/`.weight_scale` MoE expert sub-keys (BF16 +
    // NVFP4-quantized experts). These methods produce the same per-component dicts as Convert.
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>Loads + converts the three ComfyUI Lens files (DiT, GPT-OSS text encoder, Flux.2 VAE). Returns the converted weights plus the open loaders (keep them alive — passthrough tensors are memory-mapped views into the files).</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader[] loaders) LoadAndConvertComfy(
        string ditPath, string textEncoderPath, string vaePath)
    {
        SafeTensorsLoader ditLoader = new(); ditLoader.Load(ditPath);
        SafeTensorsLoader teLoader = new(); teLoader.Load(textEncoderPath);
        SafeTensorsLoader vaeLoader = new(); vaeLoader.Load(vaePath);
        ConvertedWeights converted = ConvertComfy(
            ditLoader.GetAllTensors(), teLoader.GetAllTensors(), vaeLoader.GetAllTensors());
        return (converted, [ditLoader, teLoader, vaeLoader]);
    }

    /// <summary>Converts pre-loaded ComfyUI component dicts into the per-component <see cref="ConvertedWeights"/>.</summary>
    public static ConvertedWeights ConvertComfy(
        Dictionary<string, Tensor> dit, Dictionary<string, Tensor> textEncoder, Dictionary<string, Tensor> vae)
        => new()
        {
            Transformer = ConvertComfyDit(dit),
            TextEncoder = ConvertComfyTextEncoder(textEncoder),
            Vae = vae,
        };

    /// <summary>Converts the ComfyUI DiT file: dequantizes MXFP8 linears to BF16 (no-op on the BF16 file), then applies the standard fused-QKV split + passthrough. Keys have no <c>transformer.</c> prefix.</summary>
    public static Dictionary<string, Tensor> ConvertComfyDit(Dictionary<string, Tensor> dit)
    {
        int dq = Mxfp8Codec.DequantInPlace(dit);
        if (dq > 0) Logs.Verbose($"Lens(ComfyUI): dequantized {dq} MXFP8 DiT linears to BF16.");

        Dictionary<string, Tensor> transformer = new(dit.Count);
        foreach (KeyValuePair<string, Tensor> kvp in dit)
        {
            if (kvp.Key.EndsWith(".comfy_quant", StringComparison.Ordinal)) continue;
            ConvertTransformerKey(kvp.Key, kvp.Value, transformer);
        }
        return transformer;
    }

    /// <summary>Converts the ComfyUI GPT-OSS text encoder file: renames the ComfyUI expert biases (<c>gate_up_proj.bias</c> → <c>gate_up_proj_bias</c>) to the HF form the encoder loads, prepends the <c>model.</c> prefix, casts BF16 weights to F32 (the CPU encoder GEMM is F32-only), and drops the <c>comfy_quant</c> blobs + the embedded <c>tokenizer_json</c>.
    /// <para><b>NVFP4 MoE expert banks pass through PACKED</b> (U8 nibbles + F8E4M3 swizzled block scales + F32 per-expert globals, all still mmap-backed views into the file): <see cref="HartsyInference.Diffusion.Models.TextEncoders.GptOssMoeFfn"/> dequantizes one expert at a time transiently at forward time via <see cref="Nvfp4Codec.DequantExpertSlice"/>. Eagerly materializing the 20B-parameter expert bank at F32 costs ~76 GB of host RAM and gets the process OOM-killed on 64 GB hosts.</para></summary>
    public static Dictionary<string, Tensor> ConvertComfyTextEncoder(Dictionary<string, Tensor> textEncoder)
    {
        Dictionary<string, Tensor> output = new(textEncoder.Count);
        foreach (KeyValuePair<string, Tensor> kvp in textEncoder)
        {
            string key = kvp.Key;
            if (key.EndsWith(".comfy_quant", StringComparison.Ordinal)) continue;
            if (key == "tokenizer_json") continue;

            // ComfyUI MoE expert biases use `gate_up_proj.bias`; the encoder expects HF `gate_up_proj_bias`.
            if (key.EndsWith("experts.gate_up_proj.bias", StringComparison.Ordinal) ||
                key.EndsWith("experts.down_proj.bias", StringComparison.Ordinal))
                key = key[..^".bias".Length] + "_bias";

            output[$"model.{key}"] = IsPackedNvfp4ExpertTensor(key, kvp.Value) ? kvp.Value : ToF32IfFloat(kvp.Value);
        }
        return output;
    }

    /// <summary>True for the three NVFP4 expert-bank tensors (<c>experts.*.weight</c> U8 packed nibbles, <c>.weight_scale</c>, <c>.weight_scale_2</c>) that must stay in their quantized on-disk form.</summary>
    private static bool IsPackedNvfp4ExpertTensor(string key, Tensor tensor)
    {
        if (!key.Contains("experts.", StringComparison.Ordinal))
            return false;
        if (key.EndsWith(".weight_scale", StringComparison.Ordinal) ||
            key.EndsWith(".weight_scale_2", StringComparison.Ordinal))
            return true;
        return key.EndsWith(".weight", StringComparison.Ordinal) && tensor.DType == DType.U8;
    }

    private static Tensor ToF32IfFloat(Tensor t)
    {
        DType d = t.DType;
        bool needsCast = d == DType.BF16 || d == DType.F16 || d == DType.F8E4M3 || d == DType.F8E5M2;
        return needsCast ? t.CastTo(DType.F32) : t;
    }

    private static void ConvertTransformerKey(string key, Tensor tensor, Dictionary<string, Tensor> output)
    {
        if (key.StartsWith("transformer_blocks.", StringComparison.Ordinal))
        {
            ConvertBlockKey(key["transformer_blocks.".Length..], tensor, output);
            return;
        }

        output[key] = tensor;
    }

    private static void ConvertBlockKey(string rest, Tensor tensor, Dictionary<string, Tensor> output)
    {
        int firstDot = rest.IndexOf('.');
        if (firstDot < 0)
        {
            output["transformer_blocks." + rest] = tensor;
            return;
        }

        string blockIdxStr = rest[..firstDot];
        string afterIdx = rest[(firstDot + 1)..];
        string prefix = $"transformer_blocks.{blockIdxStr}";

        if (afterIdx == "attn.img_qkv.weight")
        {
            int innerDim = (int)tensor.Shape[0] / 3;
            CheckpointConvertUtils.SplitQkvWeight(tensor, innerDim, prefix, "attn.to_q", "attn.to_k", "attn.to_v", output);
            return;
        }
        if (afterIdx == "attn.img_qkv.bias")
        {
            int innerDim = (int)tensor.Shape[0] / 3;
            CheckpointConvertUtils.SplitQkvBias(tensor, innerDim, prefix, "attn.to_q", "attn.to_k", "attn.to_v", output);
            return;
        }
        if (afterIdx == "attn.txt_qkv.weight")
        {
            int innerDim = (int)tensor.Shape[0] / 3;
            CheckpointConvertUtils.SplitQkvWeight(tensor, innerDim, prefix, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", output);
            return;
        }
        if (afterIdx == "attn.txt_qkv.bias")
        {
            int innerDim = (int)tensor.Shape[0] / 3;
            CheckpointConvertUtils.SplitQkvBias(tensor, innerDim, prefix, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", output);
            return;
        }

        output[$"{prefix}.{afterIdx}"] = tensor;
    }
}
