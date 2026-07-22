using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converts a single-file Qwen-Image checkpoint to per-component diffusers-format dictionaries. Handles the diffusers single-file convention where the transformer, VAE, and Qwen2.5-VL text encoder live under <c>transformer.*</c>, <c>vae.*</c>, and <c>text_encoder.*</c> prefixes; bare diffusers-style keys (no prefix) are treated as transformer weights so checkpoints saved without an outer wrapper still load. Folds ComfyUI-style <c>.scale_weight</c> / BFL-style <c>.weight_scale</c> companions into <see cref="Tensor.Fp8ScaleFactor"/> so FP8 backbones stay at 1 byte/element. The transformer is mostly already in the diffusers naming the model expects (<c>transformer_blocks.{i}.attn.to_q.*</c> etc.), so the converter is largely a re-bucket; the only structural transform is splitting fused QKV linears when a checkpoint ships them combined as <c>attn.qkv.weight</c>.</summary>
public sealed class QwenImageCheckpointConverter
{
    /// <summary>Result of converting a Qwen-Image single-file checkpoint into per-component weight dicts.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Diffusers-format transformer weights consumed by <c>QwenImageTransformer.LoadWeights</c>.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>Qwen2.5-VL text encoder weights (HuggingFace naming: <c>model.embed_tokens.*</c>, <c>model.layers.{i}.*</c>, <c>model.norm.*</c>).</summary>
        public required Dictionary<string, Tensor> TextEncoder { get; init; }

        /// <summary>Diffusers-format VAE weights for the 16-channel Qwen-Image autoencoder.</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }
    }

    /// <summary>Converts a raw checkpoint dictionary to per-component diffusers-format dicts.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new(2500);
        Dictionary<string, Tensor> textEncoder = new(800);
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
                string vaeKey = key["vae.".Length..];
                vae[vaeKey] = tensor;
                continue;
            }

            if (key.StartsWith("first_stage_model.", StringComparison.Ordinal))
            {
                string ldmKey = key["first_stage_model.".Length..];
                string? diffusersKey = CheckpointConvertUtils.ConvertVaeKey(ldmKey);
                if (diffusersKey is not null)
                    vae[diffusersKey] = tensor;
                continue;
            }

            ConvertTransformerKey(key, tensor, transformer);
        }

        return new ConvertedWeights
        {
            Transformer = transformer,
            TextEncoder = textEncoder,
            Vae = vae,
        };
    }

    /// <summary>Loads a single-file Qwen-Image checkpoint and converts it in one call.</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
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

        if (afterIdx == "attn.qkv.weight")
        {
            int innerDim = (int)tensor.Shape[0] / 3;
            SplitQkvWeight(tensor, innerDim, prefix, "attn.to_q", "attn.to_k", "attn.to_v", output);
            return;
        }
        if (afterIdx == "attn.qkv.bias")
        {
            int innerDim = (int)tensor.Shape[0] / 3;
            SplitQkvBias(tensor, innerDim, prefix, "attn.to_q", "attn.to_k", "attn.to_v", output);
            return;
        }
        if (afterIdx == "attn.added_qkv.weight")
        {
            int innerDim = (int)tensor.Shape[0] / 3;
            SplitQkvWeight(tensor, innerDim, prefix, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", output);
            return;
        }
        if (afterIdx == "attn.added_qkv.bias")
        {
            int innerDim = (int)tensor.Shape[0] / 3;
            SplitQkvBias(tensor, innerDim, prefix, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", output);
            return;
        }

        output[$"{prefix}.{afterIdx}"] = tensor;
    }

    /// <summary>Splits a fused QKV weight <c>[3*innerDim, inDim]</c> into three <c>[innerDim, inDim]</c> tensors. Propagates the source FP8 scale companion so cuBLAS' alpha stays correct on each split. Mirrors <see cref="FluxCheckpointConverter"/> / <see cref="Sd3CheckpointConverter"/> helpers.</summary>
    private static unsafe void SplitQkvWeight(Tensor fused, int innerDim, string prefix,
        string qName, string kName, string vName, Dictionary<string, Tensor> output)
    {
        int inDim = (int)fused.Shape[1];
        long rowBytes = (long)inDim * fused.DType.SizeInBytes;
        long chunkBytes = (long)innerDim * rowBytes;
        TensorShape splitShape = new TensorShape(innerDim, inDim);

        Tensor qWeight = new Tensor(splitShape, fused.DType);
        Tensor kWeight = new Tensor(splitShape, fused.DType);
        Tensor vWeight = new Tensor(splitShape, fused.DType);
        qWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        kWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        vWeight.Fp8ScaleFactor = fused.Fp8ScaleFactor;

        byte* src = (byte*)fused.DataPointer;
        Buffer.MemoryCopy(src, (void*)qWeight.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)kWeight.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)vWeight.DataPointer, chunkBytes, chunkBytes);

        output[$"{prefix}.{qName}.weight"] = qWeight;
        output[$"{prefix}.{kName}.weight"] = kWeight;
        output[$"{prefix}.{vName}.weight"] = vWeight;
    }

    private static unsafe void SplitQkvBias(Tensor fused, int innerDim, string prefix,
        string qName, string kName, string vName, Dictionary<string, Tensor> output)
    {
        long elemBytes = fused.DType.SizeInBytes;
        long chunkBytes = (long)innerDim * elemBytes;
        TensorShape splitShape = new TensorShape(innerDim);

        Tensor qBias = new Tensor(splitShape, fused.DType);
        Tensor kBias = new Tensor(splitShape, fused.DType);
        Tensor vBias = new Tensor(splitShape, fused.DType);
        // Propagate fp8_scaled per-tensor scale — biases aren't fp8-scaled in practice, but a non-1 factor must follow the bytes.
        qBias.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        kBias.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        vBias.Fp8ScaleFactor = fused.Fp8ScaleFactor;

        byte* src = (byte*)fused.DataPointer;
        Buffer.MemoryCopy(src, (void*)qBias.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)kBias.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)vBias.DataPointer, chunkBytes, chunkBytes);

        output[$"{prefix}.{qName}.bias"] = qBias;
        output[$"{prefix}.{kName}.bias"] = kBias;
        output[$"{prefix}.{vName}.bias"] = vBias;
    }
}
