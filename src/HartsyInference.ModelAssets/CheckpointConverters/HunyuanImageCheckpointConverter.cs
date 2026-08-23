using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads + buckets a Hunyuan Image 2.1 single-file safetensors checkpoint into transformer / VAE / CLIP / T5 dictionaries. The diffusers naming convention is preserved verbatim — Hunyuan ships in canonical diffusers layout, so no key remapping is required. FP8 <c>.scale_weight</c> companion tensors are folded into <see cref="Tensor.Fp8ScaleFactor"/> via the shared <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/> helper.</summary>
public sealed class HunyuanImageCheckpointConverter
{
    /// <summary>Result of partitioning a Hunyuan Image 2.1 safetensors file.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Hunyuan Image transformer weights — pass to <c>HunyuanImageTransformer.LoadWeights</c>.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>32-channel Hunyuan VAE weights.</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }

        /// <summary>CLIP-L text encoder weights (when bundled).</summary>
        public required Dictionary<string, Tensor> ClipL { get; init; }

        /// <summary>T5-XXL text encoder weights (when bundled).</summary>
        public required Dictionary<string, Tensor> T5 { get; init; }

        /// <summary>True when at least one transformer linear has FP8 storage.</summary>
        public required bool IsFp8Mix { get; init; }
    }

    /// <summary>Loads and partitions a Hunyuan Image 2.1 single-file checkpoint.</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }

    /// <summary>Partitions a flat dict by key prefix. Original-Tencent layouts (GGUF repacks: <c>double_blocks.*.img_attn_qkv</c>, <c>txt_in.individual_token_refiner</c>, …) are remapped to diffusers naming first via <see cref="ConvertTencentToDiffusers"/>.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        if (allWeights.ContainsKey("double_blocks.0.img_attn_qkv.weight") ||
            allWeights.ContainsKey("model.diffusion_model.double_blocks.0.img_attn_qkv.weight"))
        {
            allWeights = ConvertTencentToDiffusers(allWeights);
        }

        Dictionary<string, Tensor> dequanted = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new();
        Dictionary<string, Tensor> vae = new();
        Dictionary<string, Tensor> clipL = new();
        Dictionary<string, Tensor> t5 = new();

        foreach (KeyValuePair<string, Tensor> kvp in dequanted)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            if (key.StartsWith("vae.", StringComparison.Ordinal) ||
                key.StartsWith("first_stage_model.", StringComparison.Ordinal))
            {
                vae[key] = tensor;
                continue;
            }

            if (key.StartsWith("text_encoder.", StringComparison.Ordinal) ||
                key.StartsWith("text_encoders.clip_l.", StringComparison.Ordinal) ||
                key.StartsWith("conditioner.embedders.0.", StringComparison.Ordinal))
            {
                clipL[key] = tensor;
                continue;
            }

            if (key.StartsWith("text_encoder_2.", StringComparison.Ordinal) ||
                key.StartsWith("text_encoders.t5xxl.", StringComparison.Ordinal) ||
                key.StartsWith("conditioner.embedders.1.", StringComparison.Ordinal))
            {
                t5[key] = tensor;
                continue;
            }

            string transformerKey = key;
            if (transformerKey.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                transformerKey = transformerKey["model.diffusion_model.".Length..];
            else if (transformerKey.StartsWith("transformer.", StringComparison.Ordinal))
                transformerKey = transformerKey["transformer.".Length..];

            transformer[transformerKey] = tensor;
        }

        bool isFp8Mix = DetectFp8Mix(transformer);

        return new ConvertedWeights
        {
            Transformer = transformer,
            Vae = vae,
            ClipL = clipL,
            T5 = t5,
            IsFp8Mix = isFp8Mix,
        };
    }

    /// <summary>Remaps an original-Tencent HunyuanImage 2.1 state dict (the layout GGUF repacks ship) to the diffusers naming <c>HunyuanImageTransformer</c> loads. Mirrors diffusers <c>convert_hunyuan_image_to_diffusers.py</c>: fused qkv / single-block <c>linear1</c> are split with quant-block-aligned row slices, and the final layer's <c>[shift, scale]</c> halves swap to <c>[scale, shift]</c>.</summary>
    public static Dictionary<string, Tensor> ConvertTencentToDiffusers(Dictionary<string, Tensor> source)
    {
        Dictionary<string, Tensor> output = new(source.Count + 200);
        foreach (KeyValuePair<string, Tensor> kvp in source)
        {
            string key = kvp.Key;
            if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                key = key["model.diffusion_model.".Length..];
            Tensor t = kvp.Value;

            if (key.StartsWith("byt5_in.", StringComparison.Ordinal))
            {
                string rest = key["byt5_in.".Length..]
                    .Replace("layernorm.", "norm.").Replace("fc1.", "linear_1.")
                    .Replace("fc2.", "linear_2.").Replace("fc3.", "linear_3.");
                output["context_embedder_2." + rest] = t;
            }
            else if (key.StartsWith("img_in.proj.", StringComparison.Ordinal))
            {
                // patch_size=1 1x1-conv weight consumed as a Linear: flatten rank-4 to [out, in]. GGUF keeps
                // ggml-reversed dims ([1,1,in,out]) since the rank-2 relabel skips rank-4; bytes are already
                // row-major [out][in] either way, so only the shape metadata changes.
                if (key.EndsWith(".weight", StringComparison.Ordinal) && t.Shape.Rank == 4)
                {
                    TensorShape flat = t.Shape[0] == 1
                        ? new TensorShape(t.Shape[3], t.Shape[2])
                        : new TensorShape(t.Shape[0], t.Shape[1]);
                    Tensor copy = new Tensor(flat, t.DType);
                    copy.Fp8ScaleFactor = t.Fp8ScaleFactor;
                    unsafe
                    {
                        long bytes = t.DType.ComputeByteCount(t.ElementCount);
                        Buffer.MemoryCopy((void*)t.DataPointer, (void*)copy.DataPointer, bytes, bytes);
                    }
                    t = copy;
                }
                output["x_embedder.proj." + key["img_in.proj.".Length..]] = t;
            }
            else if (key.StartsWith("txt_in.input_embedder.", StringComparison.Ordinal))
            {
                output["context_embedder.proj_in." + key["txt_in.input_embedder.".Length..]] = t;
            }
            else if (key.StartsWith("txt_in.t_embedder.mlp.", StringComparison.Ordinal))
            {
                string rest = key["txt_in.t_embedder.mlp.".Length..].Replace("0.", "linear_1.").Replace("2.", "linear_2.");
                output["context_embedder.time_text_embed.timestep_embedder." + rest] = t;
            }
            else if (key.StartsWith("txt_in.c_embedder.", StringComparison.Ordinal))
            {
                output["context_embedder.time_text_embed.text_embedder." + key["txt_in.c_embedder.".Length..]] = t;
            }
            else if (key.StartsWith("txt_in.individual_token_refiner.blocks.", StringComparison.Ordinal))
            {
                string rest = key["txt_in.individual_token_refiner.blocks.".Length..];
                int dot = rest.IndexOf('.');
                string prefix = $"context_embedder.token_refiner.refiner_blocks.{rest[..dot]}";
                string sub = rest[(dot + 1)..];
                if (sub.StartsWith("self_attn_qkv.", StringComparison.Ordinal))
                    SplitRows(t, 3, output, $"{prefix}.attn.to_q", $"{prefix}.attn.to_k", $"{prefix}.attn.to_v", sub.EndsWith(".bias", StringComparison.Ordinal));
                else
                    output[$"{prefix}." + sub
                        .Replace("self_attn_proj.", "attn.to_out.0.")
                        .Replace("mlp.fc1.", "ff.net.0.proj.").Replace("mlp.fc2.", "ff.net.2.")
                        .Replace("adaLN_modulation.1.", "norm_out.linear.")] = t;
            }
            else if (key.StartsWith("time_in.mlp.", StringComparison.Ordinal))
            {
                string rest = key["time_in.mlp.".Length..].Replace("0.", "linear_1.").Replace("2.", "linear_2.");
                output["time_guidance_embed.timestep_embedder." + rest] = t;
            }
            else if (key.StartsWith("guidance_in.mlp.", StringComparison.Ordinal))
            {
                string rest = key["guidance_in.mlp.".Length..].Replace("0.", "linear_1.").Replace("2.", "linear_2.");
                output["time_guidance_embed.guidance_embedder." + rest] = t;
            }
            else if (key.StartsWith("double_blocks.", StringComparison.Ordinal))
            {
                string rest = key["double_blocks.".Length..];
                int dot = rest.IndexOf('.');
                string prefix = $"transformer_blocks.{rest[..dot]}";
                string sub = rest[(dot + 1)..];
                if (sub.StartsWith("img_attn_qkv.", StringComparison.Ordinal))
                    SplitRows(t, 3, output, $"{prefix}.attn.to_q", $"{prefix}.attn.to_k", $"{prefix}.attn.to_v", sub.EndsWith(".bias", StringComparison.Ordinal));
                else if (sub.StartsWith("txt_attn_qkv.", StringComparison.Ordinal))
                    SplitRows(t, 3, output, $"{prefix}.attn.add_q_proj", $"{prefix}.attn.add_k_proj", $"{prefix}.attn.add_v_proj", sub.EndsWith(".bias", StringComparison.Ordinal));
                else
                    output[$"{prefix}." + sub
                        .Replace("img_mod.linear.", "norm1.linear.").Replace("txt_mod.linear.", "norm1_context.linear.")
                        .Replace("img_attn_q_norm.", "attn.norm_q.").Replace("img_attn_k_norm.", "attn.norm_k.")
                        .Replace("txt_attn_q_norm.", "attn.norm_added_q.").Replace("txt_attn_k_norm.", "attn.norm_added_k.")
                        .Replace("img_attn_proj.", "attn.to_out.0.").Replace("txt_attn_proj.", "attn.to_add_out.")
                        .Replace("img_mlp.fc1.", "ff.net.0.proj.").Replace("img_mlp.fc2.", "ff.net.2.")
                        .Replace("txt_mlp.fc1.", "ff_context.net.0.proj.").Replace("txt_mlp.fc2.", "ff_context.net.2.")] = t;
            }
            else if (key.StartsWith("single_blocks.", StringComparison.Ordinal))
            {
                string rest = key["single_blocks.".Length..];
                int dot = rest.IndexOf('.');
                string prefix = $"single_transformer_blocks.{rest[..dot]}";
                string sub = rest[(dot + 1)..];
                if (sub.StartsWith("linear1.", StringComparison.Ordinal))
                {
                    // Rows are [3·hidden qkv | mlp_inner] with mlp_inner = 4·hidden → total = 7·hidden.
                    bool isBias = sub.EndsWith(".bias", StringComparison.Ordinal);
                    long hidden = t.Shape.Rank == 1 ? t.Shape[0] / 7 : t.Shape[1];
                    SplitQkvMlpRows(t, (int)hidden, output,
                        $"{prefix}.attn.to_q", $"{prefix}.attn.to_k", $"{prefix}.attn.to_v", $"{prefix}.proj_mlp", isBias);
                }
                else
                    output[$"{prefix}." + sub
                        .Replace("modulation.linear.", "norm.linear.")
                        .Replace("q_norm.", "attn.norm_q.").Replace("k_norm.", "attn.norm_k.")
                        .Replace("linear2.", "proj_out.")] = t;
            }
            else if (key == "final_layer.linear.weight") output["proj_out.weight"] = t;
            else if (key == "final_layer.linear.bias") output["proj_out.bias"] = t;
            else if (key.StartsWith("final_layer.adaLN_modulation.1.", StringComparison.Ordinal))
            {
                // Tencent stores [shift, scale]; the diffusers AdaLN-continuous final norm wants [scale, shift].
                output["norm_out.linear." + key["final_layer.adaLN_modulation.1.".Length..]] = CheckpointConvertUtils.SwapScaleShiftHalves(t);
            }
            else output[key] = t;
        }
        return output;
    }

    private static unsafe void SplitRows(Tensor fused, int parts, Dictionary<string, Tensor> output,
        string name0, string name1, string name2, bool isBias)
    {
        long rows = fused.Shape[0] / parts;
        TensorShape shape = isBias || fused.Shape.Rank == 1 ? new TensorShape(rows) : new TensorShape(rows, fused.Shape[1]);
        long chunkBytes = CheckpointConvertUtils.SliceByteCount(fused, shape.ElementCount);
        string suffix = isBias ? "bias" : "weight";
        string[] names = [name0, name1, name2];
        byte* src = (byte*)fused.DataPointer;
        for (int p = 0; p < parts; p++)
        {
            Tensor part = new Tensor(shape, fused.DType);
            part.Fp8ScaleFactor = fused.Fp8ScaleFactor;
            Buffer.MemoryCopy(src + p * chunkBytes, (void*)part.DataPointer, chunkBytes, chunkBytes);
            output[$"{names[p]}.{suffix}"] = part;
        }
    }

    private static unsafe void SplitQkvMlpRows(Tensor fused, int hidden, Dictionary<string, Tensor> output,
        string qName, string kName, string vName, string mlpName, bool isBias)
    {
        long totalRows = fused.Shape[0];
        long mlpRows = totalRows - 3L * hidden;
        long cols = isBias || fused.Shape.Rank == 1 ? 1 : fused.Shape[1];
        long qkvChunk = CheckpointConvertUtils.SliceByteCount(fused, hidden * cols);
        long mlpChunk = CheckpointConvertUtils.SliceByteCount(fused, mlpRows * cols);
        string suffix = isBias ? "bias" : "weight";
        TensorShape qkvShape = cols == 1 ? new TensorShape(hidden) : new TensorShape(hidden, cols);
        TensorShape mlpShape = cols == 1 ? new TensorShape(mlpRows) : new TensorShape(mlpRows, cols);
        byte* src = (byte*)fused.DataPointer;
        string[] names = [qName, kName, vName];
        long offset = 0;
        for (int p = 0; p < 3; p++)
        {
            Tensor part = new Tensor(qkvShape, fused.DType);
            part.Fp8ScaleFactor = fused.Fp8ScaleFactor;
            Buffer.MemoryCopy(src + offset, (void*)part.DataPointer, qkvChunk, qkvChunk);
            output[$"{names[p]}.{suffix}"] = part;
            offset += qkvChunk;
        }
        Tensor mlp = new Tensor(mlpShape, fused.DType);
        mlp.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        Buffer.MemoryCopy(src + offset, (void*)mlp.DataPointer, mlpChunk, mlpChunk);
        output[$"{mlpName}.{suffix}"] = mlp;
    }

    private static bool DetectFp8Mix(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor t in weights.Values)
        {
            if (t.DType == DType.F8E4M3 || t.DType == DType.F8E5M2) return true;
        }
        return false;
    }
}
