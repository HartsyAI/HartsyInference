using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Renames a HunyuanVideo-1.5 or MovieRoll DiT checkpoint (Tencent module names, <c>HunyuanVideo_1_5_DiffusionTransformer</c>)
/// to the names the engine's reused blocks load: <c>HunyuanImageBlock</c> reads diffusers names under
/// <c>transformer_blocks.{i}</c>, <c>HunyuanVideoTokenRefiner</c> reads <c>txt_in</c> with <c>in_layer</c>/<c>out_layer</c> and a fused
/// <c>self_attn.qkv</c>. The 1x1x1 <c>img_in.proj</c> Conv3d weight becomes a Linear weight. Fused <c>*_attn_qkv</c> weights are split
/// into q/k/v thirds; a checkpoint that already stores separate <c>*_attn_q/k/v</c> keys is renamed unchanged.
/// <para>Key names come from the reference modules (<c>movieroll/models/transformers</c> at 0b5f73e), not from a checkpoint header:
/// the MovieRoll repository is gated and no header has been read yet.</para></summary>
public static class HunyuanVideo15CheckpointConverter
{
    /// <summary>Returns the engine-named weights. Tensors that need no change are passed through; split and reshaped tensors are new
    /// and owned by the caller. Throws when a block or a top-level tensor the DiT needs is missing.</summary>
    public static Dictionary<string, Tensor> Convert(IReadOnlyDictionary<string, Tensor> source, int numBlocks, int hidden)
    {
        ArgumentNullException.ThrowIfNull(source);
        Dictionary<string, Tensor> output = [];

        Tensor imgIn = Require(source, "img_in.proj.weight");
        output["img_in.proj.weight"] = FlattenPointwiseConv(imgIn);
        Copy(source, output, "img_in.proj.bias");
        foreach (string suffix in new[] { "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias" })
            Copy(source, output, $"time_in.{suffix}");
        foreach (string name in new[] { "layernorm", "fc1", "fc2", "fc3" })
        {
            Copy(source, output, $"byt5_in.{name}.weight");
            Copy(source, output, $"byt5_in.{name}.bias");
        }
        Copy(source, output, "cond_type_embedding.weight");
        foreach (string name in new[] { "final_layer.adaLN_modulation.1", "final_layer.linear" })
        {
            Copy(source, output, $"{name}.weight");
            Copy(source, output, $"{name}.bias");
        }

        ConvertRefiner(source, output, hidden);
        for (int i = 0; i < numBlocks; i++) ConvertBlock(source, output, i, hidden);
        return output;
    }

    private static void ConvertRefiner(IReadOnlyDictionary<string, Tensor> source, Dictionary<string, Tensor> output, int hidden)
    {
        foreach (string part in new[] { "weight", "bias" })
        {
            Copy(source, output, $"txt_in.input_embedder.{part}");
            Rename(source, output, $"txt_in.t_embedder.mlp.0.{part}", $"txt_in.t_embedder.in_layer.{part}");
            Rename(source, output, $"txt_in.t_embedder.mlp.2.{part}", $"txt_in.t_embedder.out_layer.{part}");
            Rename(source, output, $"txt_in.c_embedder.linear_1.{part}", $"txt_in.c_embedder.in_layer.{part}");
            Rename(source, output, $"txt_in.c_embedder.linear_2.{part}", $"txt_in.c_embedder.out_layer.{part}");
        }
        for (int i = 0; ; i++)
        {
            string from = $"txt_in.individual_token_refiner.blocks.{i}";
            if (!source.ContainsKey($"{from}.norm1.weight")) break;
            string to = from;
            foreach (string part in new[] { "weight", "bias" })
            {
                Copy(source, output, $"{from}.norm1.{part}");
                Copy(source, output, $"{from}.norm2.{part}");
                Rename(source, output, $"{from}.self_attn_qkv.{part}", $"{to}.self_attn.qkv.{part}");
                Rename(source, output, $"{from}.self_attn_proj.{part}", $"{to}.self_attn.proj.{part}");
                Rename(source, output, $"{from}.mlp.fc1.{part}", $"{to}.mlp.0.{part}");
                Rename(source, output, $"{from}.mlp.fc2.{part}", $"{to}.mlp.2.{part}");
                Copy(source, output, $"{from}.adaLN_modulation.1.{part}");
            }
        }
    }

    private static void ConvertBlock(IReadOnlyDictionary<string, Tensor> source, Dictionary<string, Tensor> output, int index, int hidden)
    {
        string from = $"double_blocks.{index}";
        string to = $"transformer_blocks.{index}";
        foreach (string part in new[] { "weight", "bias" })
        {
            Rename(source, output, $"{from}.img_mod.linear.{part}", $"{to}.norm1.linear.{part}");
            Rename(source, output, $"{from}.txt_mod.linear.{part}", $"{to}.norm1_context.linear.{part}");
            Rename(source, output, $"{from}.img_attn_proj.{part}", $"{to}.attn.to_out.0.{part}");
            Rename(source, output, $"{from}.txt_attn_proj.{part}", $"{to}.attn.to_add_out.{part}");
            Rename(source, output, $"{from}.img_mlp.fc1.{part}", $"{to}.ff.net.0.proj.{part}");
            Rename(source, output, $"{from}.img_mlp.fc2.{part}", $"{to}.ff.net.2.{part}");
            Rename(source, output, $"{from}.txt_mlp.fc1.{part}", $"{to}.ff_context.net.0.proj.{part}");
            Rename(source, output, $"{from}.txt_mlp.fc2.{part}", $"{to}.ff_context.net.2.{part}");
            SplitOrRename(source, output, $"{from}.img_attn", $"{to}.attn", "to_q", "to_k", "to_v", hidden, part);
            SplitOrRename(source, output, $"{from}.txt_attn", $"{to}.attn", "add_q_proj", "add_k_proj", "add_v_proj", hidden, part);
        }
        Rename(source, output, $"{from}.img_attn_q_norm.weight", $"{to}.attn.norm_q.weight");
        Rename(source, output, $"{from}.img_attn_k_norm.weight", $"{to}.attn.norm_k.weight");
        Rename(source, output, $"{from}.txt_attn_q_norm.weight", $"{to}.attn.norm_added_q.weight");
        Rename(source, output, $"{from}.txt_attn_k_norm.weight", $"{to}.attn.norm_added_k.weight");
    }

    /// <summary>Fused <c>{fromBase}_qkv.{part}</c> is split into three thirds under <c>{toBase}.{q|k|v}.{part}</c>; separate
    /// <c>{fromBase}_q/k/v</c> keys are renamed instead.</summary>
    private static void SplitOrRename(IReadOnlyDictionary<string, Tensor> source, Dictionary<string, Tensor> output,
        string fromBase, string toBase, string qName, string kName, string vName, int hidden, string part)
    {
        if (source.TryGetValue($"{fromBase}_qkv.{part}", out Tensor? fused))
        {
            if (part == "weight") CheckpointConvertUtils.SplitQkvWeight(fused, hidden, toBase, qName, kName, vName, output);
            else CheckpointConvertUtils.SplitQkvBias(fused, hidden, toBase, qName, kName, vName, output);
            return;
        }
        Rename(source, output, $"{fromBase}_q.{part}", $"{toBase}.{qName}.{part}");
        Rename(source, output, $"{fromBase}_k.{part}", $"{toBase}.{kName}.{part}");
        Rename(source, output, $"{fromBase}_v.{part}", $"{toBase}.{vName}.{part}");
    }

    /// <summary>A 1x1x1 Conv3d weight <c>[out, in, 1, 1, 1]</c> as the Linear weight <c>[out, in]</c> over the same bytes.</summary>
    private static unsafe Tensor FlattenPointwiseConv(Tensor weight)
    {
        if (weight.Shape.Rank != 5 || weight.Shape[2] != 1 || weight.Shape[3] != 1 || weight.Shape[4] != 1)
            throw new InvalidDataException($"img_in.proj.weight must be [out, in, 1, 1, 1] (patch size 1x1x1), got {weight.Shape}.");
        Tensor flat = new(new TensorShape(weight.Shape[0], weight.Shape[1]), weight.DType);
        long bytes = weight.DType.ComputeByteCount(weight.ElementCount);
        Buffer.MemoryCopy((void*)weight.DataPointer, (void*)flat.DataPointer, bytes, bytes);
        return flat;
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> source, string key) =>
        source.TryGetValue(key, out Tensor? tensor) ? tensor : throw new KeyNotFoundException($"HunyuanVideo-1.5 checkpoint has no tensor '{key}'.");

    private static void Copy(IReadOnlyDictionary<string, Tensor> source, Dictionary<string, Tensor> output, string key) =>
        output[key] = Require(source, key);

    private static void Rename(IReadOnlyDictionary<string, Tensor> source, Dictionary<string, Tensor> output, string from, string to)
    {
        // A key absent from the source is skipped for the optional fused/separate alternatives; loaders fail on a missing required key.
        if (source.TryGetValue(from, out Tensor? tensor)) output[to] = tensor;
    }
}
