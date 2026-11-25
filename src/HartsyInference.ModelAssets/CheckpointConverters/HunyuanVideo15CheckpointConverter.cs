using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Renames a HunyuanVideo-1.5 or MovieRoll DiT checkpoint (Tencent module names, <c>HunyuanVideo_1_5_DiffusionTransformer</c>)
/// to the names the engine's reused blocks load: <c>HunyuanImageBlock</c> reads diffusers names under
/// <c>transformer_blocks.{i}</c>, <c>HunyuanVideoTokenRefiner</c> reads <c>txt_in</c> with <c>in_layer</c>/<c>out_layer</c> and a fused
/// <c>self_attn.qkv</c>. The 1x1x1 <c>img_in.proj</c> Conv3d weight becomes a Linear weight. Fused <c>*_attn_qkv</c> weights are split
/// into q/k/v thirds; a checkpoint that already stores separate <c>*_attn_q/k/v</c> keys is renamed unchanged.
/// <para>Key names come from the reference modules (<c>movieroll/models/transformers</c> at 0b5f73e), not from a checkpoint header:
/// the MovieRoll repository is gated and no header has been read yet. So the converter is strict: a source tensor it does not consume
/// (a <c>guidance_in</c> on a step-distilled build, a <c>vision_in</c> on an image-to-video build, a QK-norm in the refiner) throws instead
/// of being dropped, because a dropped tensor loads "fine" and produces wrong output.</para></summary>
public static class HunyuanVideo15CheckpointConverter
{
    /// <summary>Returns the engine-named weights. Tensors that need no change are passed through; split and reshaped tensors are new
    /// and owned by the caller. Throws when a block or top-level tensor the DiT needs is missing, and when the source holds a tensor
    /// this converter does not consume.</summary>
    public static Dictionary<string, Tensor> Convert(IReadOnlyDictionary<string, Tensor> source, int numBlocks, int hidden)
    {
        ArgumentNullException.ThrowIfNull(source);
        Conversion conversion = new(source);

        Tensor imgIn = conversion.Take("img_in.proj.weight");
        conversion.Output["img_in.proj.weight"] = FlattenPointwiseConv(imgIn);
        conversion.Copy("img_in.proj.bias");
        foreach (string suffix in new[] { "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias" })
            conversion.Copy($"time_in.{suffix}");
        foreach (string name in new[] { "layernorm", "fc1", "fc2", "fc3" })
        {
            conversion.Copy($"byt5_in.{name}.weight");
            conversion.Copy($"byt5_in.{name}.bias");
        }
        conversion.Copy("cond_type_embedding.weight");
        foreach (string name in new[] { "final_layer.adaLN_modulation.1", "final_layer.linear" })
        {
            conversion.Copy($"{name}.weight");
            conversion.Copy($"{name}.bias");
        }

        ConvertRefiner(conversion);
        for (int i = 0; i < numBlocks; i++) ConvertBlock(conversion, i, hidden);
        conversion.ThrowOnUnconsumed();
        return conversion.Output;
    }

    private static void ConvertRefiner(Conversion c)
    {
        foreach (string part in new[] { "weight", "bias" })
        {
            c.Copy($"txt_in.input_embedder.{part}");
            c.Rename($"txt_in.t_embedder.mlp.0.{part}", $"txt_in.t_embedder.in_layer.{part}");
            c.Rename($"txt_in.t_embedder.mlp.2.{part}", $"txt_in.t_embedder.out_layer.{part}");
            c.Rename($"txt_in.c_embedder.linear_1.{part}", $"txt_in.c_embedder.in_layer.{part}");
            c.Rename($"txt_in.c_embedder.linear_2.{part}", $"txt_in.c_embedder.out_layer.{part}");
        }
        for (int i = 0; ; i++)
        {
            string block = $"txt_in.individual_token_refiner.blocks.{i}";
            if (!c.Source.ContainsKey($"{block}.norm1.weight")) break;
            foreach (string part in new[] { "weight", "bias" })
            {
                c.Copy($"{block}.norm1.{part}");
                c.Copy($"{block}.norm2.{part}");
                c.Rename($"{block}.self_attn_qkv.{part}", $"{block}.self_attn.qkv.{part}");
                c.Rename($"{block}.self_attn_proj.{part}", $"{block}.self_attn.proj.{part}");
                c.Rename($"{block}.mlp.fc1.{part}", $"{block}.mlp.0.{part}");
                c.Rename($"{block}.mlp.fc2.{part}", $"{block}.mlp.2.{part}");
                c.Copy($"{block}.adaLN_modulation.1.{part}");
            }
        }
    }

    private static void ConvertBlock(Conversion c, int index, int hidden)
    {
        string from = $"double_blocks.{index}";
        string to = $"transformer_blocks.{index}";
        foreach (string part in new[] { "weight", "bias" })
        {
            c.Rename($"{from}.img_mod.linear.{part}", $"{to}.norm1.linear.{part}");
            c.Rename($"{from}.txt_mod.linear.{part}", $"{to}.norm1_context.linear.{part}");
            c.Rename($"{from}.img_attn_proj.{part}", $"{to}.attn.to_out.0.{part}");
            c.Rename($"{from}.txt_attn_proj.{part}", $"{to}.attn.to_add_out.{part}");
            c.Rename($"{from}.img_mlp.fc1.{part}", $"{to}.ff.net.0.proj.{part}");
            c.Rename($"{from}.img_mlp.fc2.{part}", $"{to}.ff.net.2.{part}");
            c.Rename($"{from}.txt_mlp.fc1.{part}", $"{to}.ff_context.net.0.proj.{part}");
            c.Rename($"{from}.txt_mlp.fc2.{part}", $"{to}.ff_context.net.2.{part}");
            SplitOrRename(c, $"{from}.img_attn", $"{to}.attn", "to_q", "to_k", "to_v", hidden, part);
            SplitOrRename(c, $"{from}.txt_attn", $"{to}.attn", "add_q_proj", "add_k_proj", "add_v_proj", hidden, part);
        }
        c.Rename($"{from}.img_attn_q_norm.weight", $"{to}.attn.norm_q.weight");
        c.Rename($"{from}.img_attn_k_norm.weight", $"{to}.attn.norm_k.weight");
        c.Rename($"{from}.txt_attn_q_norm.weight", $"{to}.attn.norm_added_q.weight");
        c.Rename($"{from}.txt_attn_k_norm.weight", $"{to}.attn.norm_added_k.weight");
    }

    /// <summary>Fused <c>{fromBase}_qkv.{part}</c> is split into three thirds under <c>{toBase}.{q|k|v}.{part}</c>; separate
    /// <c>{fromBase}_q/k/v</c> keys are renamed instead.</summary>
    private static void SplitOrRename(Conversion c, string fromBase, string toBase, string qName, string kName, string vName, int hidden, string part)
    {
        string fusedKey = $"{fromBase}_qkv.{part}";
        if (c.Source.ContainsKey(fusedKey))
        {
            Tensor fused = c.Take(fusedKey);
            if (part == "weight") CheckpointConvertUtils.SplitQkvWeight(fused, hidden, toBase, qName, kName, vName, c.Output);
            else CheckpointConvertUtils.SplitQkvBias(fused, hidden, toBase, qName, kName, vName, c.Output);
            return;
        }
        c.Rename($"{fromBase}_q.{part}", $"{toBase}.{qName}.{part}");
        c.Rename($"{fromBase}_k.{part}", $"{toBase}.{kName}.{part}");
        c.Rename($"{fromBase}_v.{part}", $"{toBase}.{vName}.{part}");
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

    /// <summary>One conversion: the source, the engine-named output, and which source keys have been consumed.</summary>
    private sealed class Conversion(IReadOnlyDictionary<string, Tensor> source)
    {
        private readonly HashSet<string> _consumed = [];

        public IReadOnlyDictionary<string, Tensor> Source { get; } = source;

        public Dictionary<string, Tensor> Output { get; } = [];

        /// <summary>The tensor under <paramref name="key"/>, marked consumed; throws when absent.</summary>
        public Tensor Take(string key)
        {
            if (!Source.TryGetValue(key, out Tensor? tensor))
                throw new KeyNotFoundException($"HunyuanVideo-1.5 checkpoint has no tensor '{key}'.");
            _consumed.Add(key);
            return tensor;
        }

        public void Copy(string key) => Output[key] = Take(key);

        /// <summary>Renames a key when present. Absence is allowed here because the fused and separate q/k/v spellings are alternatives;
        /// a key the DiT needs and the source lacks fails later, in the loader, by name.</summary>
        public void Rename(string from, string to)
        {
            if (Source.ContainsKey(from)) Output[to] = Take(from);
        }

        public void ThrowOnUnconsumed()
        {
            string[] left = Source.Keys.Where(key => !_consumed.Contains(key)).Order(StringComparer.Ordinal).ToArray();
            if (left.Length == 0) return;
            string shown = string.Join(", ", left.Take(8)) + (left.Length > 8 ? $", ... ({left.Length} in total)" : "");
            throw new InvalidDataException(
                $"HunyuanVideo-1.5 checkpoint holds {left.Length} tensor(s) this converter does not consume: {shown}. "
                + "A guidance_in, vision_in or extra norm means a build this DiT does not implement yet; dropping it would load but give wrong output.");
        }
    }
}
