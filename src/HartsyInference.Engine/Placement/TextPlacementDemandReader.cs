using HartsyInference.Core.Tensors;
using HartsyInference.Cpu.Moe;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.Engine.Placement;

/// <summary>One tensor as the checkpoint header describes it.</summary>
/// <param name="Name">GGUF tensor name (<c>blk.3.ffn_gate_exps.weight</c>).</param>
/// <param name="DType">Stored type.</param>
/// <param name="Elements">Element count.</param>
public readonly record struct TextTensorInfo(string Name, DType DType, long Elements);

/// <summary>Reads a <see cref="TextPlacementDemand"/> from a GGUF header, following what the CUDA load path keeps on the device.</summary>
/// <remarks>
/// The rules mirror <c>GgufLanguageModel.PrepareWeights</c> and <c>GenericTransformer.EnumerateWeights</c>: a quantized tensor the
/// device cannot read is widened to F32; a separate output head leaves the token embedding on the host; the routed experts
/// (<c>*_exps</c>) are counted apart from the dense weights; and when the load keeps its fused projection copies (q|k|v and the dense
/// FFN's gate|up) beside the originals, those tensors are counted twice. It is an estimate for a planning decision, so it rounds
/// toward needing more. Known overestimates: the KV cache is sized for full attention on every layer, so a sliding-window or
/// hybrid model needs less than it reads; and a per-layer array of KV head counts falls back to the query head count. Either can
/// refuse a forced <c>gpu</c> placement that would have fit.
/// </remarks>
public static class TextPlacementDemandReader
{
    /// <summary>Reads the header of the GGUF at <paramref name="path"/>.</summary>
    /// <param name="path">The checkpoint.</param>
    /// <param name="contextTokens">Tokens the KV cache is sized for.</param>
    /// <param name="includeRedundantSplits">Whether the load keeps the split originals beside the fused copies.</param>
    /// <param name="kvF16">Whether the KV cache is stored in F16 rather than F32.</param>
    public static TextPlacementDemand FromGguf(string path, int contextTokens, bool includeRedundantSplits, bool kvF16)
    {
        using GgufLoader probe = new();
        probe.Load(path);
        GgufMetadata meta = probe.Metadata;
        string arch = meta.GetString("general.architecture") ?? "";
        TextTensorInfo[] tensors = [.. probe.Descriptors.Values.Select(static d => new TextTensorInfo(d.Name, d.DType, d.Shape.ElementCount))];
        return FromHeader(tensors, key => ReadInteger(meta, $"{arch}.{key}"), contextTokens, includeRedundantSplits, kvF16);
    }

    /// <summary>The same reading over parsed header facts; <paramref name="archValue"/> looks up an architecture key such as
    /// <c>block_count</c>.</summary>
    public static TextPlacementDemand FromHeader(IReadOnlyCollection<TextTensorInfo> tensors, Func<string, long?> archValue,
        int contextTokens, bool includeRedundantSplits, bool kvF16)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentNullException.ThrowIfNull(archValue);
        bool separateHead = tensors.Any(static t => t.Name == "output.weight");
        long dense = 0, experts = 0;
        foreach (TextTensorInfo t in tensors)
        {
            if (t.Name is "per_layer_token_embd.weight" || (separateHead && t.Name == "token_embd.weight")) continue;
            long bytes = DeviceBytes(t);
            if (t.Name.Contains("_exps.", StringComparison.Ordinal))
            {
                experts += bytes;
                continue;
            }
            dense += bytes;
            if (includeRedundantSplits && IsFusedSource(t.Name)) dense += bytes;
        }

        string? hostReason = ExpertsHostReason(tensors, archValue);
        int layers = (int)(archValue("block_count") ?? 0);
        long heads = archValue("attention.head_count") ?? 1;
        long kvHeads = archValue("attention.head_count_kv") ?? heads;
        long hidden = archValue("embedding_length") ?? 0;
        long keyLength = archValue("attention.key_length") ?? (heads > 0 ? hidden / heads : 0);
        long valueLength = archValue("attention.value_length") ?? keyLength;
        long kvPerToken = layers * kvHeads * (keyLength + valueLength) * (kvF16 ? 2 : 4);
        long prefillPerToken = PrefillBytesPerToken(tensors, archValue, hidden, layers, heads, kvHeads, keyLength, valueLength);
        return new TextPlacementDemand(dense, experts, layers, kvPerToken, contextTokens, prefillPerToken)
        {
            ExpertsHostRunnable = hostReason is null,
            ExpertsHostReason = hostReason,
        };
    }

    /// <summary>F32 activation bytes one prompt token holds during prefill, which runs the whole prompt in one pass
    /// (<c>GenerationCapabilities.MaxPrefillChunk</c> defaults to unbounded). Four hidden-wide buffers (residual, normed input,
    /// attention and FFN inputs), the attention intermediates (Q, K, V and the attention output, at the widths the MLA path uses
    /// for <c>deepseek2</c>), and the FFN intermediates at the widest layer: a dense FFN's gate, up, activation and down, or a
    /// MoE token's experts. The widest FFN comes from the tensor headers, since a model may vary it by layer (Gemma-4). Gemma-4's
    /// per-layer embeddings are built for the whole prompt before the first layer and held across the layers: the direct gather,
    /// the projection and the per-layer results, each a token by per-layer-width by layer count, so three of them at the peak.</summary>
    private static long PrefillBytesPerToken(IReadOnlyCollection<TextTensorInfo> tensors, Func<string, long?> archValue,
        long hidden, long layers, long heads, long kvHeads, long keyLength, long valueLength)
    {
        bool mla = archValue("attention.key_length_mla") is not null;
        long keyWidth = archValue("attention.key_length_mla") ?? keyLength;
        long valueWidth = archValue("attention.value_length_mla") ?? valueLength;
        long attnKvHeads = mla ? heads : kvHeads;
        long attnWidth = heads * keyWidth + attnKvHeads * keyWidth + attnKvHeads * valueWidth + heads * valueWidth;

        long ffnWidth = Math.Max(archValue("feed_forward_length") ?? 0,
            (archValue("expert_feed_forward_length") ?? 0) * Math.Max(1, archValue("expert_used_count") ?? 1));
        if (hidden > 0)
        {
            foreach (TextTensorInfo t in tensors)
            {
                if (t.Name.Contains("_exps.", StringComparison.Ordinal)) continue;
                // The shared expert (ffn_*_shexp) runs beside the routed experts in the same layer, so its width counts too.
                if (t.Name.EndsWith(".ffn_up.weight", StringComparison.Ordinal) || t.Name.EndsWith(".ffn_gate.weight", StringComparison.Ordinal)
                    || t.Name.EndsWith(".ffn_up_shexp.weight", StringComparison.Ordinal) || t.Name.EndsWith(".ffn_gate_shexp.weight", StringComparison.Ordinal))
                    ffnWidth = Math.Max(ffnWidth, t.Elements / hidden);
            }
        }
        long pleWidth = archValue("embedding_length_per_layer_input") ?? 0;
        return (hidden * 4 + attnWidth + ffnWidth * 4 + 3 * pleWidth * layers) * sizeof(float);
    }

    /// <summary>Why the packed CPU kernels cannot run this model's experts in place, or null when they can: every <c>*_exps</c>
    /// tensor must be a format <see cref="CpuExpertKernels.Supports"/> accepts, and the model and expert widths must fill its blocks.</summary>
    private static string? ExpertsHostReason(IReadOnlyCollection<TextTensorInfo> tensors, Func<string, long?> archValue)
    {
        long hidden = archValue("embedding_length") ?? 0;
        long inter = archValue("expert_feed_forward_length") ?? archValue("feed_forward_length") ?? 0;
        return ExpertsHostReason(tensors.Where(static t => t.Name.Contains("_exps.", StringComparison.Ordinal)).Select(static t => t.DType),
            hidden, inter);
    }

    /// <summary>Why the packed CPU kernels cannot run routed experts stored as <paramref name="expertTypes"/> with model width
    /// <paramref name="hidden"/> and expert width <paramref name="intermediate"/>, or null when they can: every type must be one
    /// <see cref="CpuExpertKernels.Supports"/> accepts, and both widths must fill its blocks. The reason completes a sentence
    /// starting "Expert offload runs missed experts on the CPU, and".</summary>
    public static string? ExpertsHostReason(IEnumerable<DType> expertTypes, long hidden, long intermediate)
    {
        ArgumentNullException.ThrowIfNull(expertTypes);
        foreach (DType dtype in expertTypes.Distinct())
        {
            if (!CpuExpertKernels.Supports(dtype))
                return $"its experts are stored in {dtype.Name}, which the CPU expert kernels do not read (Q8_0, Q4_0, Q4_K, Q5_K or Q6_K).";
            int block = dtype.BlockElementCount;
            if (hidden <= 0 || intermediate <= 0 || hidden % block != 0 || intermediate % block != 0
                || hidden % CpuExpertKernels.QuantBlock != 0 || intermediate % CpuExpertKernels.QuantBlock != 0)
                return $"its widths ({hidden} and {intermediate}) do not fill {dtype.Name} blocks of {block}.";
        }
        return null;
    }

    /// <summary>Bytes the device holds for one tensor: stored bytes, or F32 when the device cannot read the stored quant.</summary>
    private static long DeviceBytes(TextTensorInfo t) =>
        t.DType.IsQuantized && !GgufLanguageModel.KeepsQuantizedOnGpu(t.DType.Name)
            ? t.Elements * sizeof(float)
            : t.DType.ComputeByteCount(t.Elements);

    /// <summary>Tensors the load fuses into a second copy (q|k|v, dense gate|up) and keeps beside the original.</summary>
    private static bool IsFusedSource(string name) =>
        name.EndsWith(".attn_q.weight", StringComparison.Ordinal) || name.EndsWith(".attn_k.weight", StringComparison.Ordinal)
        || name.EndsWith(".attn_v.weight", StringComparison.Ordinal) || name.EndsWith(".ffn_gate.weight", StringComparison.Ordinal)
        || name.EndsWith(".ffn_up.weight", StringComparison.Ordinal);

    private static long? ReadInteger(GgufMetadata meta, string key) =>
        meta.TryGetValue(key, out object? value)
            ? value switch
            {
                uint u => u,
                int i => i,
                ulong ul => (long)ul,
                long l => l,
                ushort us => us,
                short s => s,
                byte b => b,
                _ => null,
            }
            : null;
}
