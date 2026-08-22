using System.Text.RegularExpressions;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses Kohya/sd-scripts Flux LoRA files. Handles fused QKV (3-way split for double-stream attention) and fused linear1 (4-way split for single-stream blocks). Flux mlp_ratio is fixed at 4, so single-stream linear1 has shape [7*hidden, in].</summary>
public static class KohyaFluxMapper
{
    private const string DownSuffix = ".lora_down.weight";
    private const string UpSuffix = ".lora_up.weight";
    private const string AlphaSuffix = ".alpha";

    private static readonly Regex _doubleBlock = new(@"^double_blocks_(\d+)_(img|txt)_(attn_qkv|attn_proj|mlp_0|mlp_2|mod_lin)$", RegexOptions.Compiled);
    private static readonly Regex _singleBlock = new(@"^single_blocks_(\d+)_(linear1|linear2|modulation_lin)$", RegexOptions.Compiled);

    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader) => ParseLayers(loader, dottedBflRoots: false);

    /// <summary>Same parse, but <paramref name="dottedBflRoots"/> accepts ComfyUI-style roots: DOTTED original BFL module names under a <c>diffusion_model.</c> prefix (e.g. <c>diffusion_model.double_blocks.0.img_attn.qkv</c>, how Chroma/Flux LoRAs trained against ComfyUI checkpoints ship). The dotted body underscore-normalizes to exactly the kohya body this mapper already translates — fused-QKV splits included — so the whole BFL→diffusers mapping table is shared rather than duplicated.</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader, bool dottedBflRoots)
    {
        Dictionary<(LoraTarget, string), LoraGroupBuffer> groups = [];
        foreach (string key in loader.Descriptors.Keys)
        {
            if (!TryClassifyRoleAndRoot(key, out LoraRole role, out string root))
            {
                continue;
            }

            if (dottedBflRoots && root.StartsWith("diffusion_model.", StringComparison.Ordinal))
            {
                string dottedBody = root["diffusion_model.".Length..];
                ProcessUnetKey(loader, key, role, dottedBody.Replace('.', '_'), groups);
                continue;
            }

            if (root.StartsWith("lora_te_", StringComparison.Ordinal))
            {
                ProcessClipKey(loader, key, role, root["lora_te_".Length..], groups);
                continue;
            }

            if (!root.StartsWith("lora_unet_", StringComparison.Ordinal))
            {
                Logs.Warning($"Kohya Flux LoRA key '{key}' has unrecognized prefix; skipping.");
                continue;
            }

            string body = root["lora_unet_".Length..];
            ProcessUnetKey(loader, key, role, body, groups);
        }

        return LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"Kohya Flux LoRA group '{sourceKey}' missing down or up; skipping.");
    }

    private static void ProcessClipKey(SafeTensorsLoader loader, string sourceKey, LoraRole role, string body, Dictionary<(LoraTarget, string), LoraGroupBuffer> groups)
    {
        string canonicalKey = LoraKeyTransformer.UnderscoreToDot(body) + ".weight";
        AddSimple(loader, sourceKey, role, canonicalKey, LoraTarget.ClipL, groups);
    }

    private static void ProcessUnetKey(SafeTensorsLoader loader, string sourceKey, LoraRole role, string body, Dictionary<(LoraTarget, string), LoraGroupBuffer> groups)
    {
        Match m;
        if ((m = _doubleBlock.Match(body)).Success)
        {
            int i = int.Parse(m.Groups[1].Value);
            string stream = m.Groups[2].Value;
            string sub = m.Groups[3].Value;
            HandleDoubleBlock(loader, sourceKey, role, i, stream, sub, groups);
            return;
        }
        if ((m = _singleBlock.Match(body)).Success)
        {
            int i = int.Parse(m.Groups[1].Value);
            string sub = m.Groups[2].Value;
            HandleSingleBlock(loader, sourceKey, role, i, sub, groups);
            return;
        }

        string? topLevel = MapTopLevelBody(body);
        if (topLevel is null)
        {
            Logs.Warning($"Kohya Flux LoRA UNet key '{sourceKey}' did not match any known module pattern; skipping.");
            return;
        }
        AddSimple(loader, sourceKey, role, topLevel + ".weight", LoraTarget.Transformer, groups);
    }

    private static string? MapTopLevelBody(string body) => body switch
    {
        "img_in" => "x_embedder",
        "txt_in" => "context_embedder",
        "time_in_in_layer" => "time_text_embed.timestep_embedder.linear_1",
        "time_in_out_layer" => "time_text_embed.timestep_embedder.linear_2",
        "vector_in_in_layer" => "time_text_embed.text_embedder.linear_1",
        "vector_in_out_layer" => "time_text_embed.text_embedder.linear_2",
        "guidance_in_in_layer" => "time_text_embed.guidance_embedder.linear_1",
        "guidance_in_out_layer" => "time_text_embed.guidance_embedder.linear_2",
        "final_layer_linear" => "proj_out",
        _ => null,
    };

    private static void HandleDoubleBlock(SafeTensorsLoader loader, string sourceKey, LoraRole role, int i, string stream, string sub, Dictionary<(LoraTarget, string), LoraGroupBuffer> groups)
    {
        bool isImg = stream == "img";
        string blockPrefix = $"transformer_blocks.{i}";
        switch (sub)
        {
            case "attn_qkv":
                {
                    string[] keys = isImg
                        ? [$"{blockPrefix}.attn.to_q.weight", $"{blockPrefix}.attn.to_k.weight", $"{blockPrefix}.attn.to_v.weight"]
                        : [$"{blockPrefix}.attn.add_q_proj.weight", $"{blockPrefix}.attn.add_k_proj.weight", $"{blockPrefix}.attn.add_v_proj.weight"];
                    AddFusedSplit(loader, sourceKey, role, keys, groups, splitWays: 3);
                    return;
                }
            case "attn_proj":
                AddSimple(loader, sourceKey, role,
                    isImg ? $"{blockPrefix}.attn.to_out.0.weight" : $"{blockPrefix}.attn.to_add_out.weight",
                    LoraTarget.Transformer, groups);
                return;
            case "mlp_0":
                AddSimple(loader, sourceKey, role,
                    isImg ? $"{blockPrefix}.ff.net.0.proj.weight" : $"{blockPrefix}.ff_context.net.0.proj.weight",
                    LoraTarget.Transformer, groups);
                return;
            case "mlp_2":
                AddSimple(loader, sourceKey, role,
                    isImg ? $"{blockPrefix}.ff.net.2.weight" : $"{blockPrefix}.ff_context.net.2.weight",
                    LoraTarget.Transformer, groups);
                return;
            case "mod_lin":
                AddSimple(loader, sourceKey, role,
                    isImg ? $"{blockPrefix}.norm1.linear.weight" : $"{blockPrefix}.norm1_context.linear.weight",
                    LoraTarget.Transformer, groups);
                return;
        }
    }

    private static void HandleSingleBlock(SafeTensorsLoader loader, string sourceKey, LoraRole role, int i, string sub, Dictionary<(LoraTarget, string), LoraGroupBuffer> groups)
    {
        string blockPrefix = $"single_transformer_blocks.{i}";
        switch (sub)
        {
            case "linear1":
                {
                    string[] keys =
                    [
                        $"{blockPrefix}.attn.to_q.weight",
                        $"{blockPrefix}.attn.to_k.weight",
                        $"{blockPrefix}.attn.to_v.weight",
                        $"{blockPrefix}.proj_mlp.weight",
                    ];
                    AddFusedSplit(loader, sourceKey, role, keys, groups, splitWays: 4);
                    return;
                }
            case "linear2":
                AddSimple(loader, sourceKey, role, $"{blockPrefix}.proj_out.weight", LoraTarget.Transformer, groups);
                return;
            case "modulation_lin":
                AddSimple(loader, sourceKey, role, $"{blockPrefix}.norm.linear.weight", LoraTarget.Transformer, groups);
                return;
        }
    }

    private static void AddSimple(SafeTensorsLoader loader, string sourceKey, LoraRole role, string canonicalKey, LoraTarget target, Dictionary<(LoraTarget, string), LoraGroupBuffer> groups)
    {
        LoraGroupBuffer group = LoraGroupBuffer.GetOrCreate(groups, target, canonicalKey, sourceKey);
        switch (role)
        {
            case LoraRole.Down: group.Down = loader.GetTensor(sourceKey); break;
            case LoraRole.Up: group.Up = loader.GetTensor(sourceKey); break;
            case LoraRole.Alpha: group.Alpha = ReadScalar(loader.GetTensor(sourceKey)); break;
        }
    }

    private static unsafe void AddFusedSplit(SafeTensorsLoader loader, string sourceKey, LoraRole role, string[] canonicalKeys, Dictionary<(LoraTarget, string), LoraGroupBuffer> groups, int splitWays)
    {
        // Fused QKV / linear1: lora_down is shared across all splits, lora_up is split along dim 0,
        // alpha is shared. Slices of lora_up borrow from the same mmap region.
        Tensor source = loader.GetTensor(sourceKey);

        if (role == LoraRole.Up)
        {
            if (source.Shape.Rank != 2)
                throw new HartsyInferenceException($"Kohya Flux fused up matrix '{sourceKey}' must be 2D, got rank {source.Shape.Rank}.");
            long totalOut = source.Shape[0];
            long rank = source.Shape[1];
            if (totalOut % splitWays != 0 && splitWays == 3)
                throw new HartsyInferenceException($"Kohya Flux fused QKV up matrix '{sourceKey}' shape[0]={totalOut} not divisible by 3.");
            long sliceOut;
            long[] sliceOuts;
            if (splitWays == 3)
            {
                sliceOut = totalOut / 3;
                sliceOuts = [sliceOut, sliceOut, sliceOut];
            }
            else if (splitWays == 4)
            {
                // Flux single-block linear1: [Q | K | V | proj_mlp] with hidden=H, mlp_inner=4H
                // total = 3H + 4H = 7H
                if (totalOut % 7 != 0)
                    throw new HartsyInferenceException($"Kohya Flux fused linear1 up matrix '{sourceKey}' shape[0]={totalOut} not divisible by 7 (expected hidden + 4*hidden Q/K/V/mlp split).");
                long h = totalOut / 7;
                sliceOuts = [h, h, h, 4 * h];
            }
            else throw new InvalidOperationException();

            int elemSize = source.DType.SizeInBytes;
            long offset = 0;
            for (int s = 0; s < splitWays; s++)
            {
                long elements = sliceOuts[s] * rank;
                byte* slicePtr = (byte*)source.DataPointer + offset * elemSize;
                Tensor slice = new(slicePtr, new TensorShape(sliceOuts[s], rank), source.DType);
                LoraGroupBuffer group = LoraGroupBuffer.GetOrCreate(groups, LoraTarget.Transformer, canonicalKeys[s], sourceKey);
                group.Up = slice;
                offset += elements;
            }
            return;
        }

        // Down and Alpha: shared across all canonical keys
        for (int s = 0; s < splitWays; s++)
        {
            LoraGroupBuffer group = LoraGroupBuffer.GetOrCreate(groups, LoraTarget.Transformer, canonicalKeys[s], sourceKey);
            switch (role)
            {
                case LoraRole.Down: group.Down = source; break;
                case LoraRole.Alpha: group.Alpha = ReadScalar(source); break;
            }
        }
    }

    private static bool TryClassifyRoleAndRoot(string key, out LoraRole role, out string root)
    {
        if (key.EndsWith(DownSuffix, StringComparison.Ordinal)) { role = LoraRole.Down; root = key[..^DownSuffix.Length]; return true; }
        if (key.EndsWith(UpSuffix, StringComparison.Ordinal)) { role = LoraRole.Up; root = key[..^UpSuffix.Length]; return true; }
        // PEFT spellings of the same two roles (ComfyUI-style BFL LoRAs pair diffusion_model. roots with .lora_A/.lora_B).
        if (key.EndsWith(".lora_A.weight", StringComparison.Ordinal)) { role = LoraRole.Down; root = key[..^".lora_A.weight".Length]; return true; }
        if (key.EndsWith(".lora_B.weight", StringComparison.Ordinal)) { role = LoraRole.Up; root = key[..^".lora_B.weight".Length]; return true; }
        if (key.EndsWith(AlphaSuffix, StringComparison.Ordinal)) { role = LoraRole.Alpha; root = key[..^AlphaSuffix.Length]; return true; }
        role = default; root = string.Empty; return false;
    }

    private static unsafe float ReadScalar(Tensor t) => KohyaSdMapper.ReadScalar(t);

    private enum LoraRole { Down, Up, Alpha }
}
