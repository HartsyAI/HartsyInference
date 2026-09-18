using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses Kohya/sd-scripts SD1.5 and SDXL LoRA files into LoraLayer instances. Handles lora_unet_* (UNet), lora_te_* (CLIP-L for SD1.5), lora_te1_* (CLIP-L for SDXL), and lora_te2_* (CLIP-G for SDXL) prefixes with any suffix <see cref="LoraRoleSuffix"/> recognizes.</summary>
public static class KohyaSdMapper
{
    /// <summary>Parses every LoRA layer in the file. The format parameter selects how lora_te*_ prefixes are routed (SD1.5 has lora_te_; SDXL has lora_te1_ / lora_te2_).</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader, LoraFormat format)
    {
        if (format != LoraFormat.KohyaSd15 && format != LoraFormat.KohyaSdxl)
            throw new ArgumentException($"KohyaSdMapper only handles KohyaSd15 / KohyaSdxl, got {format}.", nameof(format));

        Dictionary<(LoraTarget, string), LoraGroupBuffer> groups = [];
        foreach (string key in loader.Descriptors.Keys)
        {
            if (!LoraRoleSuffix.TryStrip(key, out string root, out LoraRole role))
            {
                continue;
            }
            if (role is LoraRole.Diff or LoraRole.BiasDiff)
            {
                Logs.Warning($"LoRA key '{key}' is a full-weight diff, which this format does not carry; skipping.");
                continue;
            }

            if (!TryMapRoot(root, format, out string canonicalKey, out LoraTarget target))
            {
                Logs.Warning($"LoRA key '{key}' has unrecognized prefix; skipping.");
                continue;
            }

            LoraGroupBuffer group = LoraGroupBuffer.GetOrCreate(groups, target, canonicalKey, key);
            group.Assign(role, loader.GetTensor(key));
        }

        return LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"LoRA group '{sourceKey}' is missing a matrix its decomposition needs; skipping.");
    }

    private static bool TryMapRoot(string root, LoraFormat format, out string canonicalKey, out LoraTarget target)
    {
        if (root.StartsWith("lora_unet_", StringComparison.Ordinal))
        {
            string body = root["lora_unet_".Length..];
            string dotted = LoraKeyTransformer.UnderscoreToDot(body) + ".weight";
            target = LoraTarget.UNet;
            if (IsLdmUNetBody(body))
            {
                // The loaded UNet dict is diffusers-named, so LDM-named keys take the same map the checkpoint did.
                string? mapped = format == LoraFormat.KohyaSdxl
                    ? SdxlCheckpointConverter.ConvertUNetKey(dotted)
                    : Sd15CheckpointConverter.ConvertUNetKey(dotted);
                if (mapped is null)
                {
                    canonicalKey = string.Empty;
                    target = default;
                    return false;
                }
                canonicalKey = mapped;
                return true;
            }
            canonicalKey = dotted;
            return true;
        }
        if (format == LoraFormat.KohyaSdxl)
        {
            if (root.StartsWith("lora_te1_", StringComparison.Ordinal))
            {
                string body = root["lora_te1_".Length..];
                canonicalKey = LoraKeyTransformer.UnderscoreToDot(body) + ".weight";
                target = LoraTarget.ClipL;
                return true;
            }
            if (root.StartsWith("lora_te2_", StringComparison.Ordinal))
            {
                string body = root["lora_te2_".Length..];
                canonicalKey = LoraKeyTransformer.UnderscoreToDot(body) + ".weight";
                target = LoraTarget.ClipG;
                return true;
            }
        }
        if (root.StartsWith("lora_te_", StringComparison.Ordinal))
        {
            string body = root["lora_te_".Length..];
            canonicalKey = LoraKeyTransformer.UnderscoreToDot(body) + ".weight";
            target = LoraTarget.ClipL;
            return true;
        }
        canonicalKey = string.Empty;
        target = default;
        return false;
    }

    /// <summary>Whether the body after <c>lora_unet_</c> uses LDM/CompVis block names rather than the diffusers spellings.</summary>
    private static bool IsLdmUNetBody(string body) =>
        body.StartsWith("input_blocks_", StringComparison.Ordinal)
        || body.StartsWith("output_blocks_", StringComparison.Ordinal)
        || body.StartsWith("middle_block_", StringComparison.Ordinal);

    internal static unsafe float ReadScalar(Tensor t)
    {
        if (t.ElementCount < 1)
            throw new HartsyInferenceException($"LoRA alpha tensor is empty (shape={t.Shape}).");
        void* p = t.DataPointer;
        return t.DType.Name switch
        {
            "F32" => *(float*)p,
            "F16" => (float)BitConverter.UInt16BitsToHalf(*(ushort*)p),
            "BF16" => BitConverter.Int32BitsToSingle(*(ushort*)p << 16),
            "F64" => (float)*(double*)p,
            // Integer alphas appear in the wild — h94's IP-Adapter FaceID companion LoRAs store
            // alpha=128 as int64 scalars.
            "I64" => *(long*)p,
            "I32" => *(int*)p,
            _ => throw new HartsyInferenceException($"Unsupported alpha dtype: {t.DType.Name}"),
        };
    }
}
