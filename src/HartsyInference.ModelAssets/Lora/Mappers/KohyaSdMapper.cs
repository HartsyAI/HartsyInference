using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses Kohya/sd-scripts SD1.5 and SDXL LoRA files into LoraLayer instances. Handles lora_unet_* (UNet), lora_te_* (CLIP-L for SD1.5), lora_te1_* (CLIP-L for SDXL), and lora_te2_* (CLIP-G for SDXL) prefixes with .lora_down.weight / .lora_up.weight / .alpha suffixes.</summary>
public static class KohyaSdMapper
{
    private const string DownSuffix = ".lora_down.weight";
    private const string UpSuffix = ".lora_up.weight";
    private const string AlphaSuffix = ".alpha";

    /// <summary>Parses every LoRA layer in the file. The format parameter selects how lora_te*_ prefixes are routed (SD1.5 has lora_te_; SDXL has lora_te1_ / lora_te2_).</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader, LoraFormat format)
    {
        if (format != LoraFormat.KohyaSd15 && format != LoraFormat.KohyaSdxl)
            throw new ArgumentException($"KohyaSdMapper only handles KohyaSd15 / KohyaSdxl, got {format}.", nameof(format));

        Dictionary<(LoraTarget, string), LoraGroupBuffer> groups = [];
        foreach (string key in loader.Descriptors.Keys)
        {
            if (!TryClassifyRoleAndRoot(key, out LoraRole role, out string root))
            {
                continue;
            }

            if (!TryMapRoot(root, format, out string canonicalKey, out LoraTarget target))
            {
                Logs.Warning($"LoRA key '{key}' has unrecognized prefix; skipping.");
                continue;
            }

            LoraGroupBuffer group = LoraGroupBuffer.GetOrCreate(groups, target, canonicalKey, key);

            switch (role)
            {
                case LoraRole.Down:
                    group.Down = loader.GetTensor(key);
                    break;
                case LoraRole.Up:
                    group.Up = loader.GetTensor(key);
                    break;
                case LoraRole.Alpha:
                    group.Alpha = ReadScalar(loader.GetTensor(key));
                    break;
            }
        }

        return LoraGroupBuffer.BuildLayers(groups,
            sourceKey => $"LoRA group '{sourceKey}' is missing down or up matrix; skipping.");
    }

    private static bool TryClassifyRoleAndRoot(string key, out LoraRole role, out string root)
    {
        if (key.EndsWith(DownSuffix, StringComparison.Ordinal))
        {
            role = LoraRole.Down;
            root = key[..^DownSuffix.Length];
            return true;
        }
        if (key.EndsWith(UpSuffix, StringComparison.Ordinal))
        {
            role = LoraRole.Up;
            root = key[..^UpSuffix.Length];
            return true;
        }
        if (key.EndsWith(AlphaSuffix, StringComparison.Ordinal))
        {
            role = LoraRole.Alpha;
            root = key[..^AlphaSuffix.Length];
            return true;
        }
        role = default;
        root = string.Empty;
        return false;
    }

    private static bool TryMapRoot(string root, LoraFormat format, out string canonicalKey, out LoraTarget target)
    {
        if (root.StartsWith("lora_unet_", StringComparison.Ordinal))
        {
            string body = root["lora_unet_".Length..];
            canonicalKey = LoraKeyTransformer.UnderscoreToDot(body) + ".weight";
            target = LoraTarget.UNet;
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

    internal enum LoraRole { Down, Up, Alpha }
}
