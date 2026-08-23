using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Accumulates the down/up/alpha pieces of one LoRA layer while a mapper walks a file's keys. Shared by the standard-LoRA mappers; mappers with extra roles or a deliberate fixed alpha (Wan, AI-Toolkit) keep their own buffers.</summary>
internal sealed class LoraGroupBuffer
{
    /// <summary>Which model component the finished layer applies to.</summary>
    public required LoraTarget Target { get; init; }

    /// <summary>First source key seen for this group — names the group in missing-matrix warnings.</summary>
    public required string FirstSourceKey { get; init; }

    /// <summary>Down (A) matrix, [rank, in].</summary>
    public Tensor? Down { get; set; }

    /// <summary>Up (B) matrix, [out, rank].</summary>
    public Tensor? Up { get; set; }

    /// <summary>Alpha scalar when the file carries one; null defaults to rank at finalize.</summary>
    public float? Alpha { get; set; }

    /// <summary>Returns the buffer for (<paramref name="target"/>, <paramref name="canonicalKey"/>), creating it with <paramref name="sourceKey"/> recorded when first seen.</summary>
    public static LoraGroupBuffer GetOrCreate(Dictionary<(LoraTarget, string), LoraGroupBuffer> groups,
        LoraTarget target, string canonicalKey, string sourceKey)
    {
        (LoraTarget, string) groupKey = (target, canonicalKey);
        if (!groups.TryGetValue(groupKey, out LoraGroupBuffer? group))
        {
            group = new LoraGroupBuffer { Target = target, FirstSourceKey = sourceKey };
            groups[groupKey] = group;
        }
        return group;
    }

    /// <summary>Finalizes accumulated groups into standard-LoRA layers; a group missing its down or up matrix is skipped with the warning <paramref name="missingWarning"/> builds from its <see cref="FirstSourceKey"/>. Alpha defaults to rank.</summary>
    public static IReadOnlyList<LoraLayer> BuildLayers(Dictionary<(LoraTarget, string), LoraGroupBuffer> groups,
        Func<string, string> missingWarning)
    {
        List<LoraLayer> layers = new(groups.Count);
        foreach (((LoraTarget _, string canonicalKey), LoraGroupBuffer group) in groups)
        {
            if (group.Down is null || group.Up is null)
            {
                Core.Logging.Logs.Warning(missingWarning(group.FirstSourceKey));
                continue;
            }
            int rank = (int)group.Down.Shape[0];
            float alpha = group.Alpha ?? rank;
            layers.Add(new LoraLayer
            {
                TargetKey = canonicalKey,
                Target = group.Target,
                LoraDown = group.Down,
                LoraUp = group.Up,
                Alpha = alpha,
                Rank = rank,
                Variant = LoraVariant.StandardLora,
            });
        }
        return layers;
    }
}
