using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Accumulates one LoRA layer's pieces while a mapper walks a file's keys, then decides at finalize which decomposition the collected slots describe. Every standard-LoRA mapper shares it, so LyCORIS support is a property of the shared buffer rather than of any one format's parser.</summary>
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

    /// <summary>DoRA magnitude vector when the file carries one.</summary>
    public Tensor? DoraScale { get; set; }

    /// <summary>LoHa first factor's left matrix.</summary>
    public Tensor? HadaW1A { get; set; }

    /// <summary>LoHa first factor's right matrix.</summary>
    public Tensor? HadaW1B { get; set; }

    /// <summary>LoHa second factor's left matrix.</summary>
    public Tensor? HadaW2A { get; set; }

    /// <summary>LoHa second factor's right matrix.</summary>
    public Tensor? HadaW2B { get; set; }

    /// <summary>LoHa first factor's CP/Tucker core.</summary>
    public Tensor? HadaT1 { get; set; }

    /// <summary>LoHa second factor's CP/Tucker core.</summary>
    public Tensor? HadaT2 { get; set; }

    /// <summary>LoKr left factor stored whole.</summary>
    public Tensor? LokrW1 { get; set; }

    /// <summary>LoKr left factor's low-rank left matrix.</summary>
    public Tensor? LokrW1A { get; set; }

    /// <summary>LoKr left factor's low-rank right matrix.</summary>
    public Tensor? LokrW1B { get; set; }

    /// <summary>LoKr right factor stored whole.</summary>
    public Tensor? LokrW2 { get; set; }

    /// <summary>LoKr right factor's low-rank left matrix.</summary>
    public Tensor? LokrW2A { get; set; }

    /// <summary>LoKr right factor's low-rank right matrix.</summary>
    public Tensor? LokrW2B { get; set; }

    /// <summary>LoKr right factor's CP/Tucker core.</summary>
    public Tensor? LokrT2 { get; set; }

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

    /// <summary>Stores <paramref name="tensor"/> in the slot <paramref name="role"/> names. Scalar roles are read here so a mapper never has to know which suffixes are scalars.</summary>
    public void Assign(LoraRole role, Tensor tensor)
    {
        switch (role)
        {
            case LoraRole.Down: Down = tensor; break;
            case LoraRole.Up: Up = tensor; break;
            case LoraRole.Alpha: Alpha = KohyaSdMapper.ReadScalar(tensor); break;
            case LoraRole.DoraScale: DoraScale = tensor; break;
            case LoraRole.HadaW1A: HadaW1A = tensor; break;
            case LoraRole.HadaW1B: HadaW1B = tensor; break;
            case LoraRole.HadaW2A: HadaW2A = tensor; break;
            case LoraRole.HadaW2B: HadaW2B = tensor; break;
            case LoraRole.HadaT1: HadaT1 = tensor; break;
            case LoraRole.HadaT2: HadaT2 = tensor; break;
            case LoraRole.LokrW1: LokrW1 = tensor; break;
            case LoraRole.LokrW1A: LokrW1A = tensor; break;
            case LoraRole.LokrW1B: LokrW1B = tensor; break;
            case LoraRole.LokrW2: LokrW2 = tensor; break;
            case LoraRole.LokrW2A: LokrW2A = tensor; break;
            case LoraRole.LokrW2B: LokrW2B = tensor; break;
            case LoraRole.LokrT2: LokrT2 = tensor; break;
            default:
                throw new HartsyInferenceException(
                    $"LoRA role {role} on '{FirstSourceKey}' is not a layer slot; full-weight diffs are handled separately.");
        }
    }

    /// <summary>Finalizes accumulated groups into layers, picking each group's decomposition from the slots it filled. A group that filled no complete set is skipped with the warning <paramref name="missingWarning"/> builds from its <see cref="FirstSourceKey"/>. Alpha defaults to rank.</summary>
    public static IReadOnlyList<LoraLayer> BuildLayers(Dictionary<(LoraTarget, string), LoraGroupBuffer> groups,
        Func<string, string> missingWarning)
    {
        List<LoraLayer> layers = new(groups.Count);
        foreach (((LoraTarget _, string canonicalKey), LoraGroupBuffer group) in groups)
        {
            LoraDelta? delta = group.BuildDelta();
            if (delta is null)
            {
                Core.Logging.Logs.Warning(missingWarning(group.FirstSourceKey));
                continue;
            }
            layers.Add(new LoraLayer
            {
                TargetKey = canonicalKey,
                Target = group.Target,
                Delta = delta,
            });
        }
        return layers;
    }

    /// <summary>Builds this group's delta, or null when no decomposition's mandatory slots are complete.</summary>
    private LoraDelta? BuildDelta()
    {
        if (HasAnyHada)
        {
            if (HadaW1A is null || HadaW1B is null || HadaW2A is null || HadaW2B is null)
            {
                return null;
            }
            return new LoHaDelta
            {
                W1A = HadaW1A,
                W1B = HadaW1B,
                W2A = HadaW2A,
                W2B = HadaW2B,
                T1 = HadaT1,
                T2 = HadaT2,
                Alpha = Alpha ?? HadaW1B.Shape[0],
                DoraScale = DoraScale,
            };
        }
        if (HasAnyLokr)
        {
            bool leftComplete = LokrW1 is not null || (LokrW1A is not null && LokrW1B is not null);
            bool rightComplete = LokrW2 is not null || (LokrW2A is not null && LokrW2B is not null);
            if (!leftComplete || !rightComplete)
            {
                return null;
            }
            return new LoKrDelta
            {
                W1 = LokrW1,
                W1A = LokrW1A,
                W1B = LokrW1B,
                W2 = LokrW2,
                W2A = LokrW2A,
                W2B = LokrW2B,
                T2 = LokrT2,
                // Unused when neither factor is low-rank: LoKrDelta.Scale is 1.0 there, as ComfyUI's is.
                Alpha = Alpha ?? 1.0f,
                DoraScale = DoraScale,
            };
        }
        if (Down is null || Up is null)
        {
            return null;
        }
        return new StandardLoraDelta
        {
            Down = Down,
            Up = Up,
            Alpha = Alpha ?? Down.Shape[0],
            DoraScale = DoraScale,
        };
    }

    private bool HasAnyHada => HadaW1A is not null || HadaW1B is not null || HadaW2A is not null || HadaW2B is not null;

    private bool HasAnyLokr => LokrW1 is not null || LokrW1A is not null || LokrW1B is not null
        || LokrW2 is not null || LokrW2A is not null || LokrW2B is not null;
}
