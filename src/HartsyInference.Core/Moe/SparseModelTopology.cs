using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe;

/// <summary>
/// What a model is, independent of where its tensors run: layers, which are sparse, their routers, experts and state.
/// Model code builds this; the runtime reads it and decides placement, caching and scheduling.
/// </summary>
public sealed class SparseModelTopology
{
    /// <summary>Version tag of the fingerprint text. Bump it when the canonical form changes.</summary>
    private const string FingerprintVersion = "v1";

    /// <summary>Creates and validates a topology.</summary>
    /// <param name="hiddenSize">Model width H.</param>
    /// <param name="layers">Layers in execution order; each <see cref="SparseLayerDescriptor.Index"/> must equal its position.</param>
    /// <param name="family">Free-form label for diagnostics only; runtime code must not branch on it.</param>
    /// <exception cref="ArgumentException">The topology is inconsistent.</exception>
    public SparseModelTopology(int hiddenSize, IReadOnlyList<SparseLayerDescriptor> layers, string family = "unspecified")
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenSize);
        ArgumentNullException.ThrowIfNull(layers);
        if (layers.Count == 0) throw new ArgumentException("A topology needs at least one layer.", nameof(layers));
        HiddenSize = hiddenSize;
        Layers = layers.ToArray();
        Family = family;
        Validate();
        Capabilities = SparseCapabilities.From(this);
        Fingerprint = ComputeFingerprint();
    }

    /// <summary>Model width H.</summary>
    public int HiddenSize { get; }

    /// <summary>Layers in execution order (target backbone first, draft layers after it).</summary>
    public IReadOnlyList<SparseLayerDescriptor> Layers { get; }

    /// <summary>Diagnostic label; not used for dispatch.</summary>
    public string Family { get; }

    /// <summary>Capabilities the runtime discovers from this topology.</summary>
    public SparseCapabilities Capabilities { get; }

    /// <summary>
    /// Stable content hash of the topology's shapes, routing and programs. Profiles and packs record it so they are not
    /// applied to an incompatible architecture. Excludes <see cref="Family"/>. Numbers are written with the invariant
    /// culture. Enum member names are part of the canonical text: renaming a member changes the fingerprint, so such a
    /// rename must bump <see cref="FingerprintVersion"/>.
    /// </summary>
    public string Fingerprint { get; }

    /// <summary>Number of layers that are sparse.</summary>
    public int SparseLayerCount => Layers.Count(static layer => layer.IsSparse);

    /// <summary>Largest routed-expert count among sparse layers, or 0.</summary>
    public int MaxExpertsPerLayer => Layers.Where(static layer => layer.Moe is not null).Select(static layer => layer.Moe!.ExpertCount).DefaultIfEmpty(0).Max();

    /// <summary>Total payload bytes of every expert in every sparse layer.</summary>
    public long TotalExpertPayloadBytes => Layers.Sum(static layer => layer.Moe?.PayloadBytes ?? 0);

    private void Validate()
    {
        for (int i = 0; i < Layers.Count; i++)
        {
            SparseLayerDescriptor layer = Layers[i] ?? throw new ArgumentException($"Layer {i} is null.", nameof(Layers));
            if (layer.Index != i) throw new ArgumentException($"Layer at position {i} declares index {layer.Index}.", nameof(Layers));
            if (layer.Moe is null) continue;
            layer.Moe.Validated();
            ValidateGroupWidth(i, layer.Moe.Routed, "routed");
            if (layer.Moe.Shared is not null) ValidateGroupWidth(i, layer.Moe.Shared, "shared");
        }
    }

    /// <summary>Every expert in the group, including per-index overrides, must read the model width.</summary>
    private void ValidateGroupWidth(int layer, ExpertGroupDescriptor group, string role)
    {
        if (group.Shape.HiddenSize != HiddenSize)
            throw new ArgumentException($"Layer {layer} {role} experts read width {group.Shape.HiddenSize}, the model is {HiddenSize}.", nameof(Layers));
        if (group.Overrides is null) return;
        foreach (KeyValuePair<int, ExpertDescriptor> pair in group.Overrides)
            if (pair.Value.HiddenSize != HiddenSize)
                throw new ArgumentException(
                    $"Layer {layer} {role} expert {pair.Key} reads width {pair.Value.HiddenSize}, the model is {HiddenSize}.", nameof(Layers));
    }

    private string ComputeFingerprint()
    {
        StringBuilder text = new();
        text.Append(FingerprintVersion).Append(";H=").Append(Num(HiddenSize)).Append(';');
        foreach (SparseLayerDescriptor layer in Layers)
        {
            text.Append(Num(layer.Index)).Append(':').Append(layer.StateKind).Append(layer.IsDraft ? ":draft" : "");
            if (layer.Moe is MoeLayerDescriptor moe)
            {
                RouterDescriptor r = moe.Router;
                text.Append("|R").Append(Num(r.NumExperts)).Append('/').Append(Num(r.TopKDecode)).Append('/').Append(Num(r.TopKPrefill))
                    .Append('/').Append(r.Scoring).Append('/').Append(Num(r.GroupCount)).Append('/').Append(Num(r.GroupsKept))
                    .Append('/').Append(Flag(r.Renormalize)).Append('/').Append(Num(r.RenormEpsilon)).Append('/')
                    .Append(Num(r.Scale)).Append('/').Append(Num(r.LogitDivisor)).Append('/')
                    .Append(Flag(r.HasSelectionBias)).Append('/').Append(Flag(r.HasTokenKindBias)).Append('/').Append(r.BiasSpace);
                AppendGroup(text, "E", moe.Routed);
                if (moe.Shared is not null) AppendGroup(text, "S", moe.Shared);
                text.Append("|G").Append(Flag(moe.SharedIsGated)).Append('|').Append(moe.Program.Activation)
                    .Append('/').Append(Num(moe.Program.GateMax)).Append('/').Append(Num(moe.Program.UpMin))
                    .Append('/').Append(Num(moe.Program.UpMax));
            }
            text.Append(';');
        }
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(text.ToString()));
        return Convert.ToHexString(digest).ToLowerInvariant();
    }

    private static void AppendGroup(StringBuilder text, string tag, ExpertGroupDescriptor group)
    {
        text.Append('|').Append(tag).Append(Num(group.Count)).Append('x').Append(Num(group.Shape.IntermediateSize))
            .Append('x').Append(group.Shape.WeightDType.Name).Append('x').Append(group.Shape.Layout);
        if (group.Overrides is null) return;
        foreach (KeyValuePair<int, ExpertDescriptor> pair in group.Overrides.OrderBy(static p => p.Key))
            text.Append('[').Append(Num(pair.Key)).Append('=').Append(Num(pair.Value.IntermediateSize)).Append('x')
                .Append(pair.Value.WeightDType.Name).Append('x').Append(pair.Value.Layout).Append(']');
    }

    private static string Num(int value) => value.ToString(CultureInfo.InvariantCulture);

    private static string Num(float value) => value.ToString("R", CultureInfo.InvariantCulture);

    private static string Flag(bool value) => value ? "1" : "0";
}
