using System.Security.Cryptography;
using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Moe;

/// <summary>
/// What a model is, independent of where its tensors run: layers, which are sparse, their routers, experts and state.
/// Model code builds this; the runtime reads it and decides placement, caching and scheduling.
/// </summary>
public sealed class SparseModelTopology
{
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
    /// applied to an incompatible architecture. Excludes <see cref="Family"/>.
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
            if (layer.Moe.Routed.Shape.HiddenSize != HiddenSize)
                throw new ArgumentException($"Layer {i} experts read width {layer.Moe.Routed.Shape.HiddenSize}, the model is {HiddenSize}.", nameof(Layers));
            if (layer.Moe.Shared is not null && layer.Moe.Shared.Shape.HiddenSize != HiddenSize)
                throw new ArgumentException($"Layer {i} shared experts read width {layer.Moe.Shared.Shape.HiddenSize}, the model is {HiddenSize}.", nameof(Layers));
        }
    }

    private string ComputeFingerprint()
    {
        StringBuilder text = new();
        text.Append("H=").Append(HiddenSize).Append(';');
        foreach (SparseLayerDescriptor layer in Layers)
        {
            text.Append(layer.Index).Append(':').Append(layer.StateKind).Append(layer.IsDraft ? ":draft" : "");
            if (layer.Moe is MoeLayerDescriptor moe)
            {
                RouterDescriptor r = moe.Router;
                text.Append("|R").Append(r.NumExperts).Append('/').Append(r.TopKDecode).Append('/').Append(r.TopKPrefill)
                    .Append('/').Append(r.Scoring).Append('/').Append(r.GroupCount).Append('/').Append(r.GroupsKept)
                    .Append('/').Append(r.Renormalize).Append('/').Append(r.RenormEpsilon.ToString("R")).Append('/')
                    .Append(r.Scale.ToString("R")).Append('/').Append(r.LogitDivisor.ToString("R")).Append('/')
                    .Append(r.HasSelectionBias).Append('/').Append(r.HasTokenKindBias);
                AppendGroup(text, "E", moe.Routed);
                if (moe.Shared is not null) AppendGroup(text, "S", moe.Shared);
                text.Append("|G").Append(moe.SharedIsGated).Append('|').Append(moe.Program.Activation)
                    .Append('/').Append(moe.Program.GateMax.ToString("R")).Append('/').Append(moe.Program.UpMin.ToString("R"))
                    .Append('/').Append(moe.Program.UpMax.ToString("R"));
            }
            text.Append(';');
        }
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(text.ToString()));
        return Convert.ToHexString(digest).ToLowerInvariant();
    }

    private static void AppendGroup(StringBuilder text, string tag, ExpertGroupDescriptor group)
    {
        text.Append('|').Append(tag).Append(group.Count).Append('x').Append(group.Shape.IntermediateSize)
            .Append('x').Append(group.Shape.WeightDType.Name).Append('x').Append(group.Shape.Layout);
        if (group.Overrides is null) return;
        foreach (KeyValuePair<int, ExpertDescriptor> pair in group.Overrides.OrderBy(static p => p.Key))
            text.Append('[').Append(pair.Key).Append('=').Append(pair.Value.IntermediateSize).Append('x').Append(pair.Value.WeightDType.Name).Append(']');
    }
}
