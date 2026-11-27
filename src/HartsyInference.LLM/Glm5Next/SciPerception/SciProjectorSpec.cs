using System.Text.Json;

namespace HartsyInference.LLM.Glm5Next.SciPerception;

/// <summary>Shape of one Sage-SciPerception <c>atom</c> projector, read from its <c>config.json</c> (<c>format_version</c> <c>sage-projector-v1</c>).
/// Only the features the released projectors use are accepted; anything else is refused by name.</summary>
public sealed record SciProjectorSpec
{
    /// <summary>Width of one prepared feature row (the external encoder's state).</summary>
    public required int FeatureDim { get; init; }

    /// <summary>Width of the learned position embedding added per row.</summary>
    public required int IndexDim { get; init; }

    /// <summary>Position embedding rows (the longest object).</summary>
    public required int MaxAtoms { get; init; }

    /// <summary>Hidden width of the projection MLP.</summary>
    public required int HiddenDim { get; init; }

    /// <summary>Width of the rows handed to the language model (<c>output_dim</c>).</summary>
    public required int OutputDim { get; init; }

    /// <summary>Category counts of each discrete per-row feature column (each gets its own embedding).</summary>
    public required IReadOnlyList<int> DiscreteSizes { get; init; }

    /// <summary>Width of each discrete embedding.</summary>
    public required int DiscreteDim { get; init; }

    /// <summary>Width of the per-row side measurements, or 0.</summary>
    public required int SideDim { get; init; }

    /// <summary>Hidden width of the side-measurement MLP.</summary>
    public required int SideHiddenDim { get; init; }

    /// <summary>Whether a frozen per-row classifier adds a learned bridge to its probability (<c>readout_classes</c> 1).</summary>
    public required bool Readout { get; init; }

    /// <summary>Width of the MLP's input: feature, position and discrete embeddings.</summary>
    public int InputWidth => FeatureDim + IndexDim + DiscreteDim * DiscreteSizes.Count;

    /// <summary>Parses a projector <c>config.json</c>.</summary>
    /// <exception cref="InvalidDataException">The projector uses a form the engine does not run.</exception>
    public static SciProjectorSpec Parse(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;
        if (Str(root, "format_version") != "sage-projector-v1") throw new InvalidDataException("Only sage-projector-v1 projectors are supported.");
        JsonElement p = root.TryGetProperty("projector", out JsonElement pe) ? pe : throw new InvalidDataException("config.json has no 'projector'.");
        if (Str(p, "kind") != "atom") throw new InvalidDataException($"Projector kind '{Str(p, "kind")}' is not supported; only 'atom'.");
        foreach (string unsupported in new[] { "depth", "graph_layers", "context_layers", "output_scale" })
            if (p.TryGetProperty(unsupported, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.GetDouble() != 0)
                throw new InvalidDataException($"Projector option '{unsupported}' is not used by any released projector and is not supported.");
        if (p.TryGetProperty("block_exit_norm", out JsonElement b) && b.ValueKind == JsonValueKind.True)
            throw new InvalidDataException("Projector option 'block_exit_norm' is not supported.");
        int classes = IntOr(p, "readout_classes", 0);
        if (classes > 1) throw new InvalidDataException("Only per-row binary readouts (readout_classes 1) are supported.");
        // Upstream defaults readout_uses_side to true, so an absent key reads the side measurements too.
        bool readoutUsesSide = !p.TryGetProperty("readout_uses_side", out JsonElement u) || u.ValueKind != JsonValueKind.False;
        if (classes == 1 && readoutUsesSide && IntOr(p, "side_dim", 0) > 0)
            throw new InvalidDataException("A readout that reads the side measurements is not supported.");
        List<int> sizes = [];
        if (p.TryGetProperty("discrete_sizes", out JsonElement ds) && ds.ValueKind == JsonValueKind.Array)
            foreach (JsonElement e in ds.EnumerateArray()) sizes.Add(e.GetInt32());
        return new SciProjectorSpec
        {
            FeatureDim = Int(p, "feature_dim"),
            IndexDim = Int(p, "index_dim"),
            MaxAtoms = Int(p, "max_atoms"),
            HiddenDim = Int(p, "hidden_dim"),
            OutputDim = Int(root, "output_dim"),
            DiscreteSizes = sizes,
            DiscreteDim = IntOr(p, "discrete_dim", 0),
            SideDim = IntOr(p, "side_dim", 0),
            SideHiddenDim = IntOr(p, "side_hidden_dim", 128),
            Readout = classes == 1,
        };
    }

    private static string? Str(JsonElement e, string name) => e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static int Int(JsonElement e, string name) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : throw new InvalidDataException($"config.json is missing '{name}'.");

    private static int IntOr(JsonElement e, string name, int fallback) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;
}
