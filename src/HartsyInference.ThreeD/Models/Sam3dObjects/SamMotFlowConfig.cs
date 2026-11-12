namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// Shape of the structure flow (<c>SparseStructureFlowTdfyWrapper</c> in <c>mot_sparse_structure_flow.py</c>, values from <c>ss_generator.yaml</c>).
/// A <em>stream</em> is a transformer lane with its own norms, projections and MLP: every modality that is not merged into a group, in
/// <c>latent_mapping</c> order, then one per group key (<c>latent_share_transformer</c>); the members of a group share their stream's weights and
/// run as one token sequence in member order.
/// </summary>
/// <param name="Latents">All modalities in <c>latent_mapping</c> order.</param>
/// <param name="Groups">Group key to the member modalities that share its stream, in concatenation order.</param>
/// <param name="Protected">Streams that attend only to themselves while the others also attend to them (upstream <c>protect_modality_list</c>).</param>
/// <param name="IsShortcut">The model has the extra step-size embedder <c>d_embedder</c>.</param>
public sealed record SamMotFlowConfig(
    int ModelChannels,
    int Heads,
    int NumBlocks,
    int CondChannels,
    double MlpRatio,
    IReadOnlyList<SamMotLatent> Latents,
    IReadOnlyDictionary<string, IReadOnlyList<string>> Groups,
    IReadOnlyList<string> Protected,
    bool IsShortcut)
{
    /// <summary>The released model: 24 blocks of width 1024 and 16 heads over the 4096-token shape grid and the four single-token pose latents.</summary>
    public static SamMotFlowConfig Released { get; } = new(
        ModelChannels: 1024,
        Heads: 16,
        NumBlocks: 24,
        CondChannels: 1024,
        MlpRatio: 4.0,
        Latents:
        [
            new SamMotLatent("6drotation_normalized", 6),
            new SamMotLatent("scale", 3),
            new SamMotLatent("shape", 8),
            new SamMotLatent("translation", 3),
            new SamMotLatent("translation_scale", 1),
        ],
        Groups: new Dictionary<string, IReadOnlyList<string>>
        {
            ["6drotation_normalized"] = ["6drotation_normalized", "translation", "scale", "translation_scale"],
        },
        Protected: ["shape"],
        IsShortcut: true);

    public int HeadDim => ModelChannels / Heads;

    /// <summary>Hidden width of each stream's MLP, <c>int(channels · ratio)</c>.</summary>
    public int MlpHidden => (int)(ModelChannels * MlpRatio);

    /// <summary>The streams in upstream's <c>latent_names</c> order.</summary>
    public IReadOnlyList<string> StreamNames
    {
        get
        {
            HashSet<string> merged = [];
            foreach (IReadOnlyList<string> members in Groups.Values)
                merged.UnionWith(members);
            List<string> names = [];
            foreach (SamMotLatent latent in Latents)
            {
                if (!merged.Contains(latent.Name))
                    names.Add(latent.Name);
            }

            names.AddRange(Groups.Keys);
            return names;
        }
    }

    /// <summary>The stream a modality runs in: its group key when it is a member, otherwise its own name.</summary>
    public string StreamOf(string latent)
    {
        foreach (KeyValuePair<string, IReadOnlyList<string>> group in Groups)
        {
            if (group.Value.Contains(latent))
                return group.Key;
        }

        return latent;
    }

    /// <exception cref="ArgumentException">The configuration is inconsistent.</exception>
    public void Validate()
    {
        if (ModelChannels <= 0 || Heads <= 0 || NumBlocks <= 0 || CondChannels <= 0 || !(MlpRatio > 0.0) || ModelChannels % Heads != 0)
            throw new ArgumentException("The flow needs positive widths, and the model width must divide by the head count.");
        if (Latents.Count == 0 || Latents.Select(l => l.Name).Distinct().Count() != Latents.Count || Latents.Any(l => l.Channels <= 0))
            throw new ArgumentException("The latents must be named uniquely and have positive channel counts.");
        HashSet<string> seen = [];
        foreach (KeyValuePair<string, IReadOnlyList<string>> group in Groups)
        {
            if (group.Value.Count == 0)
                throw new ArgumentException($"Group '{group.Key}' has no members.");
            foreach (string member in group.Value)
            {
                if (Latents.All(l => l.Name != member))
                    throw new ArgumentException($"Group '{group.Key}' names '{member}', which is not a latent.");
                if (!seen.Add(member))
                    throw new ArgumentException($"Latent '{member}' is in more than one group.");
            }
        }

        IReadOnlyList<string> streams = StreamNames;
        if (streams.Distinct().Count() != streams.Count)
            throw new ArgumentException("A group key collides with the name of an ungrouped latent.");
        if (Protected.Distinct().Count() != Protected.Count)
            throw new ArgumentException("A stream is listed as protected more than once.");
        foreach (string name in Protected)
        {
            if (!streams.Contains(name))
                throw new ArgumentException($"Protected stream '{name}' is not a stream of this flow.");
        }
    }
}
