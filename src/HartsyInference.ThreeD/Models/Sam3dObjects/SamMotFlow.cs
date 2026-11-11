using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// The SAM 3D structure flow (<c>SparseStructureFlowTdfyWrapper</c>): maps the noisy latents (<c>shape</c> and the pose latents) and the condition
/// tokens to the velocity of each latent. Each latent enters through its <see cref="SamLatentProjection"/>, the members of a group are concatenated
/// into one stream, the streams run through <see cref="SamMotFlowConfig.NumBlocks"/> mixture-of-transformers blocks modulated by the timestep embedding
/// (plus the step-size embedding <c>d</c> for the shortcut model), and each latent leaves through its own projection.
/// Weight keys relative to the flow: <c>t_embedder.*</c>, <c>d_embedder.*</c>, <c>latent_mapping.{name}.*</c> and <c>blocks.{i}.*</c> (see <see cref="SamMotBlock"/>);
/// <see cref="Select"/> rebases a checkpoint section whose keys carry a nesting prefix. The unconditional pass of classifier-free guidance is the caller's:
/// with <c>force_zeros_cond</c> it passes an all-zero condition. Weights are F32 and stay owned by the caller.
/// </summary>
public sealed class SamMotFlow
{
    private const string AnchorKey = "t_embedder.mlp.0.weight";

    private readonly SamMotFlowConfig _config;
    private readonly SamTimestepEmbedder _time;
    private readonly SamTimestepEmbedder? _step;
    private readonly Dictionary<string, SamLatentProjection> _projections = new(StringComparer.Ordinal);
    private readonly List<SamMotBlock> _blocks = [];

    /// <param name="positionEmbeddings">Position embeddings by latent name for a conversion whose weights lack <c>pos_emb</c> (the shape grid buffer).</param>
    public SamMotFlow(SamMotFlowConfig config, IReadOnlyDictionary<string, Tensor> weights, IReadOnlyDictionary<string, Tensor>? positionEmbeddings = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        config.Validate();
        _config = config;
        _time = new SamTimestepEmbedder(weights, "t_embedder.", config.ModelChannels);
        if (config.IsShortcut)
            _step = new SamTimestepEmbedder(weights, "d_embedder.", config.ModelChannels);
        foreach (SamMotLatent latent in config.Latents)
        {
            Tensor? position = null;
            positionEmbeddings?.TryGetValue(latent.Name, out position);
            _projections[latent.Name] = new SamLatentProjection(weights, $"latent_mapping.{latent.Name}.", latent.Channels, config.ModelChannels, position);
        }

        for (int i = 0; i < config.NumBlocks; i++)
            _blocks.Add(new SamMotBlock(config, weights, $"blocks.{i}."));
    }

    /// <summary>Token count of a modality, from its position embedding.</summary>
    public int TokensOf(string latent) => _projections[latent].Tokens;

    /// <summary>
    /// Re-keys a checkpoint section so the flow's keys start at the root: finds the one key ending in <c>t_embedder.mlp.0.weight</c> and strips what
    /// precedes it from every key that starts with the same prefix. Needed because the generator section nests the flow under wrappers
    /// (<c>reverse_fn.backbone.</c> in the source) and the exact nesting is unverified until a checkpoint header is read.
    /// </summary>
    /// <exception cref="InvalidOperationException">No key, or more than one, ends in the anchor.</exception>
    public static Dictionary<string, Tensor> Select(IReadOnlyDictionary<string, Tensor> section)
    {
        ArgumentNullException.ThrowIfNull(section);
        List<string> anchors = section.Keys.Where(k => k.EndsWith(AnchorKey, StringComparison.Ordinal)
            && (k.Length == AnchorKey.Length || k[k.Length - AnchorKey.Length - 1] == '.')).ToList();
        if (anchors.Count != 1)
            throw new InvalidOperationException($"Expected exactly one '{AnchorKey}' in the generator section to locate the flow; found {anchors.Count}.");
        string prefix = anchors[0][..^AnchorKey.Length];
        Dictionary<string, Tensor> result = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in section)
        {
            if (entry.Key.StartsWith(prefix, StringComparison.Ordinal))
                result[entry.Key[prefix.Length..]] = entry.Value;
        }

        return result;
    }

    /// <summary>
    /// The velocity of every latent. <paramref name="latents"/> holds an F32 <c>[1, tokens, channels]</c> tensor per modality, <paramref name="t"/> is the model
    /// timestep (<c>1000 · t</c>), <paramref name="cond"/> the condition tokens F32 <c>[1, n, cond]</c> and <paramref name="stepSize"/> the shortcut <c>d</c>
    /// (0 at inference; required by a shortcut model, upstream adds <c>d_embedder(d)</c> whenever a <c>d</c> is passed). Returns tensors of the latents'
    /// shapes, owned by the caller; the inputs are not touched.
    /// </summary>
    public Dictionary<string, Tensor> Forward(IBackend backend, IReadOnlyDictionary<string, Tensor> latents, float t, Tensor cond, float? stepSize = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(latents);
        if (cond.DType != DType.F32 || cond.Shape.Rank != 3 || cond.Shape[0] != 1 || cond.Shape[2] != _config.CondChannels)
            throw new ArgumentException($"Expected F32 condition tokens [1, n, {_config.CondChannels}]; got {cond.DType} {cond.Shape}.", nameof(cond));
        if (stepSize is not null && _step is null)
            throw new ArgumentException("This flow is not a shortcut model and takes no step size.", nameof(stepSize));
        foreach (SamMotLatent latent in _config.Latents)
        {
            if (!latents.ContainsKey(latent.Name))
                throw new ArgumentException($"No latent was given for '{latent.Name}'.", nameof(latents));
        }

        Tensor timeEmbedding = TimeEmbedding(backend, t, stepSize);
        Dictionary<string, Tensor> streams = [];
        try
        {
            foreach (string stream in _config.StreamNames)
            {
                IReadOnlyList<string> members = _config.Groups.TryGetValue(stream, out IReadOnlyList<string>? group) ? group : [stream];
                List<Tensor> inputs = [];
                try
                {
                    foreach (string member in members)
                        inputs.Add(_projections[member].ToInput(backend, latents[member]));
                    streams[stream] = inputs.Count == 1 ? inputs[0] : SamMotOps.Concat(inputs);
                }
                finally
                {
                    foreach (Tensor input in inputs)
                    {
                        if (!ReferenceEquals(input, streams.GetValueOrDefault(stream)))
                            input.Dispose();
                    }
                }
            }

            foreach (SamMotBlock block in _blocks)
                streams = block.Forward(backend, streams, timeEmbedding, cond);

            Dictionary<string, Tensor> velocities = [];
            try
            {
                foreach (SamMotLatent latent in _config.Latents)
                {
                    string stream = _config.StreamOf(latent.Name);
                    Tensor hidden = streams[stream];
                    int start = 0;
                    if (_config.Groups.TryGetValue(stream, out IReadOnlyList<string>? group))
                    {
                        foreach (string member in group)
                        {
                            if (member == latent.Name)
                                break;
                            start += _projections[member].Tokens;
                        }
                    }

                    SamLatentProjection projection = _projections[latent.Name];
                    if (start == 0 && hidden.Shape[1] == projection.Tokens)
                    {
                        velocities[latent.Name] = projection.ToOutput(backend, hidden);
                    }
                    else
                    {
                        using Tensor part = SamMotOps.Tokens(hidden, start, projection.Tokens);
                        velocities[latent.Name] = projection.ToOutput(backend, part);
                    }
                }

                return velocities;
            }
            catch
            {
                foreach (Tensor velocity in velocities.Values)
                    velocity.Dispose();
                throw;
            }
        }
        finally
        {
            timeEmbedding.Dispose();
            foreach (Tensor stream in streams.Values)
                stream.Dispose();
        }
    }

    private Tensor TimeEmbedding(IBackend backend, float t, float? stepSize)
    {
        Tensor time = _time.Embed(backend, t);
        if (stepSize is null)
            return time;
        using Tensor step = _step!.Embed(backend, stepSize.Value);
        Tensor sum = new(time.Shape, DType.F32);
        backend.Add(sum, time, step);
        time.Dispose();
        return sum;
    }
}
