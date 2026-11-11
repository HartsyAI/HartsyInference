using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>One embedder of the fuser: its output width and the (keyword, positional group) pairs it is called for.</summary>
/// <param name="EmbedDim">Width of the embedder's tokens.</param>
/// <param name="Inputs">Each keyword is one call of the embedder; the group picks the learned index embedding added to its tokens.</param>
public sealed record SamFuserEmbedder(int EmbedDim, IReadOnlyList<(string Kwarg, string Group)> Inputs);

/// <summary>
/// The projection half of the upstream <c>EmbedderFuser</c> (released config: <c>projection_net_hidden_dim_multiplier</c> 4.0, learned
/// index embeddings, no compression projector). Each embedder's tokens go through <c>LayerNorm</c> then a Llama-3 SwiGLU
/// (<c>w2(silu(w1 x) · w3 x)</c>, no biases) to the fused width, a learned row of <c>idx_emb</c> for the call's group is added, forced-drop
/// groups are multiplied by 0, and all calls are concatenated on the token axis in embedder order. The embedders themselves (DINO,
/// the point-patch embedder) run before this. Key names follow the module structure in the source and are unverified against a checkpoint.
/// </summary>
public sealed class SamEmbedderFuser
{
    private const float LayerNormEps = 1e-5f;
    private const int HiddenMultipleOf = 256;

    private readonly IReadOnlyList<SamFuserEmbedder> _embedders;
    private readonly IReadOnlyDictionary<string, Tensor> _weights;
    private readonly Dictionary<string, int> _groupIndex = new(StringComparer.Ordinal);

    /// <param name="weights">Tensors of the fuser by name: <c>idx_emb</c> and, per embedder <c>i</c>, <c>projection_nets.i.0.{weight,bias}</c> and
    /// <c>projection_nets.i.1.{w1,w2,w3}.weight</c>, all F32. They stay owned by the caller.</param>
    /// <param name="hiddenMultiplier"><c>projection_net_hidden_dim_multiplier</c>.</param>
    public SamEmbedderFuser(IReadOnlyList<SamFuserEmbedder> embedders, int fusedDim, IReadOnlyDictionary<string, Tensor> weights, double hiddenMultiplier = 4.0)
    {
        ArgumentNullException.ThrowIfNull(embedders);
        ArgumentNullException.ThrowIfNull(weights);
        if (embedders.Count == 0 || fusedDim <= 0 || !(hiddenMultiplier > 0.0))
            throw new ArgumentException("A fuser needs embedders, a positive fused width and a positive hidden multiplier.");
        _embedders = embedders;
        _weights = weights;
        FusedDim = fusedDim;
        Hidden = HiddenDim(fusedDim, hiddenMultiplier);
        foreach (SamFuserEmbedder embedder in embedders)
        {
            foreach ((_, string group) in embedder.Inputs)
                _groupIndex.TryAdd(group, _groupIndex.Count);
        }

        Validate();
    }

    public int FusedDim { get; }

    /// <summary>The SwiGLU hidden width: <c>int(2 · int(multiplier · fused) / 3)</c> rounded up to a multiple of 256.</summary>
    public int Hidden { get; }

    /// <summary>The hidden width of upstream's <c>FeedForward(hidden_dim = int(multiplier · output_dim))</c>.</summary>
    public static int HiddenDim(int outputDim, double multiplier)
    {
        int hidden = (int)(2 * (int)(multiplier * outputDim) / 3);
        return HiddenMultipleOf * ((hidden + HiddenMultipleOf - 1) / HiddenMultipleOf);
    }

    /// <summary>
    /// Fuses the embedders' tokens. <paramref name="tokens"/> holds one F32 <c>[n, embed_dim]</c> tensor per keyword; the result is F32
    /// <c>[sum n, fused]</c> and owned by the caller. <paramref name="forceDrop"/> names the keywords whose tokens are zeroed.
    /// </summary>
    public Tensor Fuse(IBackend backend, IReadOnlyDictionary<string, Tensor> tokens, IReadOnlySet<string>? forceDrop = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(tokens);
        List<Tensor> parts = [];
        try
        {
            for (int i = 0; i < _embedders.Count; i++)
            {
                SamFuserEmbedder embedder = _embedders[i];
                foreach ((string kwarg, string group) in embedder.Inputs)
                {
                    if (!tokens.TryGetValue(kwarg, out Tensor? input))
                        throw new ArgumentException($"No tokens were given for '{kwarg}'.", nameof(tokens));
                    parts.Add(Project(backend, i, input, kwarg, _groupIndex[group], forceDrop is not null && forceDrop.Contains(kwarg)));
                }
            }

            int rows = 0;
            foreach (Tensor part in parts)
                rows += (int)part.Shape[0];
            Tensor result = new(new TensorShape(rows, FusedDim), DType.F32);
            int offset = 0;
            foreach (Tensor part in parts)
            {
                part.AsReadOnlySpan<float>().CopyTo(result.AsSpan<float>()[offset..]);
                offset += (int)part.Shape.ElementCount;
            }

            return result;
        }
        finally
        {
            foreach (Tensor part in parts)
                part.Dispose();
        }
    }

    private Tensor Project(IBackend backend, int index, Tensor input, string kwarg, int group, bool drop)
    {
        SamFuserEmbedder embedder = _embedders[index];
        if (input.DType != DType.F32 || input.Shape.Rank != 2 || input.Shape[1] != embedder.EmbedDim)
            throw new ArgumentException($"Tokens for '{kwarg}' must be F32 [n, {embedder.EmbedDim}]; got {input.DType} {input.Shape}.");
        int n = (int)input.Shape[0];
        string prefix = $"projection_nets.{index}.";
        using Tensor normed = new(input.Shape, DType.F32);
        backend.LayerNorm(normed, input, _weights[prefix + "0.weight"], _weights[prefix + "0.bias"], LayerNormEps);
        using Tensor gate = new(new TensorShape(n, Hidden), DType.F32);
        using Tensor up = new(new TensorShape(n, Hidden), DType.F32);
        backend.Linear(gate, normed, _weights[prefix + "1.w1.weight"], null);
        backend.Linear(up, normed, _weights[prefix + "1.w3.weight"], null);
        backend.Silu(gate, gate);
        backend.Mul(gate, gate, up);
        Tensor output = new(new TensorShape(n, FusedDim), DType.F32);
        try
        {
            backend.Linear(output, gate, _weights[prefix + "1.w2.weight"], null);
            Span<float> data = output.AsSpan<float>();
            ReadOnlySpan<float> row = _weights["idx_emb"].AsReadOnlySpan<float>().Slice(group * FusedDim, FusedDim);
            for (int r = 0; r < n; r++)
            {
                Span<float> target = data.Slice(r * FusedDim, FusedDim);
                for (int c = 0; c < FusedDim; c++)
                    target[c] = drop ? 0f : target[c] + row[c];
            }

            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    private void Validate()
    {
        Require("idx_emb", _groupIndex.Count + 1, FusedDim);
        for (int i = 0; i < _embedders.Count; i++)
        {
            int dim = _embedders[i].EmbedDim;
            string prefix = $"projection_nets.{i}.";
            Require(prefix + "0.weight", dim);
            Require(prefix + "0.bias", dim);
            Require(prefix + "1.w1.weight", Hidden, dim);
            Require(prefix + "1.w3.weight", Hidden, dim);
            Require(prefix + "1.w2.weight", FusedDim, Hidden);
        }
    }

    private void Require(string name, params int[] shape)
    {
        if (!_weights.TryGetValue(name, out Tensor? tensor))
            throw new InvalidOperationException($"The fuser is missing '{name}'.");
        TensorShape expected = new(Array.ConvertAll(shape, d => (long)d));
        if (tensor.DType != DType.F32 || tensor.Shape != expected)
            throw new InvalidOperationException($"'{name}' must be F32 {expected}; got {tensor.DType} {tensor.Shape}.");
    }
}
