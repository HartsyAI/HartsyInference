using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// One modality's entry and exit of the structure flow (upstream <c>mm_latent.Latent</c>): <c>input_layer</c> (a linear) plus a position row per
/// token on the way in, and a LayerNorm without affine then <c>out_layer</c> on the way out. Keys under the prefix: <c>input_layer.{weight,bias}</c>,
/// <c>out_layer.{weight,bias}</c> and <c>pos_emb</c> (<c>[tokens, model]</c>; a parameter for the pose latents, a buffer for the shape grid). The
/// weights stay owned by the caller.
/// </summary>
public sealed class SamLatentProjection
{
    private const float OutputNormEps = 1e-5f;

    private readonly Tensor _inW;
    private readonly Tensor _inB;
    private readonly Tensor _outW;
    private readonly Tensor _outB;
    private readonly Tensor _position;

    /// <param name="positionEmbedding">Used when the weights carry no <c>pos_emb</c>, for a conversion that dropped the shape grid buffer
    /// (see <see cref="SamPositionEmbedding.ShapeGrid"/>); <c>[tokens, model]</c> F32, borrowed.</param>
    public SamLatentProjection(IReadOnlyDictionary<string, Tensor> weights, string prefix, int channels, int modelChannels, Tensor? positionEmbedding = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);
        if (channels <= 0 || modelChannels <= 0)
            throw new ArgumentException("The channel counts must be positive.");
        Channels = channels;
        _inW = SamWeights.Require(weights, prefix + "input_layer.weight", modelChannels, channels);
        _inB = SamWeights.Require(weights, prefix + "input_layer.bias", modelChannels);
        _outW = SamWeights.Require(weights, prefix + "out_layer.weight", channels, modelChannels);
        _outB = SamWeights.Require(weights, prefix + "out_layer.bias", channels);
        Tensor position = weights.TryGetValue(prefix + "pos_emb", out Tensor? stored)
            ? stored
            : positionEmbedding ?? throw new InvalidOperationException($"SAM 3D weight '{prefix}pos_emb' is missing and no position embedding was supplied.");
        if (position.DType != DType.F32 || position.Shape.Rank != 2 || position.Shape[1] != modelChannels)
            throw new InvalidOperationException($"The position embedding of '{prefix}' must be F32 [tokens, {modelChannels}]; got {position.DType} {position.Shape}.");
        Tokens = (int)position.Shape[0];
        _position = position.Reshape(new TensorShape(1, Tokens, modelChannels));
        ModelChannels = modelChannels;
    }

    /// <summary>Width of the latent.</summary>
    public int Channels { get; }

    public int ModelChannels { get; }

    /// <summary>Token count of the modality (rows of <c>pos_emb</c>).</summary>
    public int Tokens { get; }

    /// <summary><c>input_layer(x) + pos_emb</c>: F32 <c>[1, tokens, model]</c> from <paramref name="latent"/> <c>[1, tokens, channels]</c>; the caller owns the result.</summary>
    public Tensor ToInput(IBackend backend, Tensor latent)
    {
        ArgumentNullException.ThrowIfNull(backend);
        RequireLatent(latent);
        using Tensor projected = new(new TensorShape(1, Tokens, ModelChannels), DType.F32);
        backend.Linear(projected, latent, _inW, _inB);
        Tensor result = new(projected.Shape, DType.F32);
        backend.Add(result, projected, _position);
        return result;
    }

    /// <summary><c>out_layer(layer_norm(h))</c>: F32 <c>[1, tokens, channels]</c> from <paramref name="hidden"/> <c>[1, tokens, model]</c>; the caller owns the result.</summary>
    public Tensor ToOutput(IBackend backend, Tensor hidden)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (hidden.DType != DType.F32 || !hidden.Shape.Equals(new TensorShape(1, Tokens, ModelChannels)))
            throw new ArgumentException($"Expected F32 [1, {Tokens}, {ModelChannels}] hidden tokens; got {hidden.DType} {hidden.Shape}.", nameof(hidden));
        using Tensor normed = new(hidden.Shape, DType.F32);
        backend.LayerNormNoAffine(normed, hidden, OutputNormEps);
        Tensor result = new(new TensorShape(1, Tokens, Channels), DType.F32);
        backend.Linear(result, normed, _outW, _outB);
        return result;
    }

    private void RequireLatent(Tensor latent)
    {
        if (latent.DType != DType.F32 || !latent.Shape.Equals(new TensorShape(1, Tokens, Channels)))
            throw new ArgumentException($"Expected an F32 [1, {Tokens}, {Channels}] latent; got {latent.DType} {latent.Shape}.", nameof(latent));
    }
}
