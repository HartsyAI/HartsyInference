using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Borrowed time-embedding weights: <c>t_embedder.mlp.0</c>, <c>t_embedder.mlp.2</c> and <c>t_block.1</c>.</summary>
internal sealed record SanaWmTimeWeights(Tensor FirstWeight, Tensor FirstBias, Tensor SecondWeight, Tensor SecondBias,
    Tensor BlockWeight, Tensor BlockBias);
