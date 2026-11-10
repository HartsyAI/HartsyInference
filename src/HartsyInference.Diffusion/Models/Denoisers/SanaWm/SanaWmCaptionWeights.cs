using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Borrowed caption-projection weights: <c>y_embedder.y_proj.fc1</c>, <c>fc2</c> and <c>attention_y_norm</c>.</summary>
internal sealed record SanaWmCaptionWeights(Tensor FirstWeight, Tensor FirstBias, Tensor SecondWeight, Tensor SecondBias, Tensor Norm);
