using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Output of the SAM 3 text tower for one prompt.</summary>
/// <param name="Memory">Token features after the resizer, <c>[1, ContextLength, DModel]</c>, F32.</param>
/// <param name="PaddingMask"><c>true</c> where the token is padding (upstream treats token id 0 as padding).</param>
public sealed record Sam3TextFeatures(Tensor Memory, bool[] PaddingMask);
