namespace HartsyInference.LLM.Hy4Flash;

/// <summary>One checkpoint tensor as a safetensors header describes it.</summary>
/// <param name="Name">Tensor key.</param>
/// <param name="DType">Safetensors dtype string, such as <c>BF16</c>, <c>F32</c>, <c>I32</c> or <c>F8_E4M3</c>.</param>
/// <param name="Shape">Dimensions.</param>
/// <param name="Bytes">Data size in bytes.</param>
public readonly record struct Hy4FlashTensorInfo(string Name, string DType, long[] Shape, long Bytes);
