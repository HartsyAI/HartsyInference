namespace HartsyInference.LLM.Hy4Flash;

/// <summary>A parameter key the model reads and the shape its config implies.</summary>
/// <param name="Key">Checkpoint key.</param>
/// <param name="Shape">Expected dimensions, in the checkpoint's storage order.</param>
public readonly record struct Hy4FlashExpectedTensor(string Key, long[] Shape);
