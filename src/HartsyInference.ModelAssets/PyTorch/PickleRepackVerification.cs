namespace HartsyInference.ModelAssets.PyTorch;

/// <summary>The outcome of <see cref="StreamingPickleRepack.Verify"/>: whether every tensor of a pickle is in a safetensors file unchanged.</summary>
/// <param name="Equal">True when names, order, shapes, dtypes and bytes all agree.</param>
/// <param name="TensorCount">Tensors compared.</param>
/// <param name="Differences">The first differences found, at most <see cref="StreamingPickleRepack.MaxReportedDifferences"/>.</param>
public readonly record struct PickleRepackVerification(bool Equal, int TensorCount, IReadOnlyList<string> Differences);
