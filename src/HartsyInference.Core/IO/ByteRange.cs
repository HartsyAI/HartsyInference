namespace HartsyInference.Core.IO;

/// <summary>A byte span of a file: where it starts and how long it is.</summary>
public readonly record struct ByteRange(long Offset, int Length);
