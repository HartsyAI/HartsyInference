using SharpInference.Core.Backends;

namespace SharpInference.Core.Tensors;

/// <summary>Lightweight value type describing a tensor without ownership semantics. Used for passing tensor descriptions through APIs.</summary>
public readonly record struct TensorMetadata(TensorShape Shape, DType DType, int DeviceId, nint DataPointer);
