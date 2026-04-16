using SharpInference.Core.Backends;

namespace SharpInference.Core.Tensors;

/// <summary>Non-owning, lightweight reference to tensor data. Zero-allocation alternative to Tensor for read-only access.</summary>
public readonly record struct TensorRef(nint DataPointer, TensorShape Shape, DType DType, DeviceKind Device);
