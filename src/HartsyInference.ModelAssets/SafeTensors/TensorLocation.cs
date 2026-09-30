using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.SafeTensors;

/// <summary>Where one tensor lives: its shard, type, shape and the absolute byte span of its data in that shard's file.</summary>
public sealed record TensorLocation(SafeTensorShard Shard, DType DType, TensorShape Shape, long FileOffset, long ByteLength);
