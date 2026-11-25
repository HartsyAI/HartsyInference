using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Dcp;

/// <summary>One tensor of a distributed checkpoint: its dtype and shape from the metadata, and the byte range of its standalone <c>torch.save</c> ZIP inside a <c>.distcp</c> file.</summary>
public readonly record struct DcpTensorEntry(string Key, DType DType, long[] Shape, string RelativePath, long Offset, long Length);
