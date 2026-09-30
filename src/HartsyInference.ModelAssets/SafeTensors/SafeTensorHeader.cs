namespace HartsyInference.ModelAssets.SafeTensors;

/// <summary>A validated safetensors header: every tensor with absolute file offsets, plus the file layout it was checked against.</summary>
public sealed class SafeTensorHeader
{
    /// <summary>Tensors keyed by name, in header order; <see cref="SafeTensorDescriptor.DataOffset"/> is absolute in the file.</summary>
    public required IReadOnlyDictionary<string, SafeTensorDescriptor> Tensors { get; init; }

    /// <summary>String entries of <c>__metadata__</c>, or null when the header has none.</summary>
    public required IReadOnlyDictionary<string, string>? Metadata { get; init; }

    /// <summary>Byte length of the JSON header (the 8-byte length prefix is not counted).</summary>
    public required long HeaderLength { get; init; }

    /// <summary>Absolute offset of the first data byte: 8 plus <see cref="HeaderLength"/>.</summary>
    public required long DataStart { get; init; }

    /// <summary>Length of the file the header was validated against.</summary>
    public required long FileLength { get; init; }
}
