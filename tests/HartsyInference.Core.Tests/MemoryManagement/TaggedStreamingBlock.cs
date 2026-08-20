using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>One-tensor streaming block whose tensor identity reverse-maps to a readable id, so a recorded
/// preload/free/upload call reads as <c>b3</c> instead of a pointer.</summary>
internal sealed class TaggedStreamingBlock : IStreamingBlock
{
    private static readonly Dictionary<Tensor, string> _tags = new();
    private readonly Tensor _tensor;

    public TaggedStreamingBlock(string id, long bytes)
    {
        _tensor = new Tensor(new TensorShape(1), DType.F32);
        lock (_tags) _tags[_tensor] = id;
        EstimatedWeightBytes = bytes;
    }

    public long EstimatedWeightBytes { get; }

    public IEnumerable<Tensor> EnumerateWeights() { yield return _tensor; }

    public static string IdFor(Tensor t)
    {
        lock (_tags) return _tags.TryGetValue(t, out string? id) ? id : "?";
    }
}
