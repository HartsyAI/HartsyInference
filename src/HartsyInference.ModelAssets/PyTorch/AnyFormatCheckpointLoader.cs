using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.PyTorch;

/// <summary>A drop-in for <see cref="PytorchPickleLoader"/> that also reads a safetensors file, chosen by the file's
/// bytes rather than its name, so a family that ships a <c>.pth</c> loads a converted copy at the same path.
/// <para>Tensors are owned in both cases, matching the pickle loader's contract that the loader may be disposed as
/// soon as <see cref="GetAllTensors"/> returns; a safetensors read would otherwise hand back views into an mmap.</para></summary>
public sealed class AnyFormatCheckpointLoader : IDisposable
{
    private PytorchPickleLoader? _pickle;
    private Dictionary<string, Tensor>? _owned;
    private int _disposed;

    /// <summary>Whether the loaded file was a safetensors file.</summary>
    public bool WasSafeTensors => _owned is not null;

    /// <summary>Loads <paramref name="filePath"/>. <paramref name="recursiveFlatten"/> applies to pickles only; a
    /// converted safetensors already carries the keys its conversion produced.</summary>
    public unsafe void Load(string filePath, bool recursiveFlatten = false)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!IsSafeTensors(filePath))
        {
            _pickle = new PytorchPickleLoader();
            _pickle.Load(filePath, recursiveFlatten);
            return;
        }
        using SafeTensorsLoader loader = new();
        loader.Load(filePath);
        Dictionary<string, Tensor> owned = new(StringComparer.Ordinal);
        foreach ((string name, Tensor view) in loader.GetAllTensors())
        {
            Tensor copy = new(view.Shape, view.DType);
            long bytes = Tensor.ComputeByteSize(view.Shape, view.DType);
            Buffer.MemoryCopy(view.DataPointer, copy.DataPointer, bytes, bytes);
            owned[name] = copy;
        }
        _owned = owned;
    }

    /// <summary>Every tensor in the file, keyed as <see cref="PytorchPickleLoader.GetAllTensors"/> keys them.</summary>
    public Dictionary<string, Tensor> GetAllTensors() =>
        _owned is not null ? new Dictionary<string, Tensor>(_owned, StringComparer.Ordinal)
        : _pickle?.GetAllTensors() ?? throw new InvalidOperationException("Load a checkpoint first.");

    /// <summary>True when the file starts with a safetensors header: a u64 length followed by a JSON object.</summary>
    public static bool IsSafeTensors(string filePath)
    {
        using FileStream stream = File.OpenRead(filePath);
        Span<byte> head = stackalloc byte[9];
        if (stream.Read(head) < 9)
        {
            return false;
        }
        long length = BitConverter.ToInt64(head);
        return head[8] == (byte)'{' && length > 1 && length < stream.Length;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        _pickle?.Dispose();
        if (_owned is not null)
        {
            foreach (Tensor tensor in _owned.Values)
            {
                tensor.Dispose();
            }
            _owned = null;
        }
    }
}
