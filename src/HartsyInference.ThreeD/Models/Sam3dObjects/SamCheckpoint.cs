using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// One SAM 3D weight file, read as a Lightning <c>.ckpt</c> (through the safe-subset pickle reader, which never runs code from the file) or as
/// a <c>.safetensors</c> conversion; the format is chosen by the file's bytes. Owns every tensor until disposed.
/// </summary>
public sealed class SamCheckpoint : IDisposable
{
    private readonly AnyFormatCheckpointLoader _loader;
    private readonly Dictionary<string, Tensor> _tensors;

    private SamCheckpoint(AnyFormatCheckpointLoader loader, Dictionary<string, Tensor> tensors)
    {
        _loader = loader;
        _tensors = tensors;
    }

    /// <summary>True when the file was a safetensors file, false for a pickle <c>.ckpt</c>.</summary>
    public bool WasSafeTensors => _loader.WasSafeTensors;

    /// <summary>Every tensor by its key in the file.</summary>
    public IReadOnlyDictionary<string, Tensor> Tensors => _tensors;

    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    public static SamCheckpoint Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"SAM 3D checkpoint not found: '{path}'.", path);
        AnyFormatCheckpointLoader loader = new();
        try
        {
            loader.Load(path);
            return new SamCheckpoint(loader, loader.GetAllTensors());
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>The tensors of one model by their name inside it; the tensors are borrowed and live until this checkpoint is disposed.</summary>
    /// <exception cref="InvalidOperationException">The section holds no tensor, which means the file is not that kind of checkpoint.</exception>
    public Dictionary<string, Tensor> Section(SamCheckpointSection section)
    {
        Dictionary<string, string> map = SamCheckpointKeys.Select(_tensors.Keys, section);
        if (map.Count == 0)
            throw new InvalidOperationException($"The checkpoint holds no '{section}' tensors ({_tensors.Count} tensors in the file).");
        Dictionary<string, Tensor> result = new(map.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, string> entry in map)
            result[entry.Key] = _tensors[entry.Value];
        return result;
    }

    public void Dispose() => _loader.Dispose();
}
