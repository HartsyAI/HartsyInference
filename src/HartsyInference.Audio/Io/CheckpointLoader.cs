using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Io;

/// <summary>Loads a weight file in either shipping format, picked by extension.</summary>
internal static class CheckpointLoader
{
    /// <summary>Loads <paramref name="path"/> and appends its loader to <paramref name="retain"/> — the returned
    /// tensors borrow the loader's mmap, so it has to outlive them.</summary>
    /// <param name="recursiveFlatten">Pickle only. <c>false</c> flattens just the outer envelope, so a flat
    /// <c>{state_dict, metadata}</c> checkpoint loses the <c>state_dict.</c> prefix instead of keeping it.</param>
    public static Dictionary<string, Tensor> Load(string path, List<IDisposable> retain, bool recursiveFlatten)
    {
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            SafeTensorsLoader loader = new();
            loader.Load(path);
            retain.Add(loader);
            return loader.GetAllTensors();
        }
        PytorchPickleLoader pickle = new();
        pickle.Load(path, recursiveFlatten: recursiveFlatten);
        retain.Add(pickle);
        return pickle.GetAllTensors();
    }
}
