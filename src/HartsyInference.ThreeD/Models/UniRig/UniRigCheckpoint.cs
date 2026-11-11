using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// The tensors of a UniRig Lightning <c>.ckpt</c>, read with the safe-subset pickle loader (nothing is executed) and keyed
/// without the <c>state_dict.model.</c> envelope. Owns the tensors; dispose it after the modules built from it.
/// </summary>
public sealed class UniRigCheckpoint : IDisposable
{
    private const string EnvelopePrefix = "state_dict.model.";

    private readonly PytorchPickleLoader _loader;

    private UniRigCheckpoint(PytorchPickleLoader loader, Dictionary<string, Tensor> tensors)
    {
        _loader = loader;
        Tensors = tensors;
    }

    /// <summary>Tensors by upstream module path, e.g. <c>mesh_encoder.encoder.input_proj.weight</c>.</summary>
    public IReadOnlyDictionary<string, Tensor> Tensors { get; }

    /// <summary>Reads <paramref name="path"/>, requiring every tensor to be F32 under the <c>state_dict.model.</c> envelope.</summary>
    public static UniRigCheckpoint Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        PytorchPickleLoader loader = new();
        try
        {
            loader.Load(path, recursiveFlatten: true);
            Dictionary<string, Tensor> raw = loader.GetAllTensors();
            Dictionary<string, Tensor> tensors = new(raw.Count, StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> pair in raw)
            {
                if (!pair.Key.StartsWith(EnvelopePrefix, StringComparison.Ordinal))
                {
                    continue;
                }

                if (pair.Value.DType != DType.F32)
                {
                    throw new HartsyInferenceException($"UniRig checkpoint tensor '{pair.Key}' is {pair.Value.DType.Name}; only F32 is supported.");
                }

                tensors[pair.Key[EnvelopePrefix.Length..]] = pair.Value;
            }

            if (tensors.Count == 0)
            {
                throw new HartsyInferenceException($"'{path}' holds no '{EnvelopePrefix}*' tensors; it is not a UniRig checkpoint.");
            }

            return new UniRigCheckpoint(loader, tensors);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Returns the named tensor after checking its exact shape.</summary>
    public Tensor Get(string name, params long[] shape) => Get(Tensors, name, shape);

    /// <summary>Returns the named tensor of <paramref name="tensors"/> after checking its exact shape.</summary>
    public static Tensor Get(IReadOnlyDictionary<string, Tensor> tensors, string name, params long[] shape)
    {
        if (!tensors.TryGetValue(name, out Tensor? tensor))
        {
            throw new HartsyInferenceException($"UniRig weight '{name}' is missing.");
        }

        TensorShape expected = new(shape);
        return tensor.Shape.Equals(expected)
            ? tensor
            : throw new HartsyInferenceException($"UniRig weight '{name}' has shape {tensor.Shape}, expected {expected}.");
    }

    public void Dispose() => _loader.Dispose();
}
