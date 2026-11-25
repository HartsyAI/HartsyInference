using System.Globalization;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.PyTorch;

/// <summary>Reads the tensors of one named envelope from a torch checkpoint whose root holds several state dicts (a training
/// checkpoint's <c>generator</c>, <c>generator_ema</c> and <c>critic</c>, say), as owned copies, and frees them through the
/// returned owner. The other envelopes are never read.</summary>
public static class PytorchEnvelopeReader
{
    /// <summary>Reads every tensor under <paramref name="envelope"/> whose name continues with <paramref name="wrapperPrefix"/>,
    /// stripping both, and returns them with an owner that frees them.</summary>
    /// <exception cref="InvalidOperationException">The file has no <paramref name="envelope"/> dict, or a tensor under it lacks the prefix.</exception>
    public static (Dictionary<string, Tensor> Weights, IDisposable Owner) Read(string path, string envelope, string wrapperPrefix, DType dtype)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        ArgumentException.ThrowIfNullOrEmpty(envelope);
        ArgumentNullException.ThrowIfNull(wrapperPrefix);

        string head = envelope + ".";
        using PytorchPickleLoader pickle = new PytorchPickleLoader();
        // Recursive flatten gives every tensor its full dotted name ("generator_ema.model._fsdp_wrapped_module.…"), so the
        // envelope can be chosen by name. The default would descend into whichever dict comes first in the pickle.
        pickle.OpenLazy(path, recursiveFlatten: true);

        List<PytorchTensorDescriptor> selected = pickle.Descriptors.Values
            .Where(d => d.Name.StartsWith(head, StringComparison.Ordinal))
            .OrderBy(d => long.TryParse(d.StorageKey, NumberStyles.None, CultureInfo.InvariantCulture, out long n) ? n : long.MaxValue)
            .ThenBy(d => d.StorageKey, StringComparer.Ordinal)
            .ThenBy(d => d.StorageByteOffset)
            .ToList();
        if (selected.Count == 0)
        {
            string[] available = pickle.Descriptors.Keys
                .Select(k => k.Split('.')[0])
                .Distinct(StringComparer.Ordinal)
                .OrderBy(k => k, StringComparer.Ordinal)
                .ToArray();
            throw new InvalidOperationException(
                $"'{path}' has no '{envelope}' dict; its top-level entries are: {string.Join(", ", available)}.");
        }

        string strip = head + wrapperPrefix;
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(selected.Count, StringComparer.Ordinal);
        List<Tensor> owned = new List<Tensor>(selected.Count);
        try
        {
            foreach (PytorchTensorDescriptor descriptor in selected)
            {
                if (!descriptor.Name.StartsWith(strip, StringComparison.Ordinal))
                {
                    throw new InvalidOperationException(
                        $"'{path}' has tensor '{descriptor.Name}' under '{envelope}' but outside the expected '{strip}' prefix.");
                }
                Tensor tensor = pickle.ReadTensor(descriptor.Name, dtype);
                owned.Add(tensor);
                weights[descriptor.Name[strip.Length..]] = tensor;
            }
        }
        catch
        {
            foreach (Tensor tensor in owned)
            {
                tensor.Dispose();
            }
            throw;
        }
        return (weights, new TensorOwner(owned));
    }

    /// <summary>Frees the tensors one <see cref="Read"/> produced. Disposing it twice is harmless.</summary>
    private sealed class TensorOwner(List<Tensor> tensors) : IDisposable
    {
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            foreach (Tensor tensor in tensors)
            {
                tensor.Dispose();
            }
            tensors.Clear();
        }
    }
}
