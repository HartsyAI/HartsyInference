using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Omni;

/// <summary>The weights of a <c>tencent/Hunyuan3D-Omni</c> folder, read tensor by tensor from the pickles (<c>model/</c>, <c>vae/</c>, <c>cond_encoder/</c> <c>pytorch_model.bin</c>, through the safe subset parser, nothing executed). The DiT file is F32 and 12.2 GB:
/// with <c>halfWeights</c> each matrix is narrowed to F16 as it is read, so the host holds 6.1 GB and never the F32 file. The tensors belong to this object; dispose the models that borrow them first.</summary>
public sealed class OmniCheckpoint : IDisposable
{
    /// <summary>The VAE checkpoint's key prefix (<c>ShapeVAEPL.sal</c>).</summary>
    public const string VaePrefix = "sal";

    private readonly List<Tensor> _owned = [];
    private readonly List<PytorchPickleLoader> _loaders = [];
    private bool _disposed;

    private OmniCheckpoint(Hunyuan3DConfig config) => Config = config;

    /// <summary>The architecture from the folder's configs.</summary>
    public Hunyuan3DConfig Config { get; }

    /// <summary>The DiT tensors under bare keys (<c>blocks.N...</c>, <c>x_embedder...</c>): matrices F16 when loaded with <c>halfWeights</c>, everything else F32.</summary>
    public Dictionary<string, Tensor> Dit { get; } = [];

    /// <summary>The VAE decoder tensors under their <c>sal.</c> keys, F32 (the encoder half of the file is not read).</summary>
    public Dictionary<string, Tensor> Vae { get; } = [];

    /// <summary>The six control-encoder tensors, F32.</summary>
    public Dictionary<string, Tensor> Control { get; } = [];

    /// <summary>Reads the weights of <paramref name="folder"/>.</summary>
    /// <param name="folder">The folder with <c>config.json</c>, <c>model/</c>, <c>vae/</c> and <c>cond_encoder/</c>.</param>
    /// <param name="halfWeights">Narrow the DiT matrices to F16 (for a backend that multiplies F16 weights); the other tensors stay F32.</param>
    /// <param name="onProgress">Called with the number of DiT tensors read so far and the total.</param>
    public static OmniCheckpoint Load(string folder, bool halfWeights, Action<int, int>? onProgress = null)
    {
        OmniCheckpoint checkpoint = new(OmniConfigFile.Read(folder));
        try
        {
            checkpoint.ReadInto(Path.Combine(folder, "cond_encoder", "pytorch_model.bin"), checkpoint.Control, _ => true, _ => null, null);
            checkpoint.ReadInto(Path.Combine(folder, "vae", "pytorch_model.bin"), checkpoint.Vae, IsDecoderKey, _ => null, null);
            checkpoint.ReadInto(Path.Combine(folder, "model", "pytorch_model.bin"), checkpoint.Dit, _ => true,
                descriptor => halfWeights && descriptor.Shape.Rank == 2 ? DType.F16 : null, onProgress);
            return checkpoint;
        }
        catch
        {
            checkpoint.Dispose();
            throw;
        }
    }

    // The decoder reads post_kl, the decoder transformer and the geometry decoder; pre_kl and the 8 encoder layers stay in the file.
    private static bool IsDecoderKey(string key) =>
        key.StartsWith($"{VaePrefix}.post_kl.", StringComparison.Ordinal) || key.StartsWith($"{VaePrefix}.transformer.", StringComparison.Ordinal)
        || key.StartsWith($"{VaePrefix}.geo_decoder.", StringComparison.Ordinal);

    private void ReadInto(string path, Dictionary<string, Tensor> target, Func<string, bool> include, Func<PytorchTensorDescriptor, DType?> convert, Action<int, int>? onProgress)
    {
        if (!RepackedCheckpoint.Exists(path))
            throw new FileNotFoundException($"The Hunyuan3D-Omni folder lacks {Path.GetFileName(Path.GetDirectoryName(path))}/{Path.GetFileName(path)}.", path);
        string repacked = RepackedCheckpoint.Resolve(path);
        if (repacked != path)
        {
            ReadRepackInto(repacked, target, include, convert, onProgress);
            return;
        }
        PytorchPickleLoader loader = new();
        _loaders.Add(loader);
        loader.OpenLazy(path);
        // In storage order: the DiT file holds every tensor as a view of one 12 GB entry, which is then read in one pass.
        string[] names = [.. loader.Descriptors.Values.Where(d => include(d.Name)).OrderBy(d => d.StorageKey, StringComparer.Ordinal).ThenBy(d => d.StorageByteOffset).Select(d => d.Name)];
        if (names.Length == 0)
            throw new InvalidDataException($"'{path}' holds none of the expected tensors.");
        for (int i = 0; i < names.Length; i++)
        {
            Tensor tensor = loader.ReadTensor(names[i], convert(loader.Descriptors[names[i]]));
            _owned.Add(tensor);
            target[names[i]] = tensor;
            onProgress?.Invoke(i + 1, names.Length);
        }

        // Nothing reads the pickles again: release the file.
        loader.Dispose();
        _loaders.Remove(loader);
    }

    // The same tensors under the same names as the pickle, so the caller's filter and conversion apply unchanged; each is copied (or narrowed) out of the mapping as it is read.
    private unsafe void ReadRepackInto(string path, Dictionary<string, Tensor> target, Func<string, bool> include, Func<PytorchTensorDescriptor, DType?> convert, Action<int, int>? onProgress)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        string[] names = [.. loader.Descriptors.Values.Where(d => include(d.Name)).OrderBy(d => d.DataOffset).Select(d => d.Name)];
        if (names.Length == 0)
            throw new InvalidDataException($"'{path}' holds none of the expected tensors.");
        for (int i = 0; i < names.Length; i++)
        {
            SafeTensorDescriptor descriptor = loader.Descriptors[names[i]];
            // The loader does not check a hand-placed file's offsets against its shapes, and the copy below trusts the shape.
            if (descriptor.ByteLength != descriptor.DType.ComputeByteCount(descriptor.Shape.ElementCount))
                throw new InvalidDataException($"'{path}' stores {descriptor.ByteLength} bytes for tensor '{descriptor.Name}', which its shape and dtype do not account for.");
            DType? narrowed = convert(new PytorchTensorDescriptor { Name = descriptor.Name, DType = descriptor.DType, Shape = descriptor.Shape });
            Tensor view = loader.GetTensor(names[i]);
            Tensor owned;
            if (narrowed is null || narrowed == descriptor.DType)
            {
                owned = new Tensor(descriptor.Shape, descriptor.DType);
                long bytes = descriptor.DType.ComputeByteCount(descriptor.Shape.ElementCount);
                Buffer.MemoryCopy((void*)view.DataPointer, (void*)owned.DataPointer, bytes, bytes);
            }
            else
            {
                owned = view.CastTo(narrowed.Value);
            }
            _owned.Add(owned);
            target[names[i]] = owned;
            onProgress?.Invoke(i + 1, names.Length);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        foreach (PytorchPickleLoader loader in _loaders)
            loader.Dispose();
        _loaders.Clear();
        foreach (Tensor tensor in _owned)
            tensor.Dispose();
        _owned.Clear();
        Dit.Clear();
        Vae.Clear();
        Control.Clear();
    }
}
