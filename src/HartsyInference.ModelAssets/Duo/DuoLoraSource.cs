using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Dcp;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Duo;

/// <summary>The LoRA A and B matrices of a Duo Realization adapter, read one tensor at a time from either the released distributed checkpoint or a safetensors file this tool wrote. Tensors are named
/// <c>module.lora_A.default.weight</c> and <c>module.lora_B.default.weight</c> with the diffusers module names; the optional <c>role_embeddings.weight</c> (unused by the released checkpoints) and anything else is not exposed.</summary>
public sealed class DuoLoraSource : IDisposable
{
    private readonly Dictionary<string, (DType DType, long[] Shape, Func<Tensor> Read)> _tensors = new(StringComparer.Ordinal);
    private readonly SortedSet<string> _modules = new(StringComparer.Ordinal);
    private readonly IDisposable? _owner;

    private DuoLoraSource(IDisposable? owner) => _owner = owner;

    /// <summary>Module names that have both an A and a B matrix, sorted.</summary>
    public IReadOnlyCollection<string> Modules => _modules;

    /// <summary>The F32 tensors the source exposes: key, dtype and shape, so a writer can lay a file out before reading anything.</summary>
    public IEnumerable<(string Key, DType DType, long[] Shape)> Describe()
    {
        foreach (string module in _modules)
        {
            foreach (string key in new[] { module + DuoLoraKeyMap.LoraASuffix, module + DuoLoraKeyMap.LoraBSuffix })
                yield return (key, _tensors[key].DType, _tensors[key].Shape);
        }
    }

    /// <summary>Opens the <c>LoRA</c> tensors of a distributed checkpoint. <paramref name="familyPrefix"/> is the key prefix in front of the module names (<c>models.generator.</c>); null picks the only prefix that carries LoRA tensors.</summary>
    public static DuoLoraSource FromDcp(string folder, string? familyPrefix = null)
    {
        DcpCheckpoint checkpoint = DcpCheckpoint.Open(folder);
        try
        {
            string prefix = familyPrefix ?? PickFamily(checkpoint);
            DuoLoraSource source = new(checkpoint);
            foreach (KeyValuePair<string, DcpTensorEntry> entry in checkpoint.Entries)
            {
                if (!entry.Key.StartsWith(prefix, StringComparison.Ordinal)) continue;
                string key = entry.Key[prefix.Length..];
                if (DuoLoraKeyMap.ModuleOf(key, out _) is null) continue;
                string dcpKey = entry.Key;
                source._tensors[key] = (entry.Value.DType, entry.Value.Shape, () => checkpoint.ReadTensor(dcpKey, DType.F32));
            }
            source.Finish();
            return source;
        }
        catch
        {
            checkpoint.Dispose();
            throw;
        }
    }

    /// <summary>Opens a LoRA safetensors file written by <see cref="DuoRepacker.WriteLoraSafetensors"/>.</summary>
    public static DuoLoraSource FromSafetensors(string path)
    {
        SafeTensorsLoader loader = new();
        try
        {
            loader.Load(path);
            DuoLoraSource source = new(loader);
            foreach (KeyValuePair<string, SafeTensorDescriptor> entry in loader.Descriptors)
            {
                if (DuoLoraKeyMap.ModuleOf(entry.Key, out _) is null) continue;
                string key = entry.Key;
                long[] shape = new long[entry.Value.Shape.Rank];
                for (int d = 0; d < shape.Length; d++) shape[d] = entry.Value.Shape[d];
                source._tensors[key] = (entry.Value.DType, shape, () => loader.GetTensor(key).CastTo(DType.F32));
            }
            source.Finish();
            return source;
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Reads a module's A <c>[rank, in]</c> and B <c>[out, rank]</c> as owned F32 tensors.</summary>
    public (Tensor A, Tensor B) Read(string module)
    {
        ArgumentNullException.ThrowIfNull(module);
        if (!_modules.Contains(module)) throw new KeyNotFoundException($"The LoRA has no module '{module}'.");
        Tensor a = _tensors[module + DuoLoraKeyMap.LoraASuffix].Read();
        try
        {
            return (a, _tensors[module + DuoLoraKeyMap.LoraBSuffix].Read());
        }
        catch
        {
            a.Dispose();
            throw;
        }
    }

    public void Dispose() => _owner?.Dispose();

    private void Finish()
    {
        foreach (string key in _tensors.Keys)
        {
            string module = DuoLoraKeyMap.ModuleOf(key, out bool isA)!;
            string partner = module + (isA ? DuoLoraKeyMap.LoraBSuffix : DuoLoraKeyMap.LoraASuffix);
            if (!_tensors.ContainsKey(partner)) throw new InvalidDataException($"LoRA module '{module}' has only one of its A and B matrices.");
            _modules.Add(module);
        }
        if (_modules.Count == 0) throw new InvalidDataException("The checkpoint holds no LoRA tensors.");
    }

    private static string PickFamily(DcpCheckpoint checkpoint)
    {
        HashSet<string> withLora = new(StringComparer.Ordinal);
        IReadOnlyDictionary<string, int> prefixes = checkpoint.FamilyPrefixes();
        foreach (string key in checkpoint.Entries.Keys)
        {
            if (!key.Contains(".lora_", StringComparison.Ordinal)) continue;
            foreach (string prefix in prefixes.Keys)
                if (key.StartsWith(prefix, StringComparison.Ordinal)) withLora.Add(prefix);
        }
        return withLora.Count == 1
            ? withLora.First()
            : throw new InvalidDataException(
                $"The checkpoint has {withLora.Count} key prefixes with LoRA tensors ({string.Join(", ", withLora)}); name the one to read.");
    }
}
