using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>The F32 weights of one TRELLIS.2 or Pixal3D flow checkpoint, checked against its <see cref="Trellis2FlowConfig"/>.
/// TRELLIS.2 tensors are BF16 and Pixal3D's F32; each is widened once at load, which is exact. Owns the safetensors mapping and
/// every widened copy.</summary>
public sealed class Trellis2FlowWeights : IDisposable
{
    /// <summary>The dense flow's complex RoPE buffer that upstream saves with the weights; <see cref="Trellis2RopeTable"/> rebuilds it.</summary>
    private const string RopePhasesKey = "rope_phases";

    private readonly List<IDisposable> _loaders;
    private readonly List<Tensor> _owned;
    private bool _disposed;

    private Trellis2FlowWeights(Trellis2FlowConfig config, Dictionary<string, Tensor> tensors, List<IDisposable> loaders, List<Tensor> owned)
    {
        Config = config;
        Tensors = tensors;
        _loaders = loaders;
        _owned = owned;
    }

    public Trellis2FlowConfig Config { get; }

    /// <summary>The tensors by upstream checkpoint key, all F32.</summary>
    public IReadOnlyDictionary<string, Tensor> Tensors { get; }

    /// <summary>Opens <paramref name="safetensorsPath"/> with the architecture in its <c>.json</c> sidecar and checks every tensor.</summary>
    /// <exception cref="FileNotFoundException">The checkpoint or its sidecar is missing.</exception>
    /// <exception cref="HartsyInferenceException">A tensor is missing, mis-shaped, or not in the architecture.</exception>
    /// <param name="linearDType">What the transformer blocks' linear weights (the 1.3 B parameters) are held as: F32 (exact widening of the checkpoint),
    /// F16 or BF16 (half the memory; the GEMM then runs in 16 bits with F32 accumulation). Null follows <see cref="EngineKnobs.FlowF16Weights"/>:
    /// F16 when set, otherwise F32. Norm gains, biases, modulation and the small embedder and output layers stay F32.</param>
    public static Trellis2FlowWeights Load(string safetensorsPath, DType? linearDType = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(safetensorsPath);
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"TRELLIS.2 flow checkpoint not found: '{safetensorsPath}'.", safetensorsPath);
        string sidecar = Path.ChangeExtension(safetensorsPath, ".json");
        if (!File.Exists(sidecar))
            throw new FileNotFoundException($"TRELLIS.2 flow config not found beside the checkpoint: '{sidecar}'.", sidecar);
        Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(File.ReadAllText(sidecar));

        SafeTensorsLoader loader = new();
        List<Tensor> owned = [];
        try
        {
            loader.Load(safetensorsPath);
            return FromTensors(config, ReadTensors(loader), [loader], owned, linearDType ?? (EngineKnobs.FlowF16Weights.Value ? DType.F16 : DType.F32));
        }
        catch
        {
            foreach (Tensor t in owned)
                t.Dispose();
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Checks and widens already-loaded <paramref name="tensors"/>; <paramref name="loaders"/> are released with the result.</summary>
    public static Trellis2FlowWeights FromTensors(
        Trellis2FlowConfig config, IReadOnlyDictionary<string, Tensor> tensors, IEnumerable<IDisposable>? loaders = null,
        DType? linearDType = null)
        => FromTensors(config, tensors, [.. loaders ?? []], [], linearDType ?? DType.F32);

    private static Trellis2FlowWeights FromTensors(
        Trellis2FlowConfig config, IReadOnlyDictionary<string, Tensor> tensors, List<IDisposable> loaders, List<Tensor> owned,
        DType linearDType)
    {
        if (linearDType != DType.F32 && linearDType != DType.F16 && linearDType != DType.BF16)
            throw new ArgumentOutOfRangeException(nameof(linearDType), linearDType, "Flow linear weights are held as F32, F16 or BF16.");
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);
        IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs = Trellis2FlowKeyMap.Specs(config);
        HashSet<string> expected = new(specs.Select(s => s.CheckpointKey), StringComparer.Ordinal);
        foreach (string key in tensors.Keys)
        {
            if (!expected.Contains(key))
                throw new HartsyInferenceException($"TRELLIS.2 flow checkpoint holds '{key}', which the architecture does not use.");
        }

        Dictionary<string, Tensor> resolved = ShapeVaeWeights.Resolve(specs, tensors, owned, "TRELLIS.2 flow", linearDType, IsBlockLinear);
        return new Trellis2FlowWeights(config, resolved, loaders, owned);
    }

    // The transformer blocks' linear weights: the matrices, not the rank-1 norm gains.
    private static bool IsBlockLinear(string key, Tensor t) =>
        key.StartsWith("blocks.", StringComparison.Ordinal) && key.EndsWith(".weight", StringComparison.Ordinal) && t.Shape.Rank == 2;

    private static Dictionary<string, Tensor> ReadTensors(SafeTensorsLoader loader)
    {
        Dictionary<string, Tensor> tensors = new(loader.Descriptors.Count);
        foreach (KeyValuePair<string, SafeTensorDescriptor> entry in loader.Descriptors)
        {
            if (entry.Key == RopePhasesKey && entry.Value.DType.IsComplex)
                continue;
            tensors[entry.Key] = loader.GetTensor(entry.Key);
        }

        return tensors;
    }

    /// <summary>The tensor <paramref name="key"/>, which <see cref="Load"/> has already shape-checked.</summary>
    internal Tensor this[string key] => Tensors[key];

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        foreach (Tensor t in _owned)
            t.Dispose();
        _owned.Clear();
        foreach (IDisposable loader in _loaders)
            loader.Dispose();
        _loaders.Clear();
    }
}
