using SharpInference.Core.Backends;
using SharpInference.Core.Exceptions;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;

namespace SharpInference.ModelHandler.Lora;

/// <summary>Composes one or more LoRAs into a single weight-space delta and merges it into a model's weight dictionary. Use one stack per model component (UNet / Transformer / ClipL / ClipG) — each ApplyTo call walks the stacked LoRAs and produces freshly-allocated owned tensors that replace the borrowed mmap entries in the dictionary. The stack owns those new tensors; dispose the stack only after the model is no longer used.</summary>
public sealed class LoraStack : IDisposable
{
    private readonly List<Entry> _entries = [];
    private readonly List<Tensor> _ownedMerged = [];
    private readonly List<LoraFile> _ownedFiles = [];
    private int _disposed;

    /// <summary>Adds a LoRA to the stack. The LoraFile remains owned by the caller — keep it alive at least until ApplyTo has been called for every target component.</summary>
    public void Add(LoraFile file, float strength = 1.0f)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(file);
        _entries.Add(new Entry(file, strength));
    }

    /// <summary>Opens a LoRA safetensors file and adds it to the stack in one call. The stack takes ownership of the loaded file and disposes it when the stack itself is disposed.</summary>
    public void AddFromPath(string filePath, float strength = 1.0f)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        LoraFile file = LoraFile.Load(filePath);
        _ownedFiles.Add(file);
        _entries.Add(new Entry(file, strength));
    }

    /// <summary>Convenience entry point that applies the stack to multiple component weight dictionaries in one call. Pass null for components the model does not have (e.g., a Flux pipeline has no clipG). Returns the total number of weights modified across all components.</summary>
    public int ApplyToWeights(
        IBackend backend,
        IDictionary<string, Tensor>? unetWeights = null,
        IDictionary<string, Tensor>? transformerWeights = null,
        IDictionary<string, Tensor>? clipLWeights = null,
        IDictionary<string, Tensor>? clipGWeights = null)
    {
        int total = 0;
        if (unetWeights is not null) total += ApplyTo(unetWeights, LoraTarget.UNet, backend);
        if (transformerWeights is not null) total += ApplyTo(transformerWeights, LoraTarget.Transformer, backend);
        if (clipLWeights is not null) total += ApplyTo(clipLWeights, LoraTarget.ClipL, backend);
        if (clipGWeights is not null) total += ApplyTo(clipGWeights, LoraTarget.ClipG, backend);
        return total;
    }

    /// <summary>Merges every stacked LoRA layer matching the given target into the weight dictionary. Replaces affected entries with freshly-allocated owned tensors. Returns the number of weights modified.</summary>
    public int ApplyTo(IDictionary<string, Tensor> weights, LoraTarget target, IBackend backend)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(backend);

        Dictionary<string, List<(LoraLayer layer, float strength)>> grouped = [];
        foreach (Entry entry in _entries)
        {
            foreach (LoraLayer layer in entry.File.Layers)
            {
                if (layer.Target != target) continue;
                if (!grouped.TryGetValue(layer.TargetKey, out List<(LoraLayer, float)>? list))
                {
                    list = [];
                    grouped[layer.TargetKey] = list;
                }
                list.Add((layer, entry.Strength));
            }
        }

        int merged = 0;
        foreach ((string canonicalKey, List<(LoraLayer layer, float strength)> deltas) in grouped)
        {
            if (!weights.TryGetValue(canonicalKey, out Tensor? baseW))
            {
                Logs.Warning($"LoRA target '{canonicalKey}' not present in {target} weights; skipping.");
                continue;
            }
            if (Math.Abs(baseW.Fp8ScaleFactor - 1.0f) > 1e-6f)
            {
                throw new SharpInferenceException(
                    $"LoRA cannot merge into FP8-quantized weight '{canonicalKey}' (Fp8ScaleFactor={baseW.Fp8ScaleFactor}). " +
                    "Cast checkpoint weights to F16 before applying LoRA.");
            }

            DType originalDtype = baseW.DType;
            Tensor accumF32 = baseW.CastTo(DType.F32); // owned copy, will be mutated in place
            try
            {
                foreach ((LoraLayer layer, float strength) in deltas)
                {
                    AccumulateDelta(backend, accumF32, layer, strength);
                }

                Tensor finalTensor = originalDtype == DType.F32 ? accumF32 : accumF32.CastTo(originalDtype);
                if (!ReferenceEquals(finalTensor, accumF32))
                {
                    accumF32.Dispose();
                }
                _ownedMerged.Add(finalTensor);
                weights[canonicalKey] = finalTensor;
                merged++;
            }
            catch
            {
                accumF32.Dispose();
                throw;
            }
        }

        if (merged > 0)
        {
            Logs.Info($"Merged {merged} LoRA-targeted weights into {target}.");
        }
        return merged;
    }

    private static void AccumulateDelta(IBackend backend, Tensor accumF32, LoraLayer layer, float strength)
    {
        Tensor upF32 = layer.LoraUp.DType == DType.F32 ? layer.LoraUp.CastTo(DType.F32) : layer.LoraUp.CastTo(DType.F32);
        Tensor downF32 = layer.LoraDown.DType == DType.F32 ? layer.LoraDown.CastTo(DType.F32) : layer.LoraDown.CastTo(DType.F32);
        Tensor delta = new Tensor(new TensorShape(upF32.Shape[0], downF32.Shape[1]), DType.F32);
        try
        {
            backend.MatMul(delta, upF32, downF32);
            float scale = strength * (layer.Alpha / layer.Rank);
            backend.Scale(delta, delta, scale);
            backend.Add(accumF32, accumF32, delta);
        }
        finally
        {
            delta.Dispose();
            upF32.Dispose();
            downF32.Dispose();
        }
    }

    /// <summary>Disposes every merged tensor allocated by ApplyTo and every LoraFile loaded via AddFromPath. Only call this after the model that consumed the merged weights is no longer in use.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _ownedMerged)
        {
            t.Dispose();
        }
        _ownedMerged.Clear();
        foreach (LoraFile f in _ownedFiles)
        {
            f.Dispose();
        }
        _ownedFiles.Clear();
    }

    private readonly record struct Entry(LoraFile File, float Strength);
}
