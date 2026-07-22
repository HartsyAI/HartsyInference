using System;
using System.Collections.Generic;
using System.Linq;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads DeepSpeed <c>mp_rank_*_model_states.pt</c> checkpoints into a flat tensor dict ready for a
/// model's <c>LoadWeights</c>. DeepSpeed saves <c>torch.save({"module": model.state_dict(), ...})</c>, and
/// <see cref="PytorchPickleLoader"/> already descends that one-level <c>{"module": {...}}</c> envelope
/// (<c>FlattenStateDict</c>), so keys come out clean (<c>denoiser.net.*</c>, <c>lcfm.cfm.net.*</c>, ...). For
/// variants saved instead as a flat dict whose keys carry a literal <c>module.</c> prefix, this additionally
/// strips that prefix. Reusable for any DeepSpeed-saved model; the per-model <c>LoadWeights</c> does its own
/// bucketing by key prefix (e.g. resemble-enhance's <c>denoiser.net.*</c> / <c>lcfm.ae.*</c> / <c>lcfm.cfm.*</c>
/// / <c>vocoder.*</c>).</summary>
public static class DeepSpeedCheckpointConverter
{
    private const string ModulePrefix = "module.";

    /// <summary>Strips a leading <c>module.</c> from every key when the checkpoint was saved as a flat dict with
    /// that prefix; a no-op when no key is prefixed (the common case, since the loader unwraps the envelope).
    /// Pure and file-free so it is unit-testable.</summary>
    public static Dictionary<string, Tensor> StripModulePrefix(IReadOnlyDictionary<string, Tensor> raw)
    {
        bool anyPrefixed = raw.Keys.Any(k => k.StartsWith(ModulePrefix, StringComparison.Ordinal));
        Dictionary<string, Tensor> outMap = new(raw.Count);
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            string key = anyPrefixed && kv.Key.StartsWith(ModulePrefix, StringComparison.Ordinal)
                ? kv.Key[ModulePrefix.Length..]
                : kv.Key;
            outMap[key] = kv.Value;
        }
        return outMap;
    }

    /// <summary>Loads the DeepSpeed <c>.pt</c> and returns the flat, <c>module.</c>-stripped tensor dict plus the
    /// loader to keep alive (the returned tensors reference its backing storage until disposed).</summary>
    public static (Dictionary<string, Tensor> Weights, IDisposable Loader) Load(string path)
    {
        PytorchPickleLoader loader = new();
        loader.Load(path);
        return (StripModulePrefix(loader.GetAllTensors()), loader);
    }
}
