using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ResembleEnhance;

/// <summary>Checkpoint accessor for the resemble-enhance DeepSpeed state dict that records every key it hands
/// out, so the pipeline can enforce a strict zero-missing/zero-unexpected load: a missing key throws immediately
/// with its name, and <see cref="UnconsumedKeys"/> lists anything the checkpoint carried that no module read.</summary>
public sealed class ResembleWeightReader
{
    private readonly IReadOnlyDictionary<string, Tensor> _weights;
    private readonly HashSet<string> _consumed = new HashSet<string>(StringComparer.Ordinal);

    public ResembleWeightReader(IReadOnlyDictionary<string, Tensor> weights)
    {
        _weights = weights ?? throw new ArgumentNullException(nameof(weights));
    }

    /// <summary>Fetches a required tensor as F32 (the checkpoint stores F16) and marks the key consumed.</summary>
    public Tensor F32(string key)
    {
        if (!_weights.TryGetValue(key, out Tensor? t))
        {
            throw new KeyNotFoundException($"Resemble-enhance checkpoint is missing required tensor '{key}'.");
        }
        _consumed.Add(key);
        return WhisperOps.EnsureF32(t);
    }

    /// <summary>Composes a legacy <c>weight_g</c>/<c>weight_v</c> weight-norm pair under <paramref name="prefix"/>
    /// into the dense F32 weight and marks both keys consumed.</summary>
    public Tensor WeightNormF32(string prefix)
    {
        Tensor g = F32($"{prefix}.weight_g");
        Tensor v = F32($"{prefix}.weight_v");
        Tensor composed = WeightNorm.Compose(g, v);
        g.Dispose();
        v.Dispose();
        return composed;
    }

    /// <summary>Reads a single-element tensor as a float scalar and marks the key consumed.</summary>
    public float ScalarF32(string key)
    {
        Tensor t = F32(key);
        if (t.ElementCount != 1)
        {
            throw new InvalidDataException($"Expected scalar tensor for '{key}', got {t.ElementCount} elements.");
        }
        unsafe
        {
            float value = *(float*)t.DataPointer;
            t.Dispose();
            return value;
        }
    }

    /// <summary>Marks checkpoint keys that exist but play no role at inference (training-only heads, recomputed
    /// DSP buffers) as accounted for, so the strict check stays honest about what was deliberately skipped.</summary>
    public void MarkUnused(params string[] keys)
    {
        foreach (string key in keys)
        {
            if (_weights.ContainsKey(key))
            {
                _consumed.Add(key);
            }
        }
    }

    /// <summary>Keys present in the checkpoint that nothing consumed — must be empty after a full load.</summary>
    public IReadOnlyList<string> UnconsumedKeys()
    {
        List<string> extra = [];
        foreach (string key in _weights.Keys)
        {
            if (!_consumed.Contains(key))
            {
                extra.Add(key);
            }
        }
        extra.Sort(StringComparer.Ordinal);
        return extra;
    }
}
