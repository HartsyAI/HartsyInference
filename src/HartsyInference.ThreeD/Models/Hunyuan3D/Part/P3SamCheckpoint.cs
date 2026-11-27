using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Utonia;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>Reads <c>p3sam/p3sam.safetensors</c> of <c>tencent/Hunyuan3D-Part</c>: 513 F32 tensors, the Sonata backbone under <c>sonata.</c> (5 embedding tensors
/// and 448 stage tensors) and the MLPs <c>mlp</c>, <c>seg_mlp_*</c>, <c>seg_s2_mlp_*</c>, <c>iou_mlp</c> and <c>iou_mlp_out</c>.</summary>
public static class P3SamCheckpoint
{
    private const string SonataPrefix = "sonata.";

    /// <summary>Returns the Sonata weights under the C# names <see cref="UtoniaWeights"/> expects. Throws when one is missing or has the wrong shape;
    /// <c>sonata.embedding.mask_token</c> (a training-time token) is the only checkpoint tensor left unused.</summary>
    public static UtoniaWeights SonataWeights(IReadOnlyDictionary<string, Tensor> state, UtoniaConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(state);
        config ??= UtoniaConfig.Sonata;
        Dictionary<string, Tensor> mapped = new(StringComparer.Ordinal);
        HashSet<string> used = new(StringComparer.Ordinal);
        foreach (UtoniaParameterSpec spec in UtoniaCheckpointKeys.Parameters(config))
        {
            string key = SonataPrefix + spec.TorchKey;
            if (!state.TryGetValue(key, out Tensor? tensor))
                throw new HartsyInferenceException($"P3-SAM checkpoint is missing '{key}'.");
            mapped[spec.CSharpName] = tensor;
            used.Add(key);
        }

        string[] unused = [.. state.Keys.Where(k => k.StartsWith(SonataPrefix, StringComparison.Ordinal) && !used.Contains(k) && k != SonataPrefix + "embedding.mask_token")];
        if (unused.Length > 0)
            throw new HartsyInferenceException($"P3-SAM checkpoint has Sonata tensors the encoder does not use, for example '{unused[0]}' ({unused.Length} in all).");
        return new UtoniaWeights(mapped);
    }
}
