using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converts a Qwen-Image 2.1 checkpoint into the dictionary <c>QwenImage21Transformer</c> loads. The
/// published <c>Comfy-Org/Qwen-Image-2.1</c> diffusion file already uses the engine's own names
/// (<c>img_in.weight</c>, <c>transformer_blocks.{i}.attn.to_q.weight</c>, <c>modulation.1.weight</c>, …), so this is
/// a re-bucket plus the optional wrapper-prefix strip a community repack may add — there is no key renaming. The one
/// structural difference is the diffusers release (<c>Qwen/Qwen-Image-2.1-Turbo</c>), whose MLP is stored as
/// <c>gate_layer</c> + <c>proj</c>; <see cref="FuseSplitMlps"/> stacks those back into <c>img_mlp.gate_up</c>.
/// <c>img_mlp.gate_up</c> stays fused at runtime: it is one GEMM, and splitting it would double the MLP's bytes and,
/// on the int8 build, require slicing the per-row scale with it.
///
/// <para>Quantized companions (<c>.weight_scale</c> / <c>.comfy_quant</c>) are folded upstream by
/// <c>CheckpointSource</c>; this refuses a dictionary that still carries them rather than dropping them, which is
/// how the int8_convrot build would otherwise serve at scale 1.0.</para></summary>
public sealed class QwenImage21CheckpointConverter
{
    /// <summary>Wrapper prefixes a repack may put in front of the bare diffusion keys.</summary>
    private static readonly string[] WrapperPrefixes = ["model.diffusion_model.", "diffusion_model.", "transformer."];

    /// <summary>The keys ComfyUI's <c>model_detection</c> requires before it will call a file Qwen-Image 2.1, minus
    /// the MLP (which may be fused or split). Used both to detect and to reject a mis-routed file by name.</summary>
    public static readonly string[] SignatureKeys =
        ["txt_in.text_norm.weight", "modulation.1.weight", "transformer_blocks.0.attn.norm_q.weight", "img_in.weight", "proj_out.weight"];

    /// <summary>The converted transformer weights.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Denoiser weights, bare-keyed.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }
    }

    /// <summary>Re-buckets a checkpoint, stripping one wrapper prefix if present.</summary>
    public static ConvertedWeights Convert(IReadOnlyDictionary<string, Tensor> allWeights)
    {
        CheckpointConvertUtils.RequireFoldedCompanions(allWeights, nameof(QwenImage21CheckpointConverter));
        Dictionary<string, Tensor> transformer = new(allWeights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string key = kvp.Key;
            if (key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || key == "scaled_fp8")
            {
                continue;
            }
            foreach (string wrapper in WrapperPrefixes)
            {
                if (key.StartsWith(wrapper, StringComparison.Ordinal))
                {
                    key = key[wrapper.Length..];
                    break;
                }
            }
            transformer[key] = kvp.Value;
        }
        FuseSplitMlps(transformer);
        return new ConvertedWeights { Transformer = transformer };
    }

    /// <summary>Re-fuses diffusers' split SwiGLU into the <c>img_mlp.gate_up</c> the block runs as one GEMM. The
    /// diffusers <c>QwenImage21SwiGLUFeedForward</c> computes <c>out(silu(gate_layer(x)) * proj(x))</c>, so stacking
    /// <c>gate_layer</c> above <c>proj</c> reproduces the fused layout exactly: gate is the first half. The Turbo
    /// checkpoint ships this split form; the Comfy build is already fused and passes through untouched.</summary>
    private static void FuseSplitMlps(Dictionary<string, Tensor> transformer)
    {
        const string GateSuffix = ".img_mlp.gate_layer.weight";
        const string ProjSuffix = ".img_mlp.proj.weight";
        foreach (string gateKey in transformer.Keys.Where(k => k.EndsWith(GateSuffix, StringComparison.Ordinal)).ToArray())
        {
            string prefix = gateKey[..^GateSuffix.Length];
            string projKey = prefix + ProjSuffix;
            if (!transformer.TryGetValue(projKey, out Tensor? proj))
            {
                throw new InvalidOperationException($"'{gateKey}' has no matching '{projKey}': the split MLP is incomplete.");
            }
            transformer[prefix + ".img_mlp.gate_up.weight"] = CheckpointConvertUtils.ConcatRowsHost(transformer[gateKey], proj);
            transformer.Remove(gateKey);
            transformer.Remove(projKey);
        }

        // Anything left in split form (a bias, or a companion that was not folded) would load partially and fail as
        // noise, so refuse it here by name.
        foreach (string key in transformer.Keys)
        {
            if (key.Contains(".img_mlp.gate_layer.", StringComparison.Ordinal) || key.Contains(".img_mlp.proj.", StringComparison.Ordinal))
            {
                throw new InvalidOperationException($"'{key}' is an unfused split-MLP tensor this converter does not handle (only .weight is fused).");
            }
        }
    }

    /// <summary>True when a key set is Qwen-Image 2.1 — ComfyUI's rule from <c>model_detection.py</c>: the five
    /// signature keys plus an <c>img_mlp</c> in either the fused (<c>gate_up</c>) or split (<c>proj</c>) form. It is
    /// deliberately stricter than "has transformer_blocks", because a v1 Qwen-Image file satisfies that too and
    /// would otherwise load into the wrong denoiser and produce noise rather than an error.</summary>
    public static bool MatchesByKeys(IReadOnlyCollection<string> keys)
    {
        HashSet<string> set = new(keys, StringComparer.Ordinal);
        foreach (string signature in SignatureKeys)
        {
            if (!HasKey(set, signature))
            {
                return false;
            }
        }
        return HasKey(set, "transformer_blocks.0.img_mlp.gate_up.weight")
            || HasKey(set, "transformer_blocks.0.img_mlp.proj.weight");
    }

    private static bool HasKey(HashSet<string> keys, string bare)
    {
        if (keys.Contains(bare))
        {
            return true;
        }
        foreach (string wrapper in WrapperPrefixes)
        {
            if (keys.Contains(wrapper + bare))
            {
                return true;
            }
        }
        return false;
    }
}
