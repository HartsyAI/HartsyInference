using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Normalizes a SeedVR2 NaDiT checkpoint for <see cref="HartsyInference.Diffusion.Models.Denoisers"/>' SeedVr2Dit, which consumes the ORIGINAL ByteDance key names verbatim (<c>blocks.{i}.attn.proj_qkv.{vid|txt|all}.weight</c>, …) — a deliberate near-identity mapping to keep the silent-wrongness surface at zero. What this converter actually does: strips bundling prefixes, drops per-block <c>attn.rope.rope.freqs</c> buffers (RoPE frequencies are recomputed exactly at runtime; keeping them would fail the consumed-inventory check), and STRICTLY validates the inventory — every remaining tensor must match the NaDiT structure including the separate-vs-shared (<c>.vid./.txt.</c> vs <c>.all.</c>) block boundary, and every structurally expected tensor must exist. Unknown keys throw; missing keys throw. Dims themselves are validated in SeedVr2Config.Detect (Diffusion side).</summary>
public static class SeedVr2CheckpointConverter
{
    private static readonly string[] BundlePrefixes = ["model.diffusion_model.", "model.", "module."];

    /// <summary>Head/tail tensors every SeedVR2 DiT checkpoint must carry (tail norm/ada are 3B-only and validated conditionally).</summary>
    private static readonly string[] FixedKeys =
    [
        "vid_in.proj.weight", "vid_in.proj.bias",
        "txt_in.weight", "txt_in.bias",
        "emb_in.proj_in.weight", "emb_in.proj_in.bias",
        "emb_in.proj_hid.weight", "emb_in.proj_hid.bias",
        "emb_in.proj_out.weight", "emb_in.proj_out.bias",
        "vid_out.proj.weight", "vid_out.proj.bias",
    ];

    private static readonly string[] TailKeys =
        ["vid_out_norm.weight", "vid_out_ada.out_shift", "vid_out_ada.out_scale"];

    private static readonly string[] BlockSuffixesCommon =
    [
        "ada.{b}.attn_shift", "ada.{b}.attn_scale", "ada.{b}.attn_gate",
        "ada.{b}.mlp_shift", "ada.{b}.mlp_scale", "ada.{b}.mlp_gate",
        "attn.proj_qkv.{b}.weight",
        "attn.proj_out.{b}.weight", "attn.proj_out.{b}.bias",
        "attn.norm_q.{b}.weight", "attn.norm_k.{b}.weight",
    ];

    // 3B: bias-free SwiGLU. 7B: plain biased MLP.
    private static readonly string[] BlockSuffixesSwiGlu =
        ["mlp.{b}.proj_in_gate.weight", "mlp.{b}.proj_in.weight", "mlp.{b}.proj_out.weight"];

    private static readonly string[] BlockSuffixesPlainMlp =
        ["mlp.{b}.proj_in.weight", "mlp.{b}.proj_in.bias", "mlp.{b}.proj_out.weight", "mlp.{b}.proj_out.bias"];

    /// <summary>Strips prefixes, drops recomputable RoPE buffers, and validates the full inventory. Returned tensors reference the source dictionary (keep its loader alive).</summary>
    public static Dictionary<string, Tensor> Convert(Dictionary<string, Tensor> allWeights)
    {
        Dictionary<string, Tensor> weights = StripBundlePrefix(allWeights);

        Dictionary<string, Tensor> result = new(weights.Count);
        foreach ((string key, Tensor value) in weights)
        {
            if (key.EndsWith(".rope.rope.freqs", StringComparison.Ordinal))
                continue;
            result[key] = value;
        }

        ValidateInventory(result);
        return result;
    }

    /// <summary>Loads a safetensors checkpoint and converts it. Caller disposes the returned loader after weights are uploaded/copied.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        Dictionary<string, Tensor> raw = loader.GetAllTensors();
        try
        {
            return (Convert(raw), loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    private static Dictionary<string, Tensor> StripBundlePrefix(Dictionary<string, Tensor> weights)
    {
        foreach (string prefix in BundlePrefixes)
        {
            bool all = true;
            foreach (string key in weights.Keys)
                if (!key.StartsWith(prefix, StringComparison.Ordinal)) { all = false; break; }
            if (!all)
                continue;
            Dictionary<string, Tensor> stripped = new(weights.Count);
            foreach ((string key, Tensor value) in weights)
                stripped[key[prefix.Length..]] = value;
            return stripped;
        }
        return weights;
    }

    private static void ValidateInventory(Dictionary<string, Tensor> weights)
    {
        int numLayers = 0, mmLayers = -1;
        foreach (string key in weights.Keys)
        {
            if (!key.StartsWith("blocks.", StringComparison.Ordinal))
                continue;
            int end = key.IndexOf('.', 7);
            if (end < 0 || !int.TryParse(key.AsSpan(7, end - 7), out int idx))
                throw new HartsyInferenceException($"Unrecognized SeedVR2 block key '{key}'.");
            numLayers = Math.Max(numLayers, idx + 1);
            if (key.Contains(".all.", StringComparison.Ordinal))
                mmLayers = mmLayers < 0 ? idx : Math.Min(mmLayers, idx);
        }
        // No `.all.` keys at all = full MM-DiT (the 7B: every block split) → boundary at numLayers.
        if (mmLayers < 0)
            mmLayers = numLayers;
        if (numLayers == 0 || mmLayers <= 0)
            throw new HartsyInferenceException(
                $"SeedVR2 block layout not recognized: layers={numLayers}, mm boundary={mmLayers}.");

        bool swiglu = weights.ContainsKey("blocks.0.mlp.vid.proj_in_gate.weight")
            || weights.ContainsKey("blocks.0.mlp.all.proj_in_gate.weight");
        string[] blockSuffixes =
            [.. BlockSuffixesCommon, .. swiglu ? BlockSuffixesSwiGlu : BlockSuffixesPlainMlp];

        HashSet<string> expected = new(FixedKeys, StringComparer.Ordinal);
        if (weights.ContainsKey("vid_out_norm.weight"))
            foreach (string key in TailKeys)
                expected.Add(key);
        for (int i = 0; i < numLayers; i++)
            foreach (string suffix in blockSuffixes)
            {
                if (i < mmLayers)
                {
                    expected.Add($"blocks.{i}.{suffix.Replace("{b}", "vid")}");
                    expected.Add($"blocks.{i}.{suffix.Replace("{b}", "txt")}");
                }
                else
                {
                    expected.Add($"blocks.{i}.{suffix.Replace("{b}", "all")}");
                }
            }

        List<string> unknown = weights.Keys.Where(k => !expected.Contains(k)).Take(8).ToList();
        if (unknown.Count > 0)
            throw new HartsyInferenceException(
                $"SeedVR2 checkpoint has {unknown.Count}+ unrecognized tensors, e.g. '{unknown[0]}'.");
        List<string> missing = expected.Where(k => !weights.ContainsKey(k)).Take(8).ToList();
        if (missing.Count > 0)
            throw new HartsyInferenceException(
                $"SeedVR2 checkpoint is missing {missing.Count}+ expected tensors, e.g. '{missing[0]}'.");
    }
}
