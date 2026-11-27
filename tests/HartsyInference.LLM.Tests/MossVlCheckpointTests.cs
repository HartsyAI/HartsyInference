using System.Text.Json;
using HartsyInference.LLM.Models.MossVl;

namespace HartsyInference.LLM.Tests;

[Trait("Category", "Integration")]
public sealed class MossVlCheckpointTests
{
    private const string IndexVariable = "HARTSY_MOSS_VL_INDEX";

    /// <summary>The tensor names a MOSS-VL checkpoint must hold, derived from its config: per-layer text decoder (self-attention or gated cross-attention), ViT blocks, merger and embeddings.</summary>
    internal static HashSet<string> ExpectedKeys(MossVlConfig config)
    {
        HashSet<string> keys = new(StringComparer.Ordinal);
        const string lm = "model.language_model";
        string[] projections = ["q_proj", "k_proj", "v_proj", "o_proj"];
        for (int i = 0; i < config.Text.NumLayers; i++)
        {
            string p = $"{lm}.layers.{i}";
            keys.Add($"{p}.input_layernorm.weight");
            keys.Add($"{p}.post_attention_layernorm.weight");
            keys.Add($"{p}.mlp.gate_proj.weight");
            keys.Add($"{p}.mlp.up_proj.weight");
            keys.Add($"{p}.mlp.down_proj.weight");
            string attn = config.Text.IsCrossAttnLayer(i) ? "cross_attn" : "self_attn";
            foreach (string proj in projections) keys.Add($"{p}.{attn}.{proj}.weight");
            keys.Add($"{p}.{attn}.q_norm.weight");
            keys.Add($"{p}.{attn}.k_norm.weight");
            if (config.Text.IsCrossAttnLayer(i))
            {
                keys.Add($"{p}.cross_attn_attn_gate");
                keys.Add($"{p}.cross_attn_mlp_gate");
            }
        }
        keys.Add($"{lm}.embed_tokens.weight");
        keys.Add($"{lm}.norm.weight");
        keys.Add("lm_head.weight");
        keys.Add("model.separator_token");

        keys.Add("model.visual.patch_embed.proj.weight");
        keys.Add("model.visual.patch_embed.proj.bias");
        keys.Add("model.visual.pos_embed.weight");
        for (int b = 0; b < config.Vision.Depth; b++)
        {
            string p = $"model.visual.blocks.{b}";
            foreach (string leaf in new[] { "norm1", "norm2", "attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2" })
            {
                keys.Add($"{p}.{leaf}.weight");
                keys.Add($"{p}.{leaf}.bias");
            }
        }
        // The merger reads the last block plus one feature stream per deepstack tap.
        for (int n = 0; n <= config.Vision.DeepstackIndexes.Count; n++)
        {
            keys.Add($"model.visual.merger.norms.{n}.weight");
            keys.Add($"model.visual.merger.norms.{n}.bias");
        }
        foreach (string leaf in new[] { "linear_fc1", "linear_fc2" })
        {
            keys.Add($"model.visual.merger.{leaf}.weight");
            keys.Add($"model.visual.merger.{leaf}.bias");
        }
        return keys;
    }

    [RealWeightsFact(IndexVariable)]
    public void CheckpointIndexMatchesConfigLayout()
    {
        string indexPath = Environment.GetEnvironmentVariable(IndexVariable)!;
        string root = Path.GetDirectoryName(indexPath)!;
        MossVlConfig config;
        using (JsonDocument cfgDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "config.json"))))
            config = MossVlConfig.FromJson(cfgDoc.RootElement);

        using JsonDocument index = JsonDocument.Parse(File.ReadAllText(indexPath));
        HashSet<string> actual = index.RootElement.GetProperty("weight_map").EnumerateObject().Select(p => p.Name).ToHashSet(StringComparer.Ordinal);
        HashSet<string> expected = ExpectedKeys(config);

        Assert.Equal(895, actual.Count);
        Assert.Empty(expected.Except(actual));
        Assert.Empty(actual.Except(expected));
    }
}
