using System.Text;
using System.Text.Json;
using HartsyInference.LLM.Kimi.Vision;
using HartsyInference.LLM.Tests.Kimi.Vision;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>A tiny Kimi K3.1 checkpoint that takes images: the text checkpoint of <see cref="KimiSyntheticCheckpoint"/> with a tiktoken file and no
/// chat template, so the built-in XTML template renders the media placeholder; the four media literals with the pad at its published id; and a
/// MoonViT-V2 tower in its own shard.</summary>
internal static class KimiVisionCheckpoint
{
    // The template's control ids end at <|media_end|>, so the embedding table must cover them.
    internal const int Vocab = 270, MediaPadId = 267, BosId = 259, EosId = 260, PublishedGrid = 4;
    private const string TowerShard = "vision.safetensors";
    private const string PreprocessorConfig = "{\"media_proc_cfg\": {\"in_patch_limit\": 65536, \"patch_size\": 2, \"image_mean\": [0.5, 0.5, 0.5], "
        + "\"image_std\": [0.5, 0.5, 0.5], \"merge_kernel_size\": 2, \"fixed_output_tokens\": null, \"patch_limit_on_one_side\": 512}}";

    /// <summary>The tower of the tiny config, widened to the synthetic decoder's hidden width.</summary>
    internal static KimiVisionConfig Tower() => KimiVisionSyntheticCheckpoint.Tiny(layers: 1) with { TextHidden = KimiSyntheticCheckpoint.Hidden };

    /// <summary>Writes the checkpoint into <paramref name="directory"/>.</summary>
    internal static void Write(string directory)
    {
        KimiSyntheticCheckpoint.Write(directory, vocab: Vocab);
        File.Delete(Path.Combine(directory, "tokenizer.json"));
        File.Delete(Path.Combine(directory, "chat_template.jinja"));
        WriteTiktoken(directory);
        File.WriteAllText(Path.Combine(directory, "tokenizer_config.json"), TokenizerConfig());
        File.WriteAllText(Path.Combine(directory, "config.json"), ConfigJson());
        File.WriteAllText(Path.Combine(directory, "preprocessor_config.json"), PreprocessorConfig);
        AddTower(directory);
    }

    // The text config with the tower's vision_config and the media pad id spliced in, and the byte-level tiktoken's bos and eos ids.
    private static string ConfigJson()
    {
        KimiVisionConfig t = Tower();
        string vision = $"\"media_placeholder_token_id\": {MediaPadId}, \"vision_config\": {{"
            + $"\"vt_hidden_size\": {t.Hidden}, \"vt_num_attention_heads\": {t.Heads}, \"qkv_hidden_size\": {t.QkvHidden}, "
            + $"\"vt_intermediate_size\": {t.MlpDim}, \"vt_num_hidden_layers\": {t.Layers}, \"patch_size\": {t.PatchSize}, "
            + $"\"merge_kernel_size\": [{t.MergeSize}, {t.MergeSize}], \"init_pos_emb_height\": {PublishedGrid}, "
            + $"\"init_pos_emb_width\": {PublishedGrid}, \"text_hidden_size\": {t.TextHidden}, \"projector_ln_eps\": 1e-05}}, ";
        string text = KimiSyntheticCheckpoint.ConfigJson(Vocab);
        string named = text.Replace(
            $"\"bos_token_id\": {KimiSyntheticCheckpoint.BosId}, \"eos_token_id\": {KimiSyntheticCheckpoint.EosId}",
            $"\"bos_token_id\": {BosId}, \"eos_token_id\": {EosId}", StringComparison.Ordinal);
        Assert.NotEqual(text, named);
        return named.Insert(1, vision);
    }

    // Named specials sit right above a 256-byte rank file plus three merges, as in the tiktoken checkpoints of KimiK31TextPathTests.
    private static void WriteTiktoken(string directory)
    {
        List<string> lines = [];
        for (int b = 0; b < 256; b++)
            lines.Add($"{Convert.ToBase64String([(byte)b])} {b}");
        lines.Add($"{Convert.ToBase64String("he"u8)} 256");
        lines.Add($"{Convert.ToBase64String("ll"u8)} 257");
        lines.Add($"{Convert.ToBase64String("hell"u8)} 258");
        File.WriteAllLines(Path.Combine(directory, "tiktoken.model"), lines);
    }

    // The named specials the XTML template and the processor need, the four media literals among them, with the pad at its published id.
    private static string TokenizerConfig()
    {
        StringBuilder sb = new("{\"added_tokens_decoder\": {");
        sb.Append($"\"{BosId}\": {{\"content\": \"[BOS]\", \"special\": true}}, \"{EosId}\": {{\"content\": \"[EOS]\", \"special\": true}}, ");
        sb.Append("\"261\": {\"content\": \"<|end_of_msg|>\", \"special\": true}, \"262\": {\"content\": \"<|open|>\", \"special\": true}, ");
        sb.Append("\"263\": {\"content\": \"<|close|>\", \"special\": true}, \"264\": {\"content\": \"<|sep|>\", \"special\": true}, ");
        sb.Append("\"265\": {\"content\": \"<|media_begin|>\", \"special\": true}, \"266\": {\"content\": \"<|media_content|>\", \"special\": true}, ");
        sb.Append($"\"{MediaPadId}\": {{\"content\": \"<|media_pad|>\", \"special\": true}}, ");
        sb.Append("\"268\": {\"content\": \"<|media_end|>\", \"special\": true}}}");
        return sb.ToString();
    }

    // The tower's shard beside the text shard, under one index that names both.
    private static void AddTower(string directory)
    {
        KimiVisionConfig tower = Tower();
        string scratch = directory + "-tower";
        KimiVisionSyntheticCheckpoint.Write(scratch, tower, KimiVisionSyntheticCheckpoint.Values(tower, seed: 31));
        File.Move(Path.Combine(scratch, "model.safetensors"), Path.Combine(directory, TowerShard));
        Directory.Delete(scratch, recursive: true);

        Dictionary<string, string> weightMap = KimiSyntheticCheckpoint.TensorNames(Vocab)
            .ToDictionary(t => t.Name, _ => "model.safetensors", StringComparer.Ordinal);
        foreach ((string name, _) in KimiVisionSyntheticCheckpoint.TensorNames(tower)) weightMap[name] = TowerShard;
        File.WriteAllText(Path.Combine(directory, "model.safetensors.index.json"),
            JsonSerializer.Serialize(new Dictionary<string, object> { ["weight_map"] = weightMap }));
    }
}
