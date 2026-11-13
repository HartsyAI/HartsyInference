using System.Text;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Imaging;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Kimi.Vision;
using HartsyInference.LLM.Tests.Kimi.Vision;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>An image request on a sharded Kimi K3.1 load: the image prefill builds its state over the decoder's ranks, so the spliced prompt runs on a
/// 2-rank decoder, gives finite logits and matches the one-rank logits of the same checkpoint. Runs on virtual CPU ranks; no GPU.</summary>
public sealed class KimiK31ShardedVisionTests : IDisposable
{
    // The media pad sits above the fixture's named specials, as the real tokenizer's <|media_pad|> does.
    private const int MediaPadId = 267, MaxSequence = 32;
    // The same tolerance KimiK31ShardedLoadTests uses for sharded against one-rank logits.
    private const float Tol = 1e-4f;
    private const string TowerShard = "vision.safetensors";
    private const string PreprocessorConfig = "{\"media_proc_cfg\": {\"in_patch_limit\": 65536, \"patch_size\": 2, \"image_mean\": [0.5, 0.5, 0.5], "
        + "\"image_std\": [0.5, 0.5, 0.5], \"merge_kernel_size\": 2, \"fixed_output_tokens\": null, \"patch_limit_on_one_side\": 512}}";

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-kimi-sharded-vision-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void Image_prefill_on_a_two_rank_load_matches_the_one_rank_logits()
    {
        KimiK31Config config = KimiSyntheticCheckpoint.ShardedConfig;
        WriteShardedVisionCheckpoint(_dir, config);
        int[] prompt = [1, 2, MediaPadId, 3];
        RawRgbFrame[] frames = [Frame(seed: 3)];

        float[] reference;
        using (CpuBackend cpu = new())
        using (KimiK31TextModel single = KimiK31TextModel.Load(cpu, _dir, MaxSequence))
        {
            reference = new KimiK31VisionGenerator(single.Generation, single.Tokenizer, single.Vision).PrefillLogits(prompt, frames);
        }

        List<IBackend> backends = [new CpuBackend(), new CpuBackend()];
        try
        {
            using KimiK31TextModel model = KimiK31TextModel.Load(backends, new HostStagedComm(2), _dir, MaxSequence);
            Assert.Equal(2, model.Generation.Decoder.Ranks);

            KimiK31VisionGenerator generator = new(model.Generation, model.Tokenizer, model.Vision);
            float[] logits = generator.PrefillLogits(prompt, frames);

            Assert.Equal(config.Vocab, logits.Length);
            Assert.All(logits, v => Assert.True(float.IsFinite(v)));
            Assert.Equal(reference.Length, logits.Length);
            for (int i = 0; i < reference.Length; i++) Assert.Equal(reference[i], logits[i], Tol);
        }
        finally
        {
            foreach (IBackend backend in backends) backend.Dispose();
        }
    }

    // The sharded text checkpoint, with the tower's shard and its index entries added under the published names and a config that names both.
    private static void WriteShardedVisionCheckpoint(string dir, KimiK31Config config)
    {
        KimiSyntheticCheckpoint.WriteSharded(dir, config, shardFiles: 3, packedExperts: true, bf16Dense: true);
        KimiTiktokenFixture.Write(dir);
        File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"), TokenizerConfig());
        File.WriteAllText(Path.Combine(dir, "preprocessor_config.json"), PreprocessorConfig);
        File.WriteAllText(Path.Combine(dir, "config.json"), CombinedConfigJson(config));

        KimiVisionConfig tower = Tower(config);
        string scratch = dir + "-tower";
        KimiVisionSyntheticCheckpoint.Write(scratch, tower, KimiVisionSyntheticCheckpoint.Values(tower, seed: 31));
        File.Move(Path.Combine(scratch, "model.safetensors"), Path.Combine(dir, TowerShard));
        Directory.Delete(scratch, recursive: true);

        string indexPath = Path.Combine(dir, "model.safetensors.index.json");
        using JsonDocument index = JsonDocument.Parse(File.ReadAllText(indexPath));
        Dictionary<string, string> weightMap = index.RootElement.GetProperty("weight_map").EnumerateObject()
            .ToDictionary(p => p.Name, p => p.Value.GetString()!, StringComparer.Ordinal);
        foreach ((string name, _) in KimiVisionSyntheticCheckpoint.TensorNames(tower)) weightMap[name] = TowerShard;
        File.WriteAllText(indexPath, JsonSerializer.Serialize(new Dictionary<string, object> { ["weight_map"] = weightMap }));
    }

    // The vision tower of the tiny config, widened to the decoder's hidden width.
    private static KimiVisionConfig Tower(KimiK31Config config) => KimiVisionSyntheticCheckpoint.Tiny(layers: 1) with { TextHidden = config.Hidden };

    // The text config with the tower's vision_config and the media placeholder id spliced in at the top level.
    private static string CombinedConfigJson(KimiK31Config config)
    {
        KimiVisionConfig t = Tower(config);
        string vision = $"\"media_placeholder_token_id\": {MediaPadId}, \"vision_config\": {{"
            + $"\"vt_hidden_size\": {t.Hidden}, \"vt_num_attention_heads\": {t.Heads}, \"qkv_hidden_size\": {t.QkvHidden}, "
            + $"\"vt_intermediate_size\": {t.MlpDim}, \"vt_num_hidden_layers\": {t.Layers}, \"patch_size\": {t.PatchSize}, "
            + $"\"merge_kernel_size\": [{t.MergeSize}, {t.MergeSize}], \"init_pos_emb_height\": {t.InitPosGrid}, "
            + $"\"init_pos_emb_width\": {t.InitPosGrid}, \"text_hidden_size\": {t.TextHidden}, \"projector_ln_eps\": 1e-05}}, ";
        return KimiSyntheticCheckpoint.ConfigJson(config).Insert(1, vision);
    }

    // The fixture's named specials, plus the four media literals with the pad at its published id.
    private static string TokenizerConfig()
    {
        StringBuilder sb = new("{\"added_tokens_decoder\": {");
        sb.Append("\"259\": {\"content\": \"[BOS]\", \"special\": true}, \"260\": {\"content\": \"[EOS]\", \"special\": true}, ");
        sb.Append("\"261\": {\"content\": \"<|end_of_msg|>\", \"special\": true}, \"262\": {\"content\": \"<|open|>\", \"special\": true}, ");
        sb.Append("\"263\": {\"content\": \"<|close|>\", \"special\": true}, \"264\": {\"content\": \"<|sep|>\", \"special\": true}, ");
        sb.Append("\"265\": {\"content\": \"<|media_begin|>\", \"special\": true}, ");
        sb.Append("\"266\": {\"content\": \"<|media_content|>\", \"special\": true}, ");
        sb.Append($"\"{MediaPadId}\": {{\"content\": \"<|media_pad|>\", \"special\": true}}, ");
        sb.Append("\"268\": {\"content\": \"<|media_end|>\", \"special\": true}}}");
        return sb.ToString();
    }

    // A deterministic 8x6 RGB frame, patterned by seed.
    private static RawRgbFrame Frame(int seed)
    {
        byte[] rgb = new byte[8 * 6 * 3];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (byte)((i * 31 + seed * 17) % 256);
        return new RawRgbFrame(rgb, 8, 6);
    }
}
