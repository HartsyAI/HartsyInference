using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41HostModelLoaderTests
{
    private static readonly JsonElement Fx = JsonDocument.Parse(File.ReadAllText(Path.Combine(
        RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", "model_forward.json"))).RootElement;

    private static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    // The fixture's config in the checkpoint's own config.json shape, on top of the pinned official one.
    private static string ConfigJson()
    {
        JsonElement c = Fx.GetProperty("config");
        return DeepSeekV41Fixtures.OfficialConfig(t =>
        {
            t["vocab_size"] = c.GetProperty("vocab_size").GetInt32();
            t["hidden_size"] = c.GetProperty("dim").GetInt32();
            t["moe_intermediate_size"] = c.GetProperty("moe_inter_dim").GetInt32();
            t["num_hidden_layers"] = c.GetProperty("n_layers").GetInt32();
            t["num_attention_heads"] = c.GetProperty("n_heads").GetInt32();
            t["head_dim"] = c.GetProperty("head_dim").GetInt32();
            t["qk_rope_head_dim"] = c.GetProperty("rope_head_dim").GetInt32();
            t["q_lora_rank"] = c.GetProperty("q_lora_rank").GetInt32();
            t["o_lora_rank"] = c.GetProperty("o_lora_rank").GetInt32();
            t["o_groups"] = c.GetProperty("o_groups").GetInt32();
            t["swiglu_limit"] = c.GetProperty("swiglu_limit").GetDouble();
            t["rms_norm_eps"] = c.GetProperty("norm_eps").GetDouble();
            t["max_position_embeddings"] = c.GetProperty("max_seq_len").GetInt32();
            t["rope_theta"] = c.GetProperty("rope_theta").GetDouble();
            t["rope_scaling"] = new JsonObject
            {
                ["rope_type"] = "yarn", ["factor"] = c.GetProperty("rope_factor").GetDouble(), ["beta_fast"] = 32, ["beta_slow"] = 1,
                ["original_max_position_embeddings"] = c.GetProperty("original_seq_len").GetInt32(),
            };
            t["n_routed_experts"] = c.GetProperty("n_routed_experts").GetInt32();
            t["n_shared_experts"] = 1;
            t["num_experts_per_tok"] = c.GetProperty("n_activated_experts").GetInt32();
            t["routed_scaling_factor"] = c.GetProperty("route_scale").GetDouble();
            t["sliding_window"] = c.GetProperty("window_size").GetInt32();
            t["compress_ratios"] = new JsonArray(Ints(c.GetProperty("compress_ratios")).Select(v => (JsonNode?)v).ToArray());
            t["compress_rope_theta"] = c.GetProperty("compress_rope_theta").GetDouble();
            t["kv_source_layer_ids"] = new JsonArray(Ints(c.GetProperty("kv_source_layers")).Select(v => (JsonNode?)v).ToArray());
            t["index_source_layer_ids"] = new JsonArray(Ints(c.GetProperty("index_source_layers")).Select(v => (JsonNode?)v).ToArray());
            t["index_n_heads"] = c.GetProperty("index_n_heads").GetInt32();
            t["index_head_dim"] = c.GetProperty("index_head_dim").GetInt32();
            t["index_topk"] = c.GetProperty("index_topk").GetInt32();
            t["candidate_source_layer_id"] = c.GetProperty("candidate_source_layer").GetInt32();
            t["candidate_topk_blocks"] = c.GetProperty("candidate_topk_blocks").GetInt32();
            t["candidate_block_size"] = c.GetProperty("candidate_block_size").GetInt32();
            t["hc_mult"] = c.GetProperty("hc_mult").GetInt32();
            t["hc_sinkhorn_iters"] = c.GetProperty("hc_sinkhorn_iters").GetInt32();
            t["hc_eps"] = c.GetProperty("hc_eps").GetDouble();
            t["engram_layer_ids"] = new JsonArray();
            t["engram_num_embeddings"] = new JsonArray();
            t["num_nextn_predict_layers"] = 0;
            t["dspark_block_size"] = 0;
            t["dspark_target_layer_ids"] = new JsonArray();
            t["dspark_n_routed_experts"] = 0;
            t["dspark_num_experts_per_tok"] = 0;
            t["dspark_markov_rank"] = 0;
        });
    }

    // Every fixture parameter as an F32 tensor under its canonical name, in one shard with an index.
    private static void WriteCheckpoint(string directory)
    {
        Dictionary<string, object> header = new();
        List<byte[]> blobs = [];
        long offset = 0;
        JsonElement shapes = Fx.GetProperty("shapes");
        foreach (JsonProperty p in Fx.GetProperty("params").EnumerateObject())
        {
            float[] values = Floats(p.Value);
            byte[] bytes = new byte[values.Length * sizeof(float)];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            header[p.Name] = new Dictionary<string, object>
            {
                ["dtype"] = "F32",
                ["shape"] = Ints(shapes.GetProperty(p.Name)).Select(v => (long)v).ToArray(),
                ["data_offsets"] = new long[] { offset, offset + bytes.Length },
            };
            blobs.Add(bytes);
            offset += bytes.Length;
        }
        const string shard = "model-00001-of-00001.safetensors";
        byte[] json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        using (FileStream stream = new(Path.Combine(directory, shard), FileMode.Create, FileAccess.Write))
        {
            stream.Write(BitConverter.GetBytes((ulong)json.Length));
            stream.Write(json);
            foreach (byte[] blob in blobs) stream.Write(blob);
        }
        File.WriteAllText(Path.Combine(directory, "model.safetensors.index.json"), JsonSerializer.Serialize(new
        {
            metadata = new { total_size = offset },
            weight_map = Fx.GetProperty("params").EnumerateObject().ToDictionary(p => p.Name, _ => shard),
        }));
        File.WriteAllText(Path.Combine(directory, "config.json"), ConfigJson());
    }

    [Fact]
    public void A_Loaded_Checkpoint_Reproduces_Upstream_Hidden_States_And_Logits()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dsv41-load-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            WriteCheckpoint(dir);
            using CpuBackend cpu = new();
            using DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(cpu, dir, new DeepSeekV41LoadOptions(MaxTokens: 64, ExpertCacheCapacity: 3));
            DeepSeekV41HostModel model = loaded.Model;
            DeepSeekV41SequenceState state = model.CreateState(64);
            int stepNo = 0;
            foreach (JsonElement step in Fx.GetProperty("steps").EnumerateArray())
            {
                int[] ids = Ints(step.GetProperty("ids"));
                float[] hidden = new float[ids.Length * model.Dim];
                model.Forward(ids, state, hidden);
                float[] expected = Floats(step.GetProperty("final"));
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(Math.Abs(expected[i] - hidden[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[i])), $"step {stepNo} hidden[{i}]: {expected[i]} vs {hidden[i]}");
                float[] logits = model.Logits(hidden.AsSpan((ids.Length - 1) * model.Dim, model.Dim));
                float[] expectedLogits = Floats(step.GetProperty("logits"));
                for (int i = 0; i < logits.Length; i++)
                    Assert.True(Math.Abs(expectedLogits[i] - logits[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expectedLogits[i])), $"step {stepNo} logits[{i}]");
                stepNo++;
            }
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void Options_Must_Be_Positive()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeepSeekV41LoadOptions(0).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeepSeekV41LoadOptions(8, 0).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeepSeekV41LoadOptions(8, 1, 0).Validate());
    }
}
