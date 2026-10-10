using System.Text.Json;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41AttentionTests
{
    private static readonly JsonElement Fx = JsonDocument.Parse(File.ReadAllText(Path.Combine(
        RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", "attention_stack.json"))).RootElement;

    private static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static DeepSeekV41Attention BuildLayer(CpuBackend cpu, int layer, out DeepSeekV41AttentionSettings settings, bool quantizeLatents = true)
    {
        JsonElement cfg = Fx.GetProperty("config");
        int ratio = Ints(cfg.GetProperty("compress_ratios"))[layer];
        bool kvSource = Ints(cfg.GetProperty("kv_source_layers")).Contains(layer);
        bool indexSource = Ints(cfg.GetProperty("index_source_layers")).Contains(layer);
        int candidateLayer = cfg.GetProperty("candidate_source_layer").GetInt32();
        settings = new DeepSeekV41AttentionSettings(cfg.GetProperty("dim").GetInt32(), cfg.GetProperty("n_heads").GetInt32(),
            cfg.GetProperty("head_dim").GetInt32(), cfg.GetProperty("rope_head_dim").GetInt32(), cfg.GetProperty("q_lora_rank").GetInt32(),
            cfg.GetProperty("o_groups").GetInt32(), cfg.GetProperty("o_lora_rank").GetInt32(), cfg.GetProperty("window_size").GetInt32(),
            ratio, kvSource && ratio > 0, indexSource && ratio > 0, layer == candidateLayer, candidateLayer >= 0 && candidateLayer < layer,
            cfg.GetProperty("index_n_heads").GetInt32(), cfg.GetProperty("index_head_dim").GetInt32(), cfg.GetProperty("index_topk").GetInt32(),
            cfg.GetProperty("candidate_topk_blocks").GetInt32(), cfg.GetProperty("candidate_block_size").GetInt32(),
            (float)cfg.GetProperty("norm_eps").GetDouble(), quantizeLatents);

        JsonElement p = Fx.GetProperty("layers")[layer];
        float[] P(string name) => Floats(p.GetProperty(name));
        DeepSeekV41CompressorWeights? compressor = settings.IsKvSource
            ? new(P("compressor.wkv.weight"), ratio > 1 ? P("compressor.wgate.weight") : null, P("compressor.norm.weight")) : null;
        DeepSeekV41IndexerWeights? indexer = settings.IsIndexSource
            ? new(P("indexer.wq_b.weight"), P("indexer.weights_proj.weight"),
                p.TryGetProperty("indexer.wk.weight", out _) ? P("indexer.wk.weight") : null,
                p.TryGetProperty("indexer.k_norm.weight", out _) ? P("indexer.k_norm.weight") : null) : null;
        DeepSeekV41AttentionWeights weights = new(P("wq_a.weight"), P("q_norm.weight"), P("wq_b.weight"), P("wkv.weight"), P("kv_norm.weight"),
            P("wo_a.weight"), P("wo_b.weight"), P("attn_sink"), compressor, indexer);

        int rd = settings.RopeDim, length = cfg.GetProperty("max_seq_len").GetInt32();
        DeepSeekV41RopeTable rope = ratio > 0
            ? DeepSeekV41RopeTable.Build(rd, length, cfg.GetProperty("compress_rope_theta").GetDouble(),
                new DeepSeekV41RopeScaling(cfg.GetProperty("rope_factor").GetDouble(), 32, 1, cfg.GetProperty("original_seq_len").GetInt32()))
            : DeepSeekV41RopeTable.Build(rd, length, cfg.GetProperty("rope_theta").GetDouble(), null);
        return new DeepSeekV41Attention(cpu, settings, weights, rope);
    }

    // The fixture runs upstream with an exact-softmax sparse_attn, so what remains is float32 accumulation order; the real kernel's bf16 probabilities are not modelled.
    private const float Tolerance = 1e-3f;

    [Fact]
    public void Layer_Stack_Matches_Upstream_Through_Prefill_And_Decode()
    {
        int layerCount = Fx.GetProperty("layers").GetArrayLength();
        using CpuBackend cpu = new();
        DeepSeekV41Attention[] layers = new DeepSeekV41Attention[layerCount];
        DeepSeekV41AttentionState[] states = new DeepSeekV41AttentionState[layerCount];
        for (int i = 0; i < layerCount; i++)
        {
            layers[i] = BuildLayer(cpu, i, out DeepSeekV41AttentionSettings settings);
            states[i] = new DeepSeekV41AttentionState(settings, 64);
        }

        // upstream's shared slots persist across forward passes: a step that completes no group reuses the previous step's index keys
        DeepSeekV41SharedAttention shared = new();
        int stepNo = 0;
        foreach (JsonElement step in Fx.GetProperty("steps").EnumerateArray())
        {
            int start = step.GetProperty("start").GetInt32(), len = step.GetProperty("len").GetInt32();
            for (int i = 0; i < layerCount; i++)
            {
                float[] x = Floats(step.GetProperty("x")[i]), expected = Floats(step.GetProperty("y")[i]), y = new float[x.Length];
                layers[i].Forward(x, len, start, states[i], shared, y);
                // the chosen indices are exact, so a wrong selection fails here without relying on the output tolerance
                JsonElement picks = step.GetProperty("topk")[i];
                if (picks.ValueKind != JsonValueKind.Null)
                    Assert.True(Ints(picks).SequenceEqual(shared.Topk!), $"step {stepNo} layer {i} indices: [{string.Join(",", Ints(picks))}] vs [{string.Join(",", shared.Topk!)}]");
                for (int j = 0; j < y.Length; j++)
                    Assert.True(Math.Abs(expected[j] - y[j]) <= Tolerance * Math.Max(1f, Math.Abs(expected[j])),
                        $"step {stepNo} (start {start}) layer {i}[{j}]: {expected[j]} vs {y[j]}");
            }
            stepNo++;
        }
        Assert.True(stepNo > 1);
    }

    [Fact]
    public void A_Multi_Token_Chunk_After_Position_Zero_Is_Refused()
    {
        using CpuBackend cpu = new();
        DeepSeekV41Attention layer = BuildLayer(cpu, 0, out DeepSeekV41AttentionSettings settings);
        float[] x = new float[2 * settings.Dim];
        Assert.Throws<NotSupportedException>(() =>
            layer.Forward(x, 2, 5, new DeepSeekV41AttentionState(settings, 64), new DeepSeekV41SharedAttention(), new float[x.Length]));
    }

    [Fact]
    public void A_State_Reused_For_A_New_Sequence_Gives_The_Same_Result_As_A_Fresh_One()
    {
        using CpuBackend cpu = new();
        DeepSeekV41Attention layer = BuildLayer(cpu, 1, out DeepSeekV41AttentionSettings settings);
        JsonElement first = Fx.GetProperty("steps")[0], second = Fx.GetProperty("steps")[1];
        float[] x = Floats(first.GetProperty("x")[1]), fresh = new float[x.Length], reused = new float[x.Length];

        layer.Forward(x, first.GetProperty("len").GetInt32(), 0, new DeepSeekV41AttentionState(settings, 64), new DeepSeekV41SharedAttention(), fresh);

        DeepSeekV41AttentionState dirty = new(settings, 64);
        DeepSeekV41SharedAttention shared = new();
        layer.Forward(x, first.GetProperty("len").GetInt32(), 0, dirty, shared, new float[x.Length]);
        layer.Forward(Floats(second.GetProperty("x")[1]), 1, 11, dirty, shared, new float[settings.Dim]);
        layer.Forward(x, first.GetProperty("len").GetInt32(), 0, dirty, shared, reused);

        Assert.Equal(fresh, reused);
    }

    [Fact]
    public void A_Cache_Too_Small_For_The_Prefill_Is_Refused_Before_Any_State_Changes()
    {
        using CpuBackend cpu = new();
        DeepSeekV41Attention layer = BuildLayer(cpu, 1, out DeepSeekV41AttentionSettings settings);
        DeepSeekV41AttentionState tiny = new(settings, 4);
        float[] x = Floats(Fx.GetProperty("steps")[0].GetProperty("x")[1]);
        Assert.Throws<InvalidOperationException>(() => layer.Forward(x, 11, 0, tiny, new DeepSeekV41SharedAttention(), new float[x.Length]));
        Assert.All(tiny.Window, v => Assert.Equal(0f, v));
        Assert.All(tiny.CompressKv!, v => Assert.Equal(0f, v));
    }
}
