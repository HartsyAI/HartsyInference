using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>A tiny random Flint run through transformers' own <c>FlintForCausalLM</c> (<c>tests/python-reference/dump_flint_tiny_reference.py</c>) must
/// match <see cref="GenericTransformer"/> loaded through the real GGUF path: the Hugging Face weights are written as a <c>flint</c> GGUF with the tensor names and
/// metadata keys of the official release (fused expert stacks split into gate and up, the router as <c>ffn_gate_inp</c>, sinks as <c>attn_sinks</c>), then
/// loaded by <see cref="GgufModelLoader"/>, <see cref="GgufLanguageModel.PrepareWeights"/> and <see cref="GgufConfigFactory"/>. The fixture keeps every block
/// trait of the release: two dense layers, sigmoid routing with normalized top-k and an ungated shared expert, per-head QK norm, learned sinks, sliding-window RoPE
/// layers (a 5-token window crossed by the 12-token prompt) and full-attention layers without positional encoding.</summary>
public sealed unsafe class FlintTransformersParityTests : IDisposable
{
    private const float Tolerance = 2e-4f;
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Flint");

    private readonly SafeTensorsLoader _hf = new();
    private readonly SafeTensorsLoader _expected = new();
    private readonly List<Tensor> _owned = [];
    private readonly string _gguf = Path.Combine(Path.GetTempPath(), $"flint-tiny-{Guid.NewGuid():N}.gguf");
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly JsonElement _config;
    private readonly JsonDocument _configDocument;
    private readonly CpuBackend _backend = new();

    public FlintTransformersParityTests()
    {
        _hf.Load(Path.Combine(Dir, "flint_tiny.safetensors"));
        _expected.Load(Path.Combine(Dir, "flint_tiny_expected.safetensors"));
        _configDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "flint_tiny_config.json")));
        _config = _configDocument.RootElement;
        WriteGguf();
        _handle = GgufModelLoader.Load(_gguf);
        (Dictionary<string, Tensor> weights, Config) = GgufLanguageModel.PrepareWeights(_handle, lowVramQuant: false, dequantizeToF32: true);
        Transformer = new GenericTransformer(Config);
        Transformer.LoadWeights(weights, "model");
        Ids = _expected.GetTensor("input_ids").AsReadOnlySpan<int>().ToArray();
    }

    private TransformerConfig Config { get; }

    private GenericTransformer Transformer { get; }

    private int[] Ids { get; }

    [Fact]
    public void Prefill_LogitsMatchTransformers()
    {
        using KvCache cache = NewCache();
        using Tensor hidden = Transformer.Forward(_backend, Ids, 0, cache);
        using Tensor logits = Transformer.ProjectLogits(_backend, hidden, Ids.Length);
        AssertRowsMatch(logits.AsReadOnlySpan<float>(), 0, Ids.Length);
    }

    [Fact]
    public void PrefillThenDecode_LogitsMatchTransformers()
    {
        using KvCache cache = NewCache();
        const int prefill = 5;
        using (Tensor hidden = Transformer.Forward(_backend, Ids.AsSpan(0, prefill), 0, cache))
        using (Tensor logits = Transformer.ProjectLogits(_backend, hidden, prefill))
            AssertRowsMatch(logits.AsReadOnlySpan<float>(), 0, prefill);
        for (int pos = prefill; pos < Ids.Length; pos++)
        {
            using Tensor hidden = Transformer.Forward(_backend, Ids.AsSpan(pos, 1), pos, cache);
            using Tensor logits = Transformer.ProjectLogits(_backend, hidden, 1);
            AssertRowsMatch(logits.AsReadOnlySpan<float>(), pos, 1);
        }
    }

    [Fact]
    public void GgufConfig_DescribesTheFlintLayout()
    {
        Assert.Equal("flint", Config.Architecture);
        Assert.Equal(RopeStyle.SplitHalf, Config.Rope);
        Assert.True(Config.AttnSink);
        Assert.True(Config.NoRopeOnGlobalLayers);
        Assert.True(Config.QkNorm);
        string layers = string.Concat(Enumerable.Range(0, Config.NumLayers).Select(i => Config.IsGlobalLayer(i) ? "G" : "L"));
        Assert.Equal("GGLLLGLG", layers);
        MoeConfig moe = Config.Moe!;
        Assert.Equal((8, 3, 16, 24, MoeScoring.Sigmoid, true, 2), (moe.NumExperts, moe.NumExpertsPerTok, moe.MoeIntermediateSize,
            moe.SharedExpertIntermediateSize, moe.Scoring, moe.NormTopKProb, moe.FirstDenseLayers));
        Assert.Equal(5, Config.SlidingWindow);
    }

    public void Dispose()
    {
        _handle.Dispose();
        foreach (Tensor t in _owned) t.Dispose();
        _hf.Dispose();
        _expected.Dispose();
        _configDocument.Dispose();
        _backend.Dispose();
        if (File.Exists(_gguf)) File.Delete(_gguf);
    }

    private KvCache NewCache() => new(Config.NumLayers, 1, Config.NumKvHeads, Config.HeadDim);

    private void AssertRowsMatch(ReadOnlySpan<float> actual, int firstRow, int rows)
    {
        int vocab = Config.VocabSize;
        ReadOnlySpan<float> expected = _expected.GetTensor("logits").AsReadOnlySpan<float>().Slice(firstRow * vocab, rows * vocab);
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        Assert.True(worst < Tolerance, $"max |Δ| = {worst} over rows {firstRow}..{firstRow + rows - 1}");
        Assert.True(expected.ToArray().Max(MathF.Abs) > 0.5f);
    }

    private void WriteGguf()
    {
        int layers = _config.GetProperty("num_hidden_layers").GetInt32();
        int moeInter = _config.GetProperty("moe_intermediate_size").GetInt32();
        string[] types = [.. _config.GetProperty("layer_types").EnumerateArray().Select(e => e.GetString()!)];
        bool[] sliding = [.. types.Select(t => t == "sliding_attention")];
        using GgufWriter w = new(_gguf);
        w.SetMetadata("general.architecture", "flint");
        w.SetMetadata("flint.block_count", (uint)layers);
        w.SetMetadata("flint.context_length", (uint)_config.GetProperty("max_position_embeddings").GetInt32());
        w.SetMetadata("flint.embedding_length", (uint)_config.GetProperty("hidden_size").GetInt32());
        w.SetMetadata("flint.feed_forward_length", (uint)_config.GetProperty("intermediate_size").GetInt32());
        w.SetMetadata("flint.attention.head_count", (uint)_config.GetProperty("num_attention_heads").GetInt32());
        w.SetMetadata("flint.attention.head_count_kv", (uint)_config.GetProperty("num_key_value_heads").GetInt32());
        w.SetMetadata("flint.rope.freq_base", (float)_config.GetProperty("rope_parameters").GetProperty("rope_theta").GetDouble());
        w.SetMetadata("flint.attention.layer_norm_rms_epsilon", (float)_config.GetProperty("rms_norm_eps").GetDouble());
        w.SetMetadata("flint.attention.key_length", (uint)_config.GetProperty("head_dim").GetInt32());
        w.SetMetadata("flint.attention.value_length", (uint)_config.GetProperty("head_dim").GetInt32());
        w.SetMetadata("flint.expert_count", (uint)_config.GetProperty("num_experts").GetInt32());
        w.SetMetadata("flint.expert_used_count", (uint)_config.GetProperty("num_experts_per_tok").GetInt32());
        w.SetMetadata("flint.expert_gating_func", 2u);
        w.SetMetadata("flint.expert_feed_forward_length", (uint)moeInter);
        w.SetMetadata("flint.expert_shared_feed_forward_length", (uint)_config.GetProperty("shared_expert_intermediate_size").GetInt32());
        w.SetMetadata("flint.expert_shared_count", 1u);
        w.SetMetadata("flint.leading_dense_block_count", (uint)_config.GetProperty("mlp_only_layers").GetArrayLength());
        w.SetMetadata("flint.expert_weights_norm", _config.GetProperty("norm_topk_prob").GetBoolean());
        w.SetMetadata("flint.expert_weights_scale", 1f);
        w.SetMetadata("flint.attention.sliding_window", (uint)_config.GetProperty("sliding_window").GetInt32());
        w.SetMetadata("flint.attention.sliding_window_pattern", sliding);
        w.SetMetadata("flint.attention.rope_pattern", sliding);

        Add(w, "token_embd.weight", "model.embed_tokens.weight");
        Add(w, "output_norm.weight", "model.norm.weight");
        Add(w, "output.weight", "lm_head.weight");
        for (int i = 0; i < layers; i++)
        {
            string hf = $"model.layers.{i}", gg = $"blk.{i}";
            Add(w, $"{gg}.attn_norm.weight", $"{hf}.input_layernorm.weight");
            Add(w, $"{gg}.ffn_norm.weight", $"{hf}.post_attention_layernorm.weight");
            foreach ((string to, string from) in new[] { ("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj") })
                Add(w, $"{gg}.{to}.weight", $"{hf}.self_attn.{from}.weight");
            Add(w, $"{gg}.attn_q_norm.weight", $"{hf}.self_attn.q_norm.weight");
            Add(w, $"{gg}.attn_k_norm.weight", $"{hf}.self_attn.k_norm.weight");
            Add(w, $"{gg}.attn_sinks.weight", $"{hf}.self_attn.sinks");
            if (i < _config.GetProperty("mlp_only_layers").GetArrayLength())
            {
                Add(w, $"{gg}.ffn_gate.weight", $"{hf}.mlp.gate_proj.weight");
                Add(w, $"{gg}.ffn_up.weight", $"{hf}.mlp.up_proj.weight");
                Add(w, $"{gg}.ffn_down.weight", $"{hf}.mlp.down_proj.weight");
                continue;
            }
            Add(w, $"{gg}.ffn_gate_inp.weight", $"{hf}.mlp.gate.weight");
            Tensor gateUp = _hf.GetTensor($"{hf}.mlp.experts.gate_up_proj");
            w.AddTensor($"{gg}.ffn_gate_exps.weight", Slice(gateUp, 0, moeInter));
            w.AddTensor($"{gg}.ffn_up_exps.weight", Slice(gateUp, moeInter, moeInter));
            Add(w, $"{gg}.ffn_down_exps.weight", $"{hf}.mlp.experts.down_proj");
            Add(w, $"{gg}.ffn_gate_shexp.weight", $"{hf}.mlp.shared_expert.gate_proj.weight");
            Add(w, $"{gg}.ffn_up_shexp.weight", $"{hf}.mlp.shared_expert.up_proj.weight");
            Add(w, $"{gg}.ffn_down_shexp.weight", $"{hf}.mlp.shared_expert.down_proj.weight");
        }
        w.Flush();
    }

    private void Add(GgufWriter w, string ggufName, string hfName) => w.AddTensor(ggufName, _hf.GetTensor(hfName));

    /// <summary>Rows <paramref name="start"/>..<paramref name="start"/>+<paramref name="count"/> of dimension 1 of an <c>[E, R, C]</c> tensor, as a new tensor.</summary>
    private Tensor Slice(Tensor stack, int start, int count)
    {
        int experts = (int)stack.Shape[0], rows = (int)stack.Shape[1], cols = (int)stack.Shape[2];
        Tensor result = new(new TensorShape(experts, count, cols), DType.F32);
        float* src = (float*)stack.DataPointer, dst = (float*)result.DataPointer;
        for (int e = 0; e < experts; e++)
            Buffer.MemoryCopy(src + ((long)e * rows + start) * cols, dst + (long)e * count * cols, (long)count * cols * 4, (long)count * cols * 4);
        _owned.Add(result);
        return result;
    }
}
