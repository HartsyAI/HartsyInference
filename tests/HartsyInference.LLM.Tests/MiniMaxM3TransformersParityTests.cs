using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>A tiny random MiniMax-M3 run through transformers' own <c>MiniMaxM3VLForCausalLM</c> (<c>tests/python-reference/dump_minimax_m3_tiny_reference.py</c>) must
/// match <see cref="GenericTransformer"/> loaded through the GGUF path: the Hugging Face weights are written as a <c>minimax-m3</c> GGUF (norm weights with the Gemma +1 baked in,
/// fused gate/up split, router bias as <c>exp_probs_b</c>, indexer tensors under <c>indexer.*</c>), then loaded by <see cref="GgufModelLoader"/>, <see cref="GgufLanguageModel.PrepareWeights"/>
/// and <see cref="GgufConfigFactory"/>. The fixture keeps the release's block traits: two dense layers then sigmoid-routed MoE layers with a selection bias, routed scaling and a shared expert,
/// the clamped alpha-SwiGLU, and block-sparse attention on every layer whose 24-token prompt really prunes (6 key blocks, 2 selected plus the local one). The GGUF tensor names for the indexer are
/// this port's own; no upstream GGUF exists yet.</summary>
public sealed unsafe class MiniMaxM3TransformersParityTests : IDisposable
{
    private const float Tolerance = 2e-4f;
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "MiniMaxM3");

    private readonly SafeTensorsLoader _hf = new();
    private readonly SafeTensorsLoader _expected = new();
    private readonly List<Tensor> _owned = [];
    private readonly string _gguf = Path.Combine(Path.GetTempPath(), $"minimax-m3-tiny-{Guid.NewGuid():N}.gguf");
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly JsonElement _config;
    private readonly JsonDocument _configDocument;
    private readonly CpuBackend _backend = new();

    public MiniMaxM3TransformersParityTests()
    {
        _hf.Load(Path.Combine(Dir, "minimax_m3_tiny.safetensors"));
        _expected.Load(Path.Combine(Dir, "minimax_m3_tiny_expected.safetensors"));
        _configDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "minimax_m3_tiny_config.json")));
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
    public void GgufConfig_DescribesTheMiniMaxLayout()
    {
        Assert.Equal("minimax-m3", Config.Architecture);
        Assert.Equal(ActivationKind.SwigluOai, Config.Activation);
        Assert.True(Config.QkNorm);
        IndexerConfig index = Config.Indexer!;
        Assert.Equal((2, 16, 4, 2, 1), (index.Heads, index.KeyLength, index.BlockSize, index.TopKBlocks, index.LocalBlocks));
        MoeConfig moe = Config.Moe!;
        Assert.Equal((8, 3, 16, 24, MoeScoring.Sigmoid, true, 2, 2f), (moe.NumExperts, moe.NumExpertsPerTok, moe.MoeIntermediateSize,
            moe.SharedExpertIntermediateSize, moe.Scoring, moe.NormTopKProb, moe.FirstDenseLayers, moe.RoutedScalingFactor));
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
        int moeInter = _config.GetProperty("intermediate_size").GetInt32();
        int denseInter = _config.GetProperty("dense_intermediate_size").GetInt32();
        int sharedInter = _config.GetProperty("shared_intermediate_size").GetInt32();
        int denseLayers = _config.GetProperty("mlp_layer_types").EnumerateArray().Count(e => e.GetString() == "dense");
        using GgufWriter w = new(_gguf);
        w.SetMetadata("general.architecture", "minimax-m3");
        w.SetMetadata("minimax-m3.block_count", (uint)layers);
        w.SetMetadata("minimax-m3.context_length", 4096u);
        w.SetMetadata("minimax-m3.embedding_length", (uint)_config.GetProperty("hidden_size").GetInt32());
        w.SetMetadata("minimax-m3.feed_forward_length", (uint)denseInter);
        w.SetMetadata("minimax-m3.attention.head_count", (uint)_config.GetProperty("num_attention_heads").GetInt32());
        w.SetMetadata("minimax-m3.attention.head_count_kv", (uint)_config.GetProperty("num_key_value_heads").GetInt32());
        w.SetMetadata("minimax-m3.rope.freq_base", (float)_config.GetProperty("rope_parameters").GetProperty("rope_theta").GetDouble());
        w.SetMetadata("minimax-m3.rope.dimension_count", (uint)(_config.GetProperty("head_dim").GetInt32() * _config.GetProperty("rope_parameters").GetProperty("partial_rotary_factor").GetDouble()));
        w.SetMetadata("minimax-m3.attention.layer_norm_rms_epsilon", (float)_config.GetProperty("rms_norm_eps").GetDouble());
        w.SetMetadata("minimax-m3.attention.key_length", (uint)_config.GetProperty("head_dim").GetInt32());
        w.SetMetadata("minimax-m3.attention.value_length", (uint)_config.GetProperty("head_dim").GetInt32());
        w.SetMetadata("minimax-m3.attention.indexer.head_count", (uint)_config.GetProperty("index_n_heads").GetInt32());
        w.SetMetadata("minimax-m3.attention.indexer.key_length", (uint)_config.GetProperty("index_head_dim").GetInt32());
        w.SetMetadata("minimax-m3.attention.indexer.block_size", (uint)_config.GetProperty("index_block_size").GetInt32());
        w.SetMetadata("minimax-m3.attention.indexer.top_k_blocks", (uint)_config.GetProperty("index_topk_blocks").GetInt32());
        w.SetMetadata("minimax-m3.attention.indexer.local_blocks", (uint)_config.GetProperty("index_local_blocks").GetInt32());
        w.SetMetadata("minimax-m3.swiglu_alpha", (float)_config.GetProperty("swiglu_alpha").GetDouble());
        w.SetMetadata("minimax-m3.swiglu_limit", (float)_config.GetProperty("swiglu_limit").GetDouble());
        w.SetMetadata("minimax-m3.expert_count", (uint)_config.GetProperty("num_local_experts").GetInt32());
        w.SetMetadata("minimax-m3.expert_used_count", (uint)_config.GetProperty("num_experts_per_tok").GetInt32());
        w.SetMetadata("minimax-m3.expert_gating_func", 2u);
        w.SetMetadata("minimax-m3.expert_feed_forward_length", (uint)moeInter);
        w.SetMetadata("minimax-m3.expert_shared_feed_forward_length", (uint)sharedInter);
        w.SetMetadata("minimax-m3.leading_dense_block_count", (uint)denseLayers);
        w.SetMetadata("minimax-m3.expert_weights_norm", true);
        w.SetMetadata("minimax-m3.expert_weights_scale", (float)_config.GetProperty("routed_scaling_factor").GetDouble());

        Add(w, "token_embd.weight", "model.embed_tokens.weight");
        AddNorm(w, "output_norm.weight", "model.norm.weight");
        Add(w, "output.weight", "lm_head.weight");
        for (int i = 0; i < layers; i++)
        {
            string hf = $"model.layers.{i}", gg = $"blk.{i}";
            AddNorm(w, $"{gg}.attn_norm.weight", $"{hf}.input_layernorm.weight");
            AddNorm(w, $"{gg}.ffn_norm.weight", $"{hf}.post_attention_layernorm.weight");
            foreach ((string to, string from) in new[] { ("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj") })
                Add(w, $"{gg}.{to}.weight", $"{hf}.self_attn.{from}.weight");
            AddNorm(w, $"{gg}.attn_q_norm.weight", $"{hf}.self_attn.q_norm.weight");
            AddNorm(w, $"{gg}.attn_k_norm.weight", $"{hf}.self_attn.k_norm.weight");
            Add(w, $"{gg}.indexer.attn_q.weight", $"{hf}.self_attn.indexer.q_proj.weight");
            Add(w, $"{gg}.indexer.attn_k.weight", $"{hf}.self_attn.indexer.k_proj.weight");
            AddNorm(w, $"{gg}.indexer.attn_q_norm.weight", $"{hf}.self_attn.indexer.q_norm.weight");
            AddNorm(w, $"{gg}.indexer.attn_k_norm.weight", $"{hf}.self_attn.indexer.k_norm.weight");
            if (i < denseLayers)
            {
                Tensor gateUp = _hf.GetTensor($"{hf}.mlp.gate_up_proj.weight");
                w.AddTensor($"{gg}.ffn_gate.weight", Rows(gateUp, 0, denseInter));
                w.AddTensor($"{gg}.ffn_up.weight", Rows(gateUp, denseInter, denseInter));
                Add(w, $"{gg}.ffn_down.weight", $"{hf}.mlp.down_proj.weight");
                continue;
            }
            Add(w, $"{gg}.ffn_gate_inp.weight", $"{hf}.mlp.gate.weight");
            Add(w, $"{gg}.exp_probs_b.bias", $"{hf}.mlp.gate.e_score_correction_bias");
            Tensor experts = _hf.GetTensor($"{hf}.mlp.experts.gate_up_proj");
            w.AddTensor($"{gg}.ffn_gate_exps.weight", Slice(experts, 0, moeInter));
            w.AddTensor($"{gg}.ffn_up_exps.weight", Slice(experts, moeInter, moeInter));
            Add(w, $"{gg}.ffn_down_exps.weight", $"{hf}.mlp.experts.down_proj");
            Tensor shared = _hf.GetTensor($"{hf}.mlp.shared_experts.gate_up_proj.weight");
            w.AddTensor($"{gg}.ffn_gate_shexp.weight", Rows(shared, 0, sharedInter));
            w.AddTensor($"{gg}.ffn_up_shexp.weight", Rows(shared, sharedInter, sharedInter));
            Add(w, $"{gg}.ffn_down_shexp.weight", $"{hf}.mlp.shared_experts.down_proj.weight");
        }
        w.Flush();
    }

    /// <summary>The Gemma-form norm weight with its +1 baked in, as llama.cpp's converter stores it.</summary>
    private void AddNorm(GgufWriter w, string ggufName, string hfName)
    {
        Tensor source = _hf.GetTensor(hfName);
        Tensor baked = new(source.Shape, DType.F32);
        ReadOnlySpan<float> from = source.AsReadOnlySpan<float>();
        Span<float> to = baked.AsSpan<float>();
        for (int i = 0; i < to.Length; i++) to[i] = from[i] + 1f;
        _owned.Add(baked);
        w.AddTensor(ggufName, baked);
    }

    /// <summary>Rows <paramref name="start"/>..<paramref name="start"/>+<paramref name="count"/> of a <c>[R, C]</c> tensor, as a new tensor.</summary>
    private Tensor Rows(Tensor matrix, int start, int count)
    {
        int cols = (int)matrix.Shape[1];
        Tensor result = new(new TensorShape(count, cols), DType.F32);
        matrix.AsReadOnlySpan<float>().Slice(start * cols, count * cols).CopyTo(result.AsSpan<float>());
        _owned.Add(result);
        return result;
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
