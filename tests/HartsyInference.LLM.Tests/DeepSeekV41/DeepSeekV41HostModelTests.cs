using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.DeepSeekV41.Engram;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41HostModelTests
{
    private static readonly JsonElement Fx = JsonDocument.Parse(File.ReadAllText(Path.Combine(
        RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", "model_forward.json"))).RootElement;

    private static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static float[] P(string name) => Floats(Fx.GetProperty("params").GetProperty(name));

    private static bool Has(string name) => Fx.GetProperty("params").TryGetProperty(name, out _);

    private sealed class FixtureExperts(DeepSeekV41SwigluWeights[] experts) : IDeepSeekV41ExpertSource
    {
        public DeepSeekV41SwigluWeights GetExpert(int expert) => experts[expert];
    }

    private static DeepSeekV41Block BuildBlock(CpuBackend cpu, int layer, DeepSeekV41EngramModule? engram = null, int engramSlot = 0)
    {
        JsonElement cfg = Fx.GetProperty("config");
        int dim = cfg.GetProperty("dim").GetInt32(), hc = cfg.GetProperty("hc_mult").GetInt32(), inter = cfg.GetProperty("moe_inter_dim").GetInt32();
        int experts = cfg.GetProperty("n_routed_experts").GetInt32(), topk = cfg.GetProperty("n_activated_experts").GetInt32();
        float normEps = (float)cfg.GetProperty("norm_eps").GetDouble();
        int ratio = Ints(cfg.GetProperty("compress_ratios"))[layer];
        bool kvSource = Ints(cfg.GetProperty("kv_source_layers")).Contains(layer) && ratio > 0;
        bool indexSource = Ints(cfg.GetProperty("index_source_layers")).Contains(layer) && ratio > 0;
        int candidateLayer = cfg.GetProperty("candidate_source_layer").GetInt32();
        DeepSeekV41AttentionSettings settings = new(dim, cfg.GetProperty("n_heads").GetInt32(), cfg.GetProperty("head_dim").GetInt32(),
            cfg.GetProperty("rope_head_dim").GetInt32(), cfg.GetProperty("q_lora_rank").GetInt32(), cfg.GetProperty("o_groups").GetInt32(),
            cfg.GetProperty("o_lora_rank").GetInt32(), cfg.GetProperty("window_size").GetInt32(), ratio, kvSource, indexSource, layer == candidateLayer,
            candidateLayer >= 0 && candidateLayer < layer, cfg.GetProperty("index_n_heads").GetInt32(), cfg.GetProperty("index_head_dim").GetInt32(),
            cfg.GetProperty("index_topk").GetInt32(), cfg.GetProperty("candidate_topk_blocks").GetInt32(), cfg.GetProperty("candidate_block_size").GetInt32(), normEps);

        string a = $"layers.{layer}.attn.";
        DeepSeekV41CompressorWeights? compressor = kvSource
            ? new(P(a + "compressor.wkv.weight"), ratio > 1 ? P(a + "compressor.wgate.weight") : null, P(a + "compressor.norm.weight")) : null;
        DeepSeekV41IndexerWeights? indexer = indexSource
            ? new(P(a + "indexer.wq_b.weight"), P(a + "indexer.weights_proj.weight"),
                Has(a + "indexer.wk.weight") ? P(a + "indexer.wk.weight") : null, Has(a + "indexer.k_norm.weight") ? P(a + "indexer.k_norm.weight") : null) : null;
        DeepSeekV41AttentionWeights weights = new(P(a + "wq_a.weight"), P(a + "q_norm.weight"), P(a + "wq_b.weight"), P(a + "wkv.weight"), P(a + "kv_norm.weight"),
            P(a + "wo_a.weight"), P(a + "wo_b.weight"), P(a + "attn_sink"), compressor, indexer);
        int length = cfg.GetProperty("max_seq_len").GetInt32(), rd = settings.RopeDim;
        DeepSeekV41RopeTable rope = ratio > 0
            ? DeepSeekV41RopeTable.Build(rd, length, cfg.GetProperty("compress_rope_theta").GetDouble(),
                new DeepSeekV41RopeScaling(cfg.GetProperty("rope_factor").GetDouble(), 32, 1, cfg.GetProperty("original_seq_len").GetInt32()))
            : DeepSeekV41RopeTable.Build(rd, length, cfg.GetProperty("rope_theta").GetDouble(), null);
        DeepSeekV41Attention attention = new(cpu, settings, weights, rope);

        string f = $"layers.{layer}.ffn.";
        DeepSeekV41SwigluWeights[] routed = Enumerable.Range(0, experts).Select(e =>
            new DeepSeekV41SwigluWeights(dim, inter, P($"{f}experts.{e}.w1.weight"), P($"{f}experts.{e}.w2.weight"), P($"{f}experts.{e}.w3.weight"))).ToArray();
        DeepSeekV41SwigluWeights shared = new(dim, inter, P(f + "shared_experts.w1.weight"), P(f + "shared_experts.w2.weight"), P(f + "shared_experts.w3.weight"));
        MoeRouteArgs route = new(experts, topk, MoeRouteScoring.SqrtSoftplus, Renormalize: topk > 1, RenormEpsilon: 1e-20f,
            Scale: (float)cfg.GetProperty("route_scale").GetDouble());
        DeepSeekV41MoeLayer ffn = new(cpu, P(f + "gate.weight"), P(f + "gate.bias"), null, route, new FixtureExperts(routed), shared,
            (float)cfg.GetProperty("swiglu_limit").GetDouble());

        int iters = cfg.GetProperty("hc_sinkhorn_iters").GetInt32();
        float hcEps = (float)cfg.GetProperty("hc_eps").GetDouble();
        string l = $"layers.{layer}.";
        DeepSeekV41HyperConnection hcAttn = new(cpu, hc, dim, iters, hcEps, normEps, P(l + "hc_attn_fn"), P(l + "hc_attn_scale"), P(l + "hc_attn_base"));
        DeepSeekV41HyperConnection hcFfn = new(cpu, hc, dim, iters, hcEps, normEps, P(l + "hc_ffn_fn"), P(l + "hc_ffn_scale"), P(l + "hc_ffn_base"));
        return new DeepSeekV41Block(dim, hc, normEps, hcAttn, hcFfn, P(l + "attn_norm.weight"), P(l + "ffn_norm.weight"), attention, ffn, engram, engramSlot);
    }

    private static DeepSeekV41HostModel BuildModel(CpuBackend cpu)
    {
        JsonElement cfg = Fx.GetProperty("config");
        DeepSeekV41Block[] blocks = Enumerable.Range(0, cfg.GetProperty("n_layers").GetInt32()).Select(i => BuildBlock(cpu, i)).ToArray();
        return new DeepSeekV41HostModel(cfg.GetProperty("dim").GetInt32(), cfg.GetProperty("hc_mult").GetInt32(), cfg.GetProperty("vocab_size").GetInt32(),
            (float)cfg.GetProperty("norm_eps").GetDouble(), P("embed.weight"), blocks, P("norm.weight"), P("head.weight"));
    }

    // The fixture runs upstream with an exact-softmax sparse_attn, so what remains is float32 accumulation order; the real kernel's bf16 probabilities are not modelled.
    private const float Tolerance = 1e-3f;

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= Tolerance * Math.Max(1f, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    [Fact]
    public void Hidden_States_And_Logits_Match_Upstream_Through_Prefill_And_Decode()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = BuildModel(cpu);
        DeepSeekV41SequenceState state = model.CreateState(64);
        int stepNo = 0;
        foreach (JsonElement step in Fx.GetProperty("steps").EnumerateArray())
        {
            int[] ids = Ints(step.GetProperty("ids"));
            float[] hidden = new float[ids.Length * model.Dim];
            model.Forward(ids, state, hidden);
            AssertClose(Floats(step.GetProperty("final")), hidden, $"step {stepNo} hidden");
            AssertClose(Floats(step.GetProperty("logits")), model.Logits(hidden.AsSpan((ids.Length - 1) * model.Dim, model.Dim)), $"step {stepNo} logits");
            stepNo++;
        }
        Assert.Equal(17, state.Length);
    }

    [Fact]
    public void A_Multi_Token_Continuation_Equals_Single_Token_Steps()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = BuildModel(cpu);
        int[] ids = Ints(Fx.GetProperty("steps")[0].GetProperty("ids"));
        int[] more = [3, 9, 17, 4];
        float[] chunked = new float[more.Length * model.Dim], stepped = new float[more.Length * model.Dim];

        DeepSeekV41SequenceState a = model.CreateState(64), b = model.CreateState(64);
        model.Forward(ids, a, new float[ids.Length * model.Dim]);
        model.Forward(ids, b, new float[ids.Length * model.Dim]);
        model.Forward(more, a, chunked);
        for (int i = 0; i < more.Length; i++) model.Forward(more.AsSpan(i, 1), b, stepped.AsSpan(i * model.Dim, model.Dim));

        Assert.Equal(stepped, chunked);
        Assert.Equal(ids.Length + more.Length, a.Length);
    }

    [Fact]
    public void Rejects_Bad_Ids_Overflow_And_A_Reset_Restarts_The_Sequence()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = BuildModel(cpu);
        DeepSeekV41SequenceState state = model.CreateState(4);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Forward(new[] { 999 }, state, new float[model.Dim]));
        Assert.Throws<InvalidOperationException>(() => model.Forward(new[] { 1, 2, 3, 4, 5 }, state, new float[5 * model.Dim]));
        float[] first = new float[2 * model.Dim], again = new float[2 * model.Dim];
        model.Forward(new[] { 1, 2 }, state, first);
        state.Reset();
        model.Forward(new[] { 1, 2 }, state, again);
        Assert.Equal(first, again);
    }

    [Fact]
    public void Engram_Receives_The_Hasher_Slice_Of_Its_Own_Layer_For_Every_Position()
    {
        JsonElement cfg = Fx.GetProperty("config");
        int dim = cfg.GetProperty("dim").GetInt32(), hc = cfg.GetProperty("hc_mult").GetInt32(), layers = cfg.GetProperty("n_layers").GetInt32();
        const int headDim = 4;
        List<long> requested = [];
        void Gather(ReadOnlySpan<long> rows, Span<ushort> dest)
        {
            requested.AddRange(rows.ToArray());
            dest.Clear();
        }
        int columns = EngramConstants.ColumnsPerLayer;
        DeepSeekV41EngramModule engram = new(dim, hc, columns, headDim, 1e-6f, new float[dim * (hc + 1) * columns * headDim], new float[hc * dim],
            new float[hc * dim], Gather);

        using CpuBackend cpu = new();
        DeepSeekV41Block[] blocks = Enumerable.Range(0, layers).Select(i => i == 0 ? BuildBlock(cpu, 0, engram, engramSlot: 1) : BuildBlock(cpu, i)).ToArray();
        DeepSeekV41HostModel model = new(dim, hc, cfg.GetProperty("vocab_size").GetInt32(), (float)cfg.GetProperty("norm_eps").GetDouble(), P("embed.weight"),
            blocks, P("norm.weight"), P("head.weight"));
        DeepSeekV41SequenceState state = model.CreateState(64);
        Assert.NotNull(state.Hasher);

        int[] ids = [5, 9, 9, 2, 30, 1];
        model.Forward(ids, state, new float[ids.Length * dim]);

        EngramHasher reference = new();
        long[] all = new long[ids.Length * reference.ValuesPerPosition];
        bool[] live = Enumerable.Repeat(true, ids.Length).ToArray();
        reference.Hash(ids, live, 0, all);
        int hashLayers = reference.ValuesPerPosition / columns;
        List<long> expected = [];
        for (int t = 0; t < ids.Length; t++) expected.AddRange(all.AsSpan((t * hashLayers + 1) * columns, columns).ToArray());
        Assert.Equal(expected, requested);
    }
}
