using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The DSpark head and the target's draft taps on the synthetic model, against the upstream run in <c>fixtures/dspark_forward.json</c> (written by
/// <c>dump_dspark_fixture.py</c>). The target is the backbone of <c>model_forward.json</c>; the draft weights are random and seeded. Ungated: runs on the CPU lane.</summary>
/// <remarks>Each decode position is drafted from a fresh window seeded with the committed rows before it, and must reproduce upstream's incremental window. The
/// positions past the 8-token window cover the wrap. Tolerance as the host fixture: float32 accumulation order, exact softmax.</remarks>
public sealed class DeepSeekV41DSparkFixtureTests
{
    private const float Tolerance = 1e-3f;

    private static readonly JsonElement Host = Read("model_forward.json");
    private static readonly JsonElement Dspark = Read("dspark_forward.json");

    /// <summary>Expert weights by index, as the host fixture's experts are supplied to the feed-forward layer.</summary>
    private sealed class DraftExperts(DeepSeekV41SwigluWeights[] experts) : IDeepSeekV41ExpertSource
    {
        public DeepSeekV41SwigluWeights GetExpert(int expert) => experts[expert];
    }

    [Fact]
    public void Target_Taps_Reproduce_Upstream_Main_Hidden_At_Every_Position()
    {
        using CpuBackend cpu = new();
        int[] ids = Ints(Dspark.GetProperty("ids"));
        float[] upstream = Floats(Dspark.GetProperty("main_hidden"));
        int width = Dspark.GetProperty("main_hidden_width").GetInt32();
        float[] tapped = RunTarget(cpu, ids, width);
        AssertClose(upstream, tapped, "main_hidden");
    }

    [Fact]
    public void Each_Decode_Draft_Matches_Upstream_From_A_Fresh_Seed()
    {
        using CpuBackend cpu = new();
        DeepSeekV41DSpark dspark = BuildDSpark(cpu);
        int[] ids = Ints(Dspark.GetProperty("ids"));
        int width = Dspark.GetProperty("main_hidden_width").GetInt32();
        float[] upstream = Floats(Dspark.GetProperty("main_hidden"));
        foreach ((int pos, JsonElement expected) in Drafts())
        {
            DeepSeekV41DSparkState state = dspark.CreateState(64);
            dspark.Seed(upstream.AsSpan(0, pos * width), pos, state);
            DeepSeekV41DSparkDraft draft = dspark.Draft(ids[pos], upstream.AsSpan(pos * width, width), pos, state);
            Compare(draft, expected, pos);
        }
    }

    [Fact]
    public void Drafts_From_The_Target_Taps_Match_Upstream_At_Every_Decode_Step()
    {
        using CpuBackend cpu = new();
        DeepSeekV41DSpark dspark = BuildDSpark(cpu);
        int[] ids = Ints(Dspark.GetProperty("ids"));
        int width = Dspark.GetProperty("main_hidden_width").GetInt32();
        float[] tapped = RunTarget(cpu, ids, width);
        foreach ((int pos, JsonElement expected) in Drafts())
        {
            DeepSeekV41DSparkState state = dspark.CreateState(64);
            dspark.Seed(tapped.AsSpan(0, pos * width), pos, state);
            DeepSeekV41DSparkDraft draft = dspark.Draft(ids[pos], tapped.AsSpan(pos * width, width), pos, state);
            Compare(draft, expected, pos);
        }
    }

    /// <summary>Runs the target over the fixture's ids the way upstream did: the 11-token prefill as one chunk, then one token per decode step. Returns the taps of
    /// every position, <c>[ids, width]</c>.</summary>
    private static float[] RunTarget(CpuBackend cpu, int[] ids, int width)
    {
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu, Ints(Dspark.GetProperty("target_layers")));
        Assert.Equal(width, model.MainHiddenWidth);
        DeepSeekV41SequenceState state = model.CreateState(64);
        float[] tapped = new float[ids.Length * width];
        model.Forward(ids.AsSpan(0, 11), state, new float[11 * model.Dim], tapped.AsSpan(0, 11 * width));
        for (int p = 11; p < ids.Length; p++) model.Forward(ids.AsSpan(p, 1), state, new float[model.Dim], tapped.AsSpan(p * width, width));
        return tapped;
    }

    private static IEnumerable<(int Pos, JsonElement Expected)> Drafts() =>
        Dspark.GetProperty("drafts").EnumerateObject().Select(p => (int.Parse(p.Name), p.Value));

    private static void Compare(DeepSeekV41DSparkDraft draft, JsonElement expected, int pos)
    {
        Assert.Equal(Ints(expected.GetProperty("ids")), draft.Ids);
        AssertClose(Floats(expected.GetProperty("logits")), draft.Logits, $"pos {pos} logits");
        AssertClose(Floats(expected.GetProperty("confidence")), draft.Confidence, $"pos {pos} confidence");
    }

    /// <summary>The DSpark head from the fixture's mtp tensors, built the way <see cref="DeepSeekV41DSpark.Load(IBackend, DeepSeekV41Checkpoint, DeepSeekV41LoadOptions)"/>
    /// builds it from a checkpoint, with the backbone's embedding and head.</summary>
    private static DeepSeekV41DSpark BuildDSpark(CpuBackend cpu)
    {
        JsonElement cfg = Dspark.GetProperty("config");
        int dim = cfg.GetProperty("dim").GetInt32(), hc = cfg.GetProperty("hc_mult").GetInt32(), vocab = cfg.GetProperty("vocab_size").GetInt32();
        int heads = cfg.GetProperty("n_heads").GetInt32(), hd = cfg.GetProperty("head_dim").GetInt32(), rd = cfg.GetProperty("rope_head_dim").GetInt32();
        int qLora = cfg.GetProperty("q_lora_rank").GetInt32(), inter = cfg.GetProperty("moe_inter_dim").GetInt32(), window = cfg.GetProperty("window_size").GetInt32();
        int experts = cfg.GetProperty("dspark_n_routed_experts").GetInt32(), topk = cfg.GetProperty("dspark_n_activated_experts").GetInt32();
        int stages = cfg.GetProperty("n_mtp_layers").GetInt32(), iters = cfg.GetProperty("hc_sinkhorn_iters").GetInt32();
        float normEps = (float)cfg.GetProperty("norm_eps").GetDouble(), hcEps = (float)cfg.GetProperty("hc_eps").GetDouble();
        int rank = Dspark.GetProperty("markov_rank").GetInt32(), block = Dspark.GetProperty("block_size").GetInt32(), noise = Dspark.GetProperty("noise_token_id").GetInt32();
        int targets = Dspark.GetProperty("target_layers").GetArrayLength();
        int mix = (2 + hc) * hc;

        DeepSeekV41RopeTable rope = DeepSeekV41RopeTable.Build(rd, cfg.GetProperty("max_seq_len").GetInt32(), cfg.GetProperty("rope_theta").GetDouble(), null);
        DeepSeekV41DSparkStage[] draftStages = new DeepSeekV41DSparkStage[stages];
        for (int s = 0; s < stages; s++)
        {
            string l = $"mtp.{s}.", a = l + "attn.", f = l + "ffn.";
            DeepSeekV41AttentionSettings settings = new(dim, heads, hd, rd, qLora, cfg.GetProperty("o_groups").GetInt32(), cfg.GetProperty("o_lora_rank").GetInt32(),
                window, 0, false, false, false, false, cfg.GetProperty("index_n_heads").GetInt32(), cfg.GetProperty("index_head_dim").GetInt32(),
                cfg.GetProperty("index_topk").GetInt32(), cfg.GetProperty("candidate_topk_blocks").GetInt32(), cfg.GetProperty("candidate_block_size").GetInt32(), normEps);
            DeepSeekV41AttentionWeights weights = new(Mtp(a + "wq_a.weight"), Mtp(a + "q_norm.weight"), Mtp(a + "wq_b.weight"), Mtp(a + "wkv.weight"), Mtp(a + "kv_norm.weight"),
                Mtp(a + "wo_a.weight"), Mtp(a + "wo_b.weight"), Mtp(a + "attn_sink"), null, null);
            DeepSeekV41DSparkAttention attention = new(cpu, settings, weights, rope);

            DeepSeekV41SwigluWeights[] routed = Enumerable.Range(0, experts).Select(e =>
                new DeepSeekV41SwigluWeights(dim, inter, Mtp($"{f}experts.{e}.w1.weight"), Mtp($"{f}experts.{e}.w2.weight"), Mtp($"{f}experts.{e}.w3.weight"))).ToArray();
            DeepSeekV41SwigluWeights shared = new(dim, inter, Mtp(f + "shared_experts.w1.weight"), Mtp(f + "shared_experts.w2.weight"), Mtp(f + "shared_experts.w3.weight"));
            MoeRouteArgs route = new(experts, topk, MoeRouteScoring.SqrtSoftplus, Renormalize: topk > 1, RenormEpsilon: 1e-20f,
                Scale: (float)cfg.GetProperty("route_scale").GetDouble());
            DeepSeekV41MoeLayer ffn = new(cpu, Mtp(f + "gate.weight"), Mtp(f + "gate.bias"), null, route, new DraftExperts(routed), shared,
                (float)cfg.GetProperty("swiglu_limit").GetDouble());

            DeepSeekV41HyperConnection hcAttn = new(cpu, hc, dim, iters, hcEps, normEps, Mtp(l + "hc_attn_fn"), Mtp(l + "hc_attn_scale"), Mtp(l + "hc_attn_base"));
            DeepSeekV41HyperConnection hcFfn = new(cpu, hc, dim, iters, hcEps, normEps, Mtp(l + "hc_ffn_fn"), Mtp(l + "hc_ffn_scale"), Mtp(l + "hc_ffn_base"));
            draftStages[s] = new DeepSeekV41DSparkStage(dim, hc, normEps, hcAttn, hcFfn, Mtp(l + "attn_norm.weight"), Mtp(l + "ffn_norm.weight"), attention, ffn);
        }

        int last = stages - 1;
        return new DeepSeekV41DSpark(cpu, dim, hc, vocab, rank, block, noise, normEps, Mtp("mtp.0.main_proj.weight"), targets * dim, Mtp("mtp.0.main_norm.weight"),
            Floats(Host.GetProperty("params").GetProperty("embed.weight")), Floats(Host.GetProperty("params").GetProperty("head.weight")),
            draftStages, Mtp($"mtp.{last}.norm.weight"), Mtp($"mtp.{last}.markov_head.embed.weight"), Mtp($"mtp.{last}.markov_head.head.weight"),
            Mtp($"mtp.{last}.confidence_head.proj.weight"));
    }

    private static JsonElement Read(string fileName) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", fileName))).RootElement;

    private static float[] Mtp(string name) => Floats(Dspark.GetProperty("mtp_params").GetProperty(name));

    private static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= Tolerance * Math.Max(1f, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }
}
