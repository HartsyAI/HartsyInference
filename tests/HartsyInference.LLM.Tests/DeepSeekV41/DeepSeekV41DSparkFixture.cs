using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Sampling;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The synthetic DSpark fixture shared by the draft-head and proposer tests: the backbone of <c>model_forward.json</c>, the upstream draft head in
/// <c>dspark_forward.json</c> (written by <c>dump_dspark_fixture.py</c>), and a builder that makes the head from the fixture's tensors the way
/// <see cref="DeepSeekV41DSpark.Load(IBackend, DeepSeekV41Checkpoint, DeepSeekV41LoadOptions)"/> makes it from a checkpoint.</summary>
internal static class DeepSeekV41DSparkFixture
{
    /// <summary>The longest sequence the tests keep state for.</summary>
    public const int MaxTokens = 64;

    /// <summary>float32 accumulation order and exact softmax, as in the host fixture.</summary>
    public const float Tolerance = 1e-3f;

    private static readonly JsonElement Host = Read("model_forward.json");
    private static readonly JsonElement Dspark = Read("dspark_forward.json");

    /// <summary>The 17 token ids: the 11-token prefill and six decode steps.</summary>
    public static int[] Ids => Ints(Dspark.GetProperty("ids"));

    /// <summary>Hidden width of one target row: one <c>dim</c> per draft layer.</summary>
    public static int Width => Dspark.GetProperty("main_hidden_width").GetInt32();

    /// <summary>Upstream's <c>main_hidden</c> for every position, <c>[17, Width]</c>.</summary>
    public static float[] MainHidden => Floats(Dspark.GetProperty("main_hidden"));

    /// <summary>The target layers whose entry streams the draft reads.</summary>
    public static int[] TargetLayers => Ints(Dspark.GetProperty("target_layers"));

    /// <summary>Upstream's draft at each decode position, keyed by position.</summary>
    public static IEnumerable<(int Pos, JsonElement Expected)> Drafts() =>
        Dspark.GetProperty("drafts").EnumerateObject().Select(p => (int.Parse(p.Name), p.Value));

    /// <summary>Runs the target over <paramref name="ids"/> the way upstream did: the 11-token prefill as one chunk, then one token per decode step. Returns the taps
    /// of every position, <c>[ids, Width]</c>.</summary>
    public static float[] RunTarget(CpuBackend cpu, int[] ids)
    {
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu, TargetLayers);
        Assert.Equal(Width, model.MainHiddenWidth);
        DeepSeekV41SequenceState state = model.CreateState(MaxTokens);
        float[] tapped = new float[ids.Length * Width];
        model.Forward(ids.AsSpan(0, 11), state, new float[11 * model.Dim], tapped.AsSpan(0, 11 * Width));
        for (int p = 11; p < ids.Length; p++) model.Forward(ids.AsSpan(p, 1), state, new float[model.Dim], tapped.AsSpan(p * Width, Width));
        return tapped;
    }

    /// <summary>Plain greedy decoding on the backbone: the prompt as one prefill, then one token at a time, with no speculation.</summary>
    public static List<int> PlainGreedy(DeepSeekV41HostModel model, int[] prompt, int count)
    {
        List<int> tokens = [.. prompt];
        DeepSeekV41GenerationState state = new(model, MaxTokens);
        float[] hidden = new float[prompt.Length * model.Dim];
        state.Append(prompt, hidden);
        float[] logits = model.Logits(hidden.AsSpan((prompt.Length - 1) * model.Dim, model.Dim));
        SamplerChain chain = SamplerChain.FromOptions(new SamplingOptions { Greedy = true });
        for (int i = 0; i < count; i++)
        {
            int next = chain.Next(logits, tokens);
            tokens.Add(next);
            float[] step = new float[model.Dim];
            state.Append([next], step);
            logits = model.Logits(step);
        }
        return tokens;
    }

    /// <summary>The embedding and head of the backbone, as the draft head shares them.</summary>
    public static (float[] Embed, float[] Head) TargetEmbedAndHead() =>
        (Floats(Host.GetProperty("params").GetProperty("embed.weight")), Floats(Host.GetProperty("params").GetProperty("head.weight")));

    /// <summary>The draft head from the fixture's tensors, with the backbone's embedding and head.</summary>
    public static DeepSeekV41DSpark BuildDSpark(CpuBackend cpu)
    {
        JsonElement cfg = Dspark.GetProperty("config");
        int dim = cfg.GetProperty("dim").GetInt32(), hc = cfg.GetProperty("hc_mult").GetInt32(), vocab = cfg.GetProperty("vocab_size").GetInt32();
        int heads = cfg.GetProperty("n_heads").GetInt32(), hd = cfg.GetProperty("head_dim").GetInt32(), rd = cfg.GetProperty("rope_head_dim").GetInt32();
        int qLora = cfg.GetProperty("q_lora_rank").GetInt32(), inter = cfg.GetProperty("moe_inter_dim").GetInt32(), window = cfg.GetProperty("window_size").GetInt32();
        int experts = cfg.GetProperty("dspark_n_routed_experts").GetInt32(), topk = cfg.GetProperty("dspark_n_activated_experts").GetInt32();
        int stages = cfg.GetProperty("n_mtp_layers").GetInt32(), iters = cfg.GetProperty("hc_sinkhorn_iters").GetInt32();
        float normEps = (float)cfg.GetProperty("norm_eps").GetDouble(), hcEps = (float)cfg.GetProperty("hc_eps").GetDouble();
        int rank = Dspark.GetProperty("markov_rank").GetInt32(), block = Dspark.GetProperty("block_size").GetInt32(), noise = Dspark.GetProperty("noise_token_id").GetInt32();
        int targets = TargetLayers.Length;

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
        (float[] embed, float[] head) = TargetEmbedAndHead();
        return new DeepSeekV41DSpark(cpu, dim, hc, vocab, rank, block, noise, normEps, Mtp("mtp.0.main_proj.weight"), targets * dim, Mtp("mtp.0.main_norm.weight"),
            embed, head, draftStages, Mtp($"mtp.{last}.norm.weight"), Mtp($"mtp.{last}.markov_head.embed.weight"), Mtp($"mtp.{last}.markov_head.head.weight"),
            Mtp($"mtp.{last}.confidence_head.proj.weight"));
    }

    /// <summary>One draft tensor, checked against its recorded shape, so a generator and loader mismatch fails here with its name.</summary>
    private static float[] Mtp(string name)
    {
        float[] values = Floats(Dspark.GetProperty("mtp_params").GetProperty(name));
        long expected = Dspark.GetProperty("mtp_shapes").GetProperty(name).EnumerateArray().Aggregate(1L, (n, d) => n * d.GetInt64());
        Assert.True(values.Length == expected, $"{name} holds {values.Length} values; its recorded shape has {expected}");
        return values;
    }

    public static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    public static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    public static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= Tolerance * Math.Max(1f, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    private static JsonElement Read(string fileName) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", fileName))).RootElement;

    /// <summary>Expert weights by index, as the host fixture's experts are supplied to the feed-forward layer.</summary>
    private sealed class DraftExperts(DeepSeekV41SwigluWeights[] experts) : IDeepSeekV41ExpertSource
    {
        public DeepSeekV41SwigluWeights GetExpert(int expert) => experts[expert];
    }
}
