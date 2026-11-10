using System.Globalization;
using System.Text.RegularExpressions;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ernie5;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The ERNIE-5.5 multi-token-prediction layer on the host reference path, against the synthetic checkpoint's layer 2.</summary>
public sealed class Ernie5MtpTests : IDisposable
{
    // Layer 43 of the real model.safetensors.index.json, scale companions dropped and expert indices folded to N.
    private static readonly string[] RealMtpLeaves =
    [
        "block_sparse_moe.e_score_correction_bias", "block_sparse_moe.experts.N.w1.weight", "block_sparse_moe.experts.N.w2.weight",
        "block_sparse_moe.experts.N.w3.weight", "block_sparse_moe.fc1_latent_proj.weight", "block_sparse_moe.fc2_latent_proj.weight",
        "block_sparse_moe.gate.routed_scaling_factor_param", "block_sparse_moe.gate.weight", "block_sparse_moe.gate.weight_1",
        "block_sparse_moe.latent_norm.weight", "block_sparse_moe.shared_experts.w1.weight", "block_sparse_moe.shared_experts.w2.weight",
        "block_sparse_moe.shared_experts.w3.weight", "e_proj.weight", "enorm.weight", "h_proj.weight", "hc_head_base", "hc_head_fn",
        "hc_head_scale", "hnorm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.core_attention.indexer.k_norm.bias", "self_attn.core_attention.indexer.k_norm.weight",
        "self_attn.core_attention.indexer.weights_proj.weight", "self_attn.core_attention.indexer.wk.weight",
        "self_attn.core_attention.indexer.wq_b.weight", "self_attn.core_attention.softmax_offset", "self_attn.gate_proj.weight",
        "self_attn.kv_a_layernorm.weight", "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_a_layernorm.weight", "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.vha_postmix_U",
        "self_attn.vha_postmix_V", "shared_head.norm.weight", "transformer_layer.mlp_hyper_connection.alpha_post",
        "transformer_layer.mlp_hyper_connection.alpha_pre", "transformer_layer.mlp_hyper_connection.alpha_res",
        "transformer_layer.mlp_hyper_connection.bias", "transformer_layer.mlp_hyper_connection.mapping_proj.weight",
        "transformer_layer.self_attention_hyper_connection.alpha_post", "transformer_layer.self_attention_hyper_connection.alpha_pre",
        "transformer_layer.self_attention_hyper_connection.alpha_res", "transformer_layer.self_attention_hyper_connection.bias",
        "transformer_layer.self_attention_hyper_connection.mapping_proj.weight",
    ];

    private static readonly int[] Prompt = [1, 2, 3, 4, 5, 6];
    private static readonly int[] Next = [2, 3, 4, 5, 6, 7];
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-ernie5-mtp-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void Mtp_keys_match_the_real_checkpoint_layer_43()
    {
        string ratios = "[" + string.Join(", ", Enumerable.Repeat("2", 43)) + ", -2]";
        Ernie5Config config = Ernie5Config.Parse(Ernie5SyntheticCheckpoint.ConfigJson(new Dictionary<string, string>
        {
            ["num_hidden_layers"] = "43", ["compress_ratios"] = ratios, ["use_gated_attn"] = "true", ["use_vha_attention"] = "true",
            ["moe_split_feature_routing"] = "true", ["routed_scaling_factor_learnable"] = "true",
        }));

        SortedSet<string> leaves = new(StringComparer.Ordinal);
        foreach (string key in Ernie5WeightMap.ExpectedMtpKeys(config))
        {
            Assert.StartsWith("model.layers.43.", key);
            leaves.Add(Regex.Replace(key["model.layers.43.".Length..], @"experts\.\d+\.", "experts.N."));
        }

        Assert.Equal(RealMtpLeaves, leaves);
        Assert.DoesNotContain(Ernie5WeightMap.ExpectedKeys(config), k => k.StartsWith("model.layers.43.", StringComparison.Ordinal));
    }

    [Fact]
    public void Mtp_keys_map_only_when_the_layer_is_loaded()
    {
        string dir = Write("coverage");
        Ernie5Config config = Ernie5Config.Load(Path.Combine(dir, "config.json"));
        using Ernie5CheckpointWeights weights = Ernie5CheckpointWeights.Open(dir);
        int main = Ernie5WeightMap.ExpectedKeys(config).Count, mtp = Ernie5WeightMap.ExpectedMtpKeys(config).Count;

        Ernie5WeightCoverage skipped = Ernie5WeightMap.Check(config, weights.Names);
        Ernie5WeightCoverage loaded = Ernie5WeightMap.Check(config, weights.Names, mtp: true);

        skipped.RequireComplete();
        loaded.RequireComplete();
        Assert.Equal((main, mtp), (skipped.Mapped, skipped.SkippedMtp));
        Assert.Equal((main + mtp, 0), (loaded.Mapped, loaded.SkippedMtp));
    }

    [Fact]
    public void Model_loaded_without_mtp_skips_its_keys_and_refuses_a_draft()
    {
        using CpuBackend backend = new();
        using Ernie5Model model = Ernie5Model.Load(Write("off"), backend, new Ernie5LoadOptions { MaxSequenceTokens = 16 });
        using Ernie5SequenceState state = model.CreateState(16);
        float[] stream = model.ForwardStream(Prompt, state);

        Assert.True(model.Coverage.SkippedMtp > 0);
        Assert.False(state.HasMtp);
        Assert.Throws<InvalidOperationException>(() => model.MtpForward(Next, stream, state));
        Assert.Throws<InvalidOperationException>(() => model.MtpHidden(stream));
    }

    [Fact]
    public void Mtp_draft_is_the_same_resident_or_rebuilt_per_call()
    {
        string dir = Write("residency");
        using CpuBackend backend = new();
        (float[] Stream, float[] Logits) Draft(bool resident)
        {
            using Ernie5Model model = Ernie5Model.Load(dir, backend, Options(resident));
            using Ernie5SequenceState state = model.CreateState(16);
            float[] mtp = model.MtpForward(Next, model.ForwardStream(Prompt, state), state);
            Assert.Equal(Prompt.Length, state.MtpLength);
            return (mtp, model.Logits(model.MtpHidden(mtp), Next.Length));
        }

        (float[] stream, float[] logits) = Draft(resident: true);
        (float[] streamAgain, float[] logitsAgain) = Draft(resident: false);

        Assert.Equal(Next.Length * Ernie5SyntheticCheckpoint.Hc * Ernie5SyntheticCheckpoint.Hidden, stream.Length);
        Assert.Equal(Next.Length * Ernie5SyntheticCheckpoint.Vocab, logits.Length);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(stream, streamAgain);
        Assert.Equal(logits, logitsAgain);
    }

    [Fact]
    public void Mtp_appended_one_position_at_a_time_matches_one_batch()
    {
        using CpuBackend backend = new();
        using Ernie5Model model = Ernie5Model.Load(Write("incremental"), backend, Options(resident: true));
        int flat = Ernie5SyntheticCheckpoint.Hc * Ernie5SyntheticCheckpoint.Hidden;

        using Ernie5SequenceState batchState = model.CreateState(16);
        float[] batch = model.MtpForward(Next, model.ForwardStream(Prompt, batchState), batchState);

        using Ernie5SequenceState stepState = model.CreateState(16);
        float[] target = model.ForwardStream(Prompt, stepState);
        for (int i = 0; i < Next.Length; i++)
        {
            float[] row = model.MtpForward(Next.AsSpan(i, 1), target.AsSpan(i * flat, flat), stepState);
            for (int j = 0; j < flat; j++)
                Assert.True(Math.Abs(batch[i * flat + j] - row[j]) <= 1e-4f,
                    string.Create(CultureInfo.InvariantCulture, $"position {i} value {j}: {batch[i * flat + j]} vs {row[j]}"));
        }
        Assert.Equal(Next.Length, stepState.MtpLength);
    }

    [Theory]
    [InlineData(Ernie5WeightNames.MtpEmbedProj, true)]
    [InlineData(Ernie5WeightNames.MtpHiddenProj, false)]
    public void Mtp_draft_reads_both_the_target_stream_and_the_next_token(string zeroedLeaf, bool tokensIgnored)
    {
        string dir = Write("inputs");
        Ernie5Config config = Ernie5Config.Load(Path.Combine(dir, "config.json"));
        int[] otherNext = [9, 8, 7, 6, 5, 4];
        using CpuBackend backend = new();
        using Ernie5CheckpointWeights weights = Ernie5CheckpointWeights.Open(dir);

        float[] Draft(IErnie5WeightSource source, ReadOnlySpan<int> next, bool shiftStream)
        {
            using Ernie5Model model = new(backend, config, source, Options(resident: false));
            using Ernie5SequenceState state = model.CreateState(16);
            float[] target = model.ForwardStream(Prompt, state);
            if (shiftStream) for (int i = 0; i < target.Length; i++) target[i] += 0.5f;
            return model.MtpForward(next, target, state);
        }

        // with every weight, changing either input moves the draft
        Assert.NotEqual(Draft(weights, Next, false), Draft(weights, otherNext, false));
        Assert.NotEqual(Draft(weights, Next, false), Draft(weights, Next, true));

        // zeroing e_proj cuts the token out of the draft; zeroing h_proj cuts the target stream out
        ZeroedSource zeroed = new(weights, Ernie5WeightNames.Mtp(config.NumHiddenLayers, zeroedLeaf));
        float[] baseline = Draft(zeroed, Next, false);
        float[] varied = tokensIgnored ? Draft(zeroed, otherNext, false) : Draft(zeroed, Next, true);
        Assert.Equal(baseline, varied);
    }

    private static Ernie5LoadOptions Options(bool resident) => new() { MaxSequenceTokens = 16, ResidentWeights = resident, Mtp = true };

    private string Write(string name)
    {
        string dir = Path.Combine(_root, name);
        Ernie5SyntheticCheckpoint.Write(dir);
        return dir;
    }

    // Serves every weight from the checkpoint except one, which reads as zeros of the same shape.
    private sealed class ZeroedSource(IErnie5WeightSource inner, string zeroed) : IErnie5WeightSource
    {
        public IEnumerable<string> Names => inner.Names;

        public Tensor Read(string name)
        {
            Tensor t = inner.Read(name);
            if (!string.Equals(name, zeroed, StringComparison.Ordinal)) return t;
            using (t)
            {
                Tensor zero = new(t.Shape, DType.F32);
                zero.AsSpan<float>().Clear();
                return zero;
            }
        }
    }
}
