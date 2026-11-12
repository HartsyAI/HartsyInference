using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Transformer;
using Xunit;
using TensorSpec = HartsyInference.LLM.Tests.Kimi.KimiSyntheticCheckpoint.TensorSpec;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>Each rank of a sharded checkpoint loads only its own share (its heads, vocabulary rows, FFN columns and routed experts), and the
/// sharded decoder's logits match the single-rank decoder on the same checkpoint. Runs on virtual CPU ranks; no GPU.</summary>
public sealed class KimiK31ShardedLoadTests : IDisposable
{
    private const float Tol = 1e-4f;

    private static readonly int[] Tokens = [4, 4, 9, 0, 13, 7];

    private static readonly KimiK31Config Config = KimiSyntheticCheckpoint.ShardedConfig;

    // The identity policy keeps the stored BF16 on the CPU, where the GEMM casts each weight per call: the BF16 storage path runs through the model.
    private static readonly Func<Tensor, string, Tensor> KeepStoredDType = (tensor, _) => tensor;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-kimi-shard-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void PackedExperts_ShardedCheckpoint_MatchesTheSingleRankLogits(int ranks)
    {
        KimiSyntheticCheckpoint.WriteSharded(_dir, Config, shardFiles: 3, packedExperts: true, bf16Dense: false);

        using KimiK31Checkpoint single = KimiK31Checkpoint.Load(_dir, new CpuBackend(), maxPositions: 16);
        float[] reference = Forward(single.Model, single.Config, ranks: 1);

        List<IBackend> backends = Backends(ranks);
        try
        {
            using KimiK31Checkpoint sharded = KimiK31Checkpoint.Load(_dir, backends, new HostStagedComm(ranks), maxPositions: 16);
            AssertClose(reference, Forward(sharded.Model, sharded.Config, ranks));
        }
        finally
        {
            DisposeAll(backends);
        }
    }

    [Fact]
    public void EightRanks_PackedExperts_ShardedCheckpoint_MatchesTheSingleRankLogits()
    {
        KimiK31Config config = KimiSyntheticCheckpoint.ShardedEightConfig;
        KimiSyntheticCheckpoint.WriteSharded(_dir, config, shardFiles: 4, packedExperts: true, bf16Dense: true);

        using KimiK31Checkpoint single = KimiK31Checkpoint.Load(_dir, new CpuBackend(), maxPositions: 16);
        float[] reference = Forward(single.Model, single.Config, ranks: 1);

        List<IBackend> backends = Backends(8);
        try
        {
            using KimiK31Checkpoint sharded = KimiK31Checkpoint.Load(_dir, backends, new HostStagedComm(8), maxPositions: 16);
            AssertClose(reference, Forward(sharded.Model, sharded.Config, ranks: 8));
        }
        finally
        {
            DisposeAll(backends);
        }
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void Bf16Dense_ShardedLoad_MatchesTheSingleRankLogits(int ranks)
    {
        KimiSyntheticCheckpoint.WriteSharded(_dir, Config, shardFiles: 2, packedExperts: true, bf16Dense: true);
        KimiK31Config parsed = ParsedConfig();
        using KimiK31CheckpointWeights source = KimiK31CheckpointWeights.Open(_dir);

        float[] reference = SingleRank(source, parsed);
        float[] sharded = Sharded(source, parsed, ranks);

        AssertClose(reference, sharded);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void EachRank_LoadsOnlyItsOwnSlice(int ranks)
    {
        KimiSyntheticCheckpoint.WriteSharded(_dir, Config, shardFiles: 3, packedExperts: true, bf16Dense: true);
        KimiK31Config parsed = ParsedConfig();
        List<TensorSpec> specs = KimiSyntheticCheckpoint.Specs(Config, packedExperts: true, bf16Dense: true);
        long wholeBytes = specs.Sum(spec => Bytes(spec));
        using KimiK31CheckpointWeights inner = KimiK31CheckpointWeights.Open(_dir);
        CountingSource source = new(inner);
        int perRank = Config.Experts / ranks;

        for (int r = 0; r < ranks; r++)
        {
            source.Reset();
            KimiK31Weights weights = KimiK31WeightLoader.Load(parsed, source, new KimiK31Shard(r, ranks), KeepStoredDType);
            try
            {
                Assert.Equal(ExpectedBytes(specs, r, ranks), source.Bytes);
                Assert.True(source.Bytes < wholeBytes, $"rank {r} read {source.Bytes} bytes of {wholeBytes}");
                Assert.All(source.Names.Where(name => name.Contains(".experts.", StringComparison.Ordinal)),
                    name => Assert.InRange(ExpertIndex(name), r * perRank, (r + 1) * perRank - 1));
            }
            finally
            {
                foreach (Tensor t in weights.Owned) t.Dispose();
            }
        }
    }

    private static float[] Forward(KimiK31Model model, KimiK31Config config, int ranks)
    {
        using KimiK31State state = new(config, capacity: 16, ranks);
        return model.Forward(Tokens, state);
    }

    private KimiK31Config ParsedConfig() => KimiK31HfConfig.Parse(File.ReadAllText(Path.Combine(_dir, "config.json")));

    private static float[] SingleRank(KimiK31CheckpointWeights source, KimiK31Config config)
    {
        KimiK31Weights weights = KimiK31WeightLoader.Load(config, source, KimiK31Shard.Whole, KeepStoredDType);
        try
        {
            using CpuBackend backend = new();
            using KimiK31Model model = new(backend, config, weights, maxPositions: 16);
            return Forward(model, config, ranks: 1);
        }
        finally
        {
            foreach (Tensor t in weights.Owned) t.Dispose();
        }
    }

    private static float[] Sharded(KimiK31CheckpointWeights source, KimiK31Config config, int ranks)
    {
        List<IBackend> backends = Backends(ranks);
        List<KimiK31Weights> loaded = new(ranks);
        try
        {
            for (int r = 0; r < ranks; r++) loaded.Add(KimiK31WeightLoader.Load(config, source, new KimiK31Shard(r, ranks), KeepStoredDType));
            using KimiK31Model model = new(new TpPlacement(backends, new HostStagedComm(ranks)), config, loaded, maxPositions: 16);
            return Forward(model, config, ranks);
        }
        finally
        {
            foreach (KimiK31Weights rank in loaded) foreach (Tensor t in rank.Owned) t.Dispose();
            DisposeAll(backends);
        }
    }

    private static List<IBackend> Backends(int ranks)
    {
        List<IBackend> backends = new(ranks);
        for (int r = 0; r < ranks; r++) backends.Add(new CpuBackend());
        return backends;
    }

    private static void DisposeAll(List<IBackend> backends)
    {
        foreach (IBackend backend in backends) backend.Dispose();
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], Tol);
    }

    // What rank r reads: a sharded tensor's share, a replicated tensor whole, and the routed experts of its range.
    private static long ExpectedBytes(List<TensorSpec> specs, int rank, int ranks)
    {
        int perRank = Config.Experts / ranks;
        long total = 0;
        foreach (TensorSpec spec in specs)
        {
            long bytes = Bytes(spec);
            if (spec.Name.Contains(".experts.", StringComparison.Ordinal))
            {
                if (ExpertIndex(spec.Name) / perRank == rank) total += bytes;
            }
            else
            {
                total += IsSharded(spec.Name) ? bytes / ranks : bytes;
            }
        }
        return total;
    }

    private static long Bytes(TensorSpec spec)
    {
        long elements = 1;
        foreach (long d in spec.Shape) elements *= d;
        return elements * KimiSyntheticCheckpoint.ElementBytes(spec.Dtype);
    }

    private static int ExpertIndex(string name) => int.Parse(name.Split(".experts.")[1].Split('.')[0], System.Globalization.CultureInfo.InvariantCulture);

    // The tensors the loader splits by rows (dim 0) or columns (dim 1); every other tensor is replicated.
    private static bool IsSharded(string name)
    {
        string[] rows =
        [
            "embed_tokens.weight", "language_model.lm_head.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight", "self_attn.dt_bias", "self_attn.f_b_proj.weight",
            "self_attn.b_proj.weight", "self_attn.g_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_b_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "shared_experts.gate_proj.weight", "shared_experts.up_proj.weight",
        ];
        string[] columns = ["self_attn.o_proj.weight", "mlp.down_proj.weight", "shared_experts.down_proj.weight"];
        return rows.Any(suffix => name.EndsWith(suffix, StringComparison.Ordinal)) || columns.Any(suffix => name.EndsWith(suffix, StringComparison.Ordinal));
    }

    // Counts the bytes every read returns, so a test can assert what a rank loaded.
    private sealed class CountingSource(IKimiK31RankWeightSource inner) : IKimiK31RankWeightSource
    {
        public long Bytes { get; private set; }

        public List<string> Names { get; } = [];

        public bool Has(string name) => inner.Has(name);

        public Tensor Read(string name)
        {
            Tensor t = inner.Read(name);
            Count(name, t);
            return t;
        }

        public Tensor ReadSlice(string name, int dim, long start, long count)
        {
            Tensor t = inner.ReadSlice(name, dim, start, count);
            Count(name, t);
            return t;
        }

        public void Reset()
        {
            Bytes = 0;
            Names.Clear();
        }

        private void Count(string name, Tensor t)
        {
            Bytes += t.DType.ComputeByteCount(t.ElementCount);
            Names.Add(name);
        }
    }
}
