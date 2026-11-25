using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Glm5Next;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;
using Xunit;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The structure of the k-pool selection: each row is whole complete pools that end at or before the query, plus the query's incomplete pool as
/// raw tokens. The scoring itself is checked against transformers through <see cref="Glm5NextReferenceParityTests"/>.</summary>
public sealed class Glm5NextKpoolIndexerTests
{
    private const int Hidden = 12, Heads = 3, HeadDim = 8, QRank = 10, Pool = 4, Tokens = 27, KeptPools = 2;

    [Fact]
    public void Rows_AreKeptCompletePoolsThatHaveEnded_PlusTheQueryTail()
    {
        Glm5NextConfig c = Config();
        Random rng = new(5);
        using Glm5NextKpoolIndexer indexer = Build(c, rng);
        float[] x = Randoms(rng, Tokens * Hidden, 1f), qLatent = Randoms(rng, Tokens * QRank, 1f);
        using Tensor store = new(new TensorShape(Tokens, indexer.KeyWidth), DType.F32);
        indexer.Append(x, Tokens, 0, store);
        int[] positions = [.. Enumerable.Range(0, Tokens)];

        DsaKeySelection s = indexer.Select(x, qLatent, positions, Tokens, store);

        for (int p = 0; p < Tokens; p++)
        {
            int[] row = s.Rows.AsSpan(p * s.Width, s.Counts[p]).ToArray();
            int tail = (p + 1) % Pool, endedPools = (p + 1) / Pool;
            int[] tailTokens = row.Where(t => t >= p + 1 - tail).ToArray();
            int[] pooled = row.Where(t => t < p + 1 - tail).ToArray();
            Assert.Equal(row.Order(), row);
            Assert.Equal(Enumerable.Range(p + 1 - tail, tail), tailTokens);
            Assert.Equal(Math.Min(KeptPools, endedPools) * Pool, pooled.Length);
            foreach (IGrouping<int, int> pool in pooled.GroupBy(t => t / Pool))
            {
                Assert.Equal(Pool, pool.Count());
                Assert.True(pool.Key * Pool + Pool - 1 <= p, $"Query {p} picked pool {pool.Key}, which has not ended.");
            }
        }
        Assert.True(s.Counts[Tokens - 1] < Tokens, "The last query must not see every token, or the test selects nothing.");
    }

    [Fact]
    public void LastSlotOfAPool_KeepsItsOwnCompletedPool_WhenTheTopKHasRoomForIt()
    {
        // A query on the last slot of a pool has no tail, and its own pool has just ended. The reference (get_visible_tokens on the pool's last
        // token) makes that pool selectable, and it is taken by score like any other, so with room for every pool it must be in the row.
        Glm5NextConfig c = Config() with { IndexTopK = 1000 };
        Random rng = new(5);
        using Glm5NextKpoolIndexer indexer = Build(c, rng);
        float[] x = Randoms(rng, Tokens * Hidden, 1f), qLatent = Randoms(rng, Tokens * QRank, 1f);
        using Tensor store = new(new TensorShape(Tokens, indexer.KeyWidth), DType.F32);
        indexer.Append(x, Tokens, 0, store);
        int[] positions = [.. Enumerable.Range(0, Tokens)];

        DsaKeySelection s = indexer.Select(x, qLatent, positions, Tokens, store);

        int p = 7;
        int[] row = s.Rows.AsSpan(p * s.Width, s.Counts[p]).ToArray();
        Assert.Equal(Enumerable.Range(0, p + 1), row.Order());
    }

    private static Glm5NextConfig Config()
    {
        string json = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next", "tiny", "config.json"));
        return Glm5NextHfConfig.Parse(json) with
        {
            Hidden = Hidden, QLoraRank = QRank, IndexHeads = Heads, IndexHeadDim = HeadDim, IndexPool = Pool, IndexTopK = KeptPools * Pool,
        };
    }

    private static Glm5NextKpoolIndexer Build(Glm5NextConfig c, Random rng) => new(new CpuBackend(), c,
        Matrix(rng, Heads * HeadDim, QRank), Matrix(rng, HeadDim, Hidden), Vector(HeadDim, 1f), Vector(HeadDim, 0f), Matrix(rng, Heads, Hidden),
        Matrix(rng, HeadDim, Hidden), Matrix(rng, Pool, HeadDim));

    private static Tensor Matrix(Random rng, int rows, int cols) => HostMath.Tensor(Randoms(rng, rows * cols, 0.6f), rows, cols);

    private static Tensor Vector(int n, float value) => HostMath.Tensor(Enumerable.Repeat(value, n).ToArray(), n);

    private static float[] Randoms(Random rng, int n, float amp) => [.. Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1) * amp)];
}
