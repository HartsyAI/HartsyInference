using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using static HartsyInference.Cpu.Tests.Attention.AttentionFixtures;

namespace HartsyInference.Cpu.Tests.Attention;

public sealed class IndexerScoresReferenceTests
{
    private static float[] Run(out System.Text.Json.JsonElement f, byte[]? candidates = null)
    {
        f = Load("indexer_scores.json");
        int dim = f.GetProperty("dim").GetInt32(), heads = f.GetProperty("heads").GetInt32();
        int tokens = f.GetProperty("tokens").GetInt32(), n = f.GetProperty("keys").GetInt32();
        LatentSource keys = Source(LatentEncoding.Fp4E2M1E8M0x32, Bytes(f.GetProperty("keyCodes")), Bytes(f.GetProperty("keyScales")), n, dim);
        try
        {
            using Tensor q = F32(Floats(f.GetProperty("query")), tokens, heads, dim);
            using Tensor w = F32(Floats(f.GetProperty("headWeights")), tokens, heads);
            using Tensor lens = I32(Ints(f.GetProperty("compressLens")), tokens);
            using Tensor? cand = candidates is null ? null : U8(candidates, tokens, n);
            using Tensor scores = Empty(DType.F32, tokens, n);
            using CpuBackend cpu = new();
            cpu.IndexerScores(scores, q, keys, w, lens, cand, 1f);
            return ReadF32(scores);
        }
        finally { Dispose(keys); }
    }

    [Fact]
    public void Matches_The_Upstream_Indexer_Expression_And_Masks_Mid_Group_Queries()
    {
        float[] actual = Run(out System.Text.Json.JsonElement f);
        int[] masked = Ints(f.GetProperty("masked"));
        float[] expected = Floats(f.GetProperty("scores"));
        Assert.Contains(masked, m => m == 1);
        for (int i = 0; i < actual.Length; i++)
        {
            if (masked[i] == 1) Assert.True(float.IsNegativeInfinity(actual[i]), $"score {i} must be masked");
            else Assert.True(MathF.Abs(actual[i] - expected[i]) < 1e-5f, $"score {i}: {actual[i]} vs {expected[i]}");
        }
    }

    [Fact]
    public void A_Zero_Candidate_Flag_Masks_An_Otherwise_Visible_Key()
    {
        System.Text.Json.JsonElement f = Load("indexer_scores.json");
        int tokens = f.GetProperty("tokens").GetInt32(), n = f.GetProperty("keys").GetInt32();
        byte[] cand = Enumerable.Repeat((byte)1, tokens * n).ToArray();
        cand[(tokens - 1) * n] = 0;
        float[] actual = Run(out _, cand);
        Assert.True(float.IsNegativeInfinity(actual[(tokens - 1) * n]));
        Assert.True(float.IsFinite(actual[(tokens - 1) * n + 1]));
    }

    [Fact]
    public void Invalid_Operands_Throw()
    {
        using Tensor q = F32(new float[64], 1, 1, 64), w = F32(new float[1], 1, 1), s = Empty(DType.F32, 1, 0);
        using Tensor badLens = F32(new float[1], 1);
        using CpuBackend cpu = new();
        Assert.Throws<ArgumentException>(() => cpu.IndexerScores(s, q, LatentSource.Empty, w, badLens, null, 1f));
    }
}
