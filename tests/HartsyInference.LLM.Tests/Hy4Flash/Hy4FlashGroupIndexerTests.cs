using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Hy4Flash;
using HartsyInference.LLM.Transformer.Dsa;

namespace HartsyInference.LLM.Tests.Hy4Flash;

/// <summary>The Hy4 Flash group indexer against a NumPy transcription of the reference (numpy_group_indexer.py, not the Hugging Face class): 37 cached keys
/// over an index_topk of 16, so selection is exercised, with a length that is not a multiple of the group size, fed in one chunk and in two.</summary>
public sealed class Hy4FlashGroupIndexerTests
{
    private static JsonElement Case()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "Hy4Flash", "group_indexer_case.json")));
        return document.RootElement.Clone();
    }

    private static float[] Floats(JsonElement e) => [.. e.EnumerateArray().Select(static v => (float)v.GetDouble())];

    private static Tensor Make(JsonElement e, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        Floats(e).CopyTo(t.AsSpan<float>());
        return t;
    }

    private sealed class Fixture : IDisposable
    {
        public readonly JsonElement Json = Case();
        public readonly float[] X, QLatent;
        public readonly int Dim, QRank, Length;
        public readonly Hy4FlashGroupIndexer Indexer;
        private readonly Hy4FlashIndexerWeights _weights;

        public Fixture(IBackend backend)
        {
            Dim = Json.GetProperty("dim").GetInt32();
            QRank = Json.GetProperty("q_lora_rank").GetInt32();
            Length = Json.GetProperty("length").GetInt32();
            int heads = Json.GetProperty("heads").GetInt32(), hd = Json.GetProperty("head_dim").GetInt32(), kpool = Json.GetProperty("kpool").GetInt32();
            JsonElement w = Json.GetProperty("weights");
            _weights = new Hy4FlashIndexerWeights
            {
                QueryProjection = Make(w.GetProperty("query_projection"), heads * hd, QRank),
                HeadWeightProjection = Make(w.GetProperty("head_weight_projection"), heads, Dim),
                TokenKeyProjection = Make(w.GetProperty("token_key_projection"), hd, Dim),
                TokenKeyNorm = Make(w.GetProperty("token_key_norm"), hd),
                KeyValueProjection = Make(w.GetProperty("key_value_projection"), hd, hd),
                KeyGateProjection = Make(w.GetProperty("key_gate_projection"), hd, hd),
                IntraGroupApe = Make(w.GetProperty("intra_group_ape"), kpool, hd),
                KeyNorm = Make(w.GetProperty("key_norm"), hd),
            };
            Hy4FlashIndexerSettings settings = new()
            {
                Heads = heads, HeadDim = hd, RopeDim = Json.GetProperty("rope_dim").GetInt32(), Kpool = kpool,
                TopK = Json.GetProperty("topk").GetInt32(), NormEps = (float)Json.GetProperty("norm_eps").GetDouble(),
            };
            Indexer = new Hy4FlashGroupIndexer(backend, settings, _weights,
                Hy4FlashPlainRope.Create(settings.RopeDim / 2, 256, Json.GetProperty("theta").GetDouble()), Dim, QRank);
            X = Floats(Json.GetProperty("x"));
            QLatent = Floats(Json.GetProperty("q_latent"));
        }

        public void Dispose()
        {
            Indexer.Dispose();
            foreach (Tensor t in new[] { _weights.QueryProjection, _weights.HeadWeightProjection, _weights.TokenKeyProjection, _weights.TokenKeyNorm,
                _weights.KeyValueProjection, _weights.KeyGateProjection, _weights.IntraGroupApe, _weights.KeyNorm }) t.Dispose();
        }
    }

    private static int[][] Rows(DsaKeySelection s) =>
        [.. Enumerable.Range(0, s.Counts.Length).Select(t => s.Rows.AsSpan(t * s.Width, s.Counts[t]).ToArray())];

    private static int[][] Expected(JsonElement e) => [.. e.EnumerateArray().Select(static r => r.EnumerateArray().Select(static v => v.GetInt32()).ToArray())];

    [Fact]
    public void TokenKeys_MatchTheTranscription()
    {
        using Fixture f = new(new CpuBackend());
        int hd = f.Indexer.KeyWidth;
        using Tensor store = new(new TensorShape(64, hd), DType.F32);
        f.Indexer.Append(f.X, f.Length, 0, store);
        float[] expected = Floats(f.Json.GetProperty("token_keys"));
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], store.AsReadOnlySpan<float>()[i], 4);
    }

    [Fact]
    public void LongContext_RecallsTheSameKeysAsTheTranscription()
    {
        using Fixture f = new(new CpuBackend());
        int hd = f.Indexer.KeyWidth, d = f.Dim, q = f.QRank, n = f.Length;
        using Tensor store = new(new TensorShape(64, hd), DType.F32);
        f.Indexer.Append(f.X, n, 0, store);
        int[] positions = [.. Enumerable.Range(0, n)];
        DsaKeySelection selection = f.Indexer.Select(f.X, f.QLatent, positions, n, store);
        int[][] expected = Expected(f.Json.GetProperty("keep_long"));
        Assert.Equal(expected.Length, selection.Counts.Length);
        int[][] actual = Rows(selection);
        for (int t = 0; t < n; t++) Assert.True(expected[t].SequenceEqual(actual[t]), $"query {t}: expected [{string.Join(",", expected[t])}], got [{string.Join(",", actual[t])}]");
        Assert.True(actual.Sum(static r => r.Length) < Enumerable.Range(0, n).Sum(static t => t + 1), "the case must actually drop keys");
        _ = (d, q);
    }

    [Fact]
    public void PrefillThenDecodeSteps_SelectLikeOneChunk()
    {
        using Fixture f = new(new CpuBackend());
        int hd = f.Indexer.KeyWidth, d = f.Dim, q = f.QRank, n = f.Length, split = 30;
        using Tensor store = new(new TensorShape(64, hd), DType.F32);
        f.Indexer.Append(f.X.AsSpan(0, split * d), split, 0, store);
        List<int[]> rows = [];
        rows.AddRange(Rows(f.Indexer.Select(f.X.AsSpan(0, split * d), f.QLatent.AsSpan(0, split * q), [.. Enumerable.Range(0, split)], split, store)));
        for (int t = split; t < n; t++)
        {
            f.Indexer.Append(f.X.AsSpan(t * d, d), 1, t, store);
            rows.AddRange(Rows(f.Indexer.Select(f.X.AsSpan(t * d, d), f.QLatent.AsSpan(t * q, q), [t], t + 1, store)));
        }
        int[][] expected = Expected(f.Json.GetProperty("keep_long"));
        for (int t = split; t < n; t++) Assert.True(expected[t].SequenceEqual(rows[t]), $"decode query {t}");
        // The first 30 positions were selected with 30 cached keys, so they use the context that existed then, not 37.
        Assert.Equal(n, rows.Count);
    }

    [Fact]
    public void AtOrBelowIndexTopk_AttentionIsDense()
    {
        using Fixture f = new(new CpuBackend());
        int hd = f.Indexer.KeyWidth, d = f.Dim, q = f.QRank;
        using Tensor store = new(new TensorShape(64, hd), DType.F32);
        f.Indexer.Append(f.X.AsSpan(0, 12 * d), 12, 0, store);
        int[][] actual = Rows(f.Indexer.Select(f.X.AsSpan(0, 12 * d), f.QLatent.AsSpan(0, 12 * q), [.. Enumerable.Range(0, 12)], 12, store));
        int[][] expected = Expected(f.Json.GetProperty("keep_dense"));
        for (int t = 0; t < 12; t++) Assert.True(expected[t].SequenceEqual(actual[t]));
    }
}
