using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Hy4Flash;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.LLM.Tests.Hy4Flash;

/// <summary>Stage checks against the Hugging Face classes of transformers PR 50903 at eb1d5db, run on a 3-layer CPU float32 config by
/// tests/python-reference/hy4_flash/dump_tiny_reference.py (see its README). The committed dump is in tests/python-reference/hy4_flash/data.</summary>
public sealed class Hy4FlashHfDumpTests
{
    private static string DataDir
    {
        get
        {
            string? dir = AppContext.BaseDirectory;
            while (dir is not null && !Directory.Exists(Path.Combine(dir, "tests", "python-reference", "hy4_flash", "data"))) dir = Path.GetDirectoryName(dir);
            return dir is null ? throw new DirectoryNotFoundException("tests/python-reference/hy4_flash/data not found above the test output.")
                : Path.Combine(dir, "tests", "python-reference", "hy4_flash", "data");
        }
    }

    private static string Npy(string name) => Path.Combine(DataDir, name + ".npy");

    [Fact]
    public void DumpedCheckpoint_MapsEveryKeyAndShapeUnderTheTinyConfig()
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([Path.Combine(DataDir, "weights.safetensors")]);
        List<Hy4FlashTensorInfo> tensors = [.. set.Inventory.Select(static kv => new Hy4FlashTensorInfo(kv.Key,
            kv.Value.DType == DType.I32 ? "I32" : "F32", [.. Enumerable.Range(0, kv.Value.Shape.Rank).Select(i => kv.Value.Shape[i])], kv.Value.ByteLength))];
        Hy4FlashWeightMap.Check(config, tensors).RequireComplete();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void GroupIndexer_RecallsTheKeysTheHfIndexerKept(int layer)
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([Path.Combine(DataDir, "weights.safetensors")]);
        string p = $"model.layers.{layer}.self_attn.compressed_indexer.";
        Hy4FlashIndexerWeights weights = new()
        {
            QueryProjection = set.GetTensor(p + "query_projection.weight"),
            HeadWeightProjection = set.GetTensor(p + "head_weight_projection.weight"),
            TokenKeyProjection = set.GetTensor(p + "token_key_projection.weight"),
            TokenKeyNorm = set.GetTensor(p + "token_key_norm.weight"),
            KeyValueProjection = set.GetTensor(p + "key_value_projection.weight"),
            KeyGateProjection = set.GetTensor(p + "key_gate_projection.weight"),
            IntraGroupApe = set.GetTensor(p + "intra_group_ape"),
            KeyNorm = set.GetTensor(p + "key_norm.weight"),
        };
        Hy4FlashIndexerSettings settings = new()
        {
            Heads = config.IndexNHeads, HeadDim = config.IndexHeadDim, RopeDim = config.QkRopeHeadDim, Kpool = config.IndexKpool,
            TopK = config.IndexTopk, NormEps = (float)config.RmsNormEps,
        };
        int dim = config.HiddenSize;
        (long[] hcShape, float[] hcIn) = NpyReader.ReadFloat32(Npy($"text__layer_{layer}_attn_hc_in"));
        int tokens = (int)hcShape[1];
        float[] x = RmsNorm(hcIn, tokens, dim, set.GetTensor($"model.layers.{layer}.input_layernorm.weight"), (float)config.RmsNormEps);
        (_, float[] qLatent) = NpyReader.ReadFloat32(Npy($"text__layer_{layer}_q_latent"));
        (long[] keepShape, bool[] keep) = NpyReader.ReadBool(Npy($"text__indexer_{layer}_keep"));
        Assert.Equal([1L, 1, tokens, tokens], keepShape);

        IBackend backend = new CpuBackend();
        using Hy4FlashGroupIndexer indexer = new(backend, settings, weights,
            new Hy4FlashRopeRows(settings.RopeDim / 2, 256, config.IndexedRopeTheta), dim, config.QLoraRank);
        using Tensor store = new(new TensorShape(64, settings.HeadDim), DType.F32);
        indexer.Append(x, tokens, 0, store);

        (_, float[] hfTokenKeys) = NpyReader.ReadFloat32(Npy($"text__indexer_{layer}_token_keys"));
        for (int i = 0; i < hfTokenKeys.Length; i++) Assert.Equal(hfTokenKeys[i], store.AsReadOnlySpan<float>()[i], 4);

        int[] positions = [.. Enumerable.Range(0, tokens)];
        DsaKeySelection selection = indexer.Select(x, qLatent, positions, tokens, store);
        float[] scores = indexer.Scores(x, qLatent, positions, tokens, store, out int groups);
        List<string> differing = [];
        int tied = 0;
        for (int t = 0; t < tokens; t++)
        {
            int[] expected = [.. Enumerable.Range(0, tokens).Where(k => keep[t * tokens + k])];
            int[] actual = selection.Rows.AsSpan(t * selection.Width, selection.Counts[t]).ToArray();
            if (expected.SequenceEqual(actual)) continue;
            // torch top-k leaves ties undefined: a different recall is acceptable only when the recalled groups score exactly alike.
            int own = t / settings.Kpool;
            float[] hf = [.. expected.Select(k => k / settings.Kpool).Distinct().Where(g => g != own).Select(g => scores[t * groups + g]).OrderDescending()];
            float[] mine = [.. actual.Select(k => k / settings.Kpool).Distinct().Where(g => g != own).Select(g => scores[t * groups + g]).OrderDescending()];
            if (hf.Length == mine.Length && hf.Zip(mine).All(static p => MathF.Abs(p.First - p.Second) <= 1e-7f)) tied++;
            else differing.Add($"query {t}: HF [{string.Join(",", expected)}] C# [{string.Join(",", actual)}]");
        }
        Assert.True(differing.Count == 0, string.Join(Environment.NewLine, differing));
        Assert.True(tied <= 2, $"{tied} queries differ only by a tie; the dump should be mostly unambiguous");
        Assert.True(keep.Count(static b => b) < Enumerable.Range(0, tokens).Sum(static t => t + 1), "the dump must actually drop keys");
    }

    private static float[] RmsNorm(float[] hcIn, int tokens, int dim, Tensor gain, float eps)
    {
        float[] x = new float[tokens * dim];
        ReadOnlySpan<float> g = gain.AsReadOnlySpan<float>();
        for (int t = 0; t < tokens; t++)
        {
            double sum = 0;
            for (int i = 0; i < dim; i++) sum += (double)hcIn[t * dim + i] * hcIn[t * dim + i];
            float scale = 1f / MathF.Sqrt((float)(sum / dim) + eps);
            for (int i = 0; i < dim; i++) x[t * dim + i] = hcIn[t * dim + i] * scale * g[i];
        }
        return x;
    }
}
