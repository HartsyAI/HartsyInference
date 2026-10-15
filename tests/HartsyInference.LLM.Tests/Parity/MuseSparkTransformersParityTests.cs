using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Models.MuseSpark;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>A tiny random Muse Spark run through transformers' <c>MuseSparkForConditionalGeneration</c>
/// (<c>tests/python-reference/dump_muse_spark_tiny_reference.py</c>) must match <see cref="GenericTransformer"/> loaded through
/// <see cref="MuseSparkCheckpointMapper"/> from the same HF-keyed weights. The fixture keeps every block trait of the released
/// checkpoint: three sliding RoPE layers with a 6-token window and one NoPE full-attention layer, so the 12-token prompt crosses the
/// window, plus the gated attention output, residual gates and the latent MoE with a selection-only router bias.</summary>
public sealed unsafe class MuseSparkTransformersParityTests
{
    private const float Tolerance = 2e-4f;
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "MuseSpark");

    [Fact]
    public void Prefill_LogitsMatchTransformers()
    {
        using Fixture f = new();
        using KvCache cache = f.NewCache();
        using Tensor hidden = f.Transformer.Forward(f.Backend, f.Ids, 0, cache);
        using Tensor logits = f.Transformer.ProjectLogits(f.Backend, hidden, f.Ids.Length);
        AssertRowsMatch(f, logits.AsReadOnlySpan<float>(), 0, f.Ids.Length);
    }

    [Fact]
    public void PrefillThenDecode_LogitsMatchTransformers()
    {
        using Fixture f = new();
        using KvCache cache = f.NewCache();
        const int prefill = 5;
        using (Tensor hidden = f.Transformer.Forward(f.Backend, f.Ids.AsSpan(0, prefill), 0, cache))
        using (Tensor logits = f.Transformer.ProjectLogits(f.Backend, hidden, prefill))
            AssertRowsMatch(f, logits.AsReadOnlySpan<float>(), 0, prefill);
        for (int pos = prefill; pos < f.Ids.Length; pos++)
        {
            using Tensor hidden = f.Transformer.Forward(f.Backend, f.Ids.AsSpan(pos, 1), pos, cache);
            using Tensor logits = f.Transformer.ProjectLogits(f.Backend, hidden, 1);
            AssertRowsMatch(f, logits.AsReadOnlySpan<float>(), pos, 1);
        }
    }

    [Fact]
    public void BatchDecode_LogitsMatchTransformers()
    {
        using Fixture f = new();
        int[] prefixes = [7, 10];
        KvCache[] caches = [f.NewCache(), f.NewCache()];
        try
        {
            for (int s = 0; s < prefixes.Length; s++)
            {
                using Tensor prefill = f.Transformer.Forward(f.Backend, f.Ids.AsSpan(0, prefixes[s]), 0, caches[s]);
            }
            int[] tokens = [f.Ids[prefixes[0]], f.Ids[prefixes[1]]];
            using Tensor embeds = new(new TensorShape(1, tokens.Length, f.Config.Transformer.HiddenSize), DType.F32);
            f.Transformer.EmbedLookup(embeds, tokens);
            using Tensor hidden = f.Transformer.ForwardBatchDecode(f.Backend, embeds, prefixes, [caches[0], caches[1]]);
            using Tensor logits = f.Transformer.ProjectLogits(f.Backend, hidden, tokens.Length);
            int vocab = f.Config.Transformer.VocabSize;
            for (int s = 0; s < prefixes.Length; s++)
                AssertRowsMatch(f, logits.AsReadOnlySpan<float>().Slice(s * vocab, vocab), prefixes[s], 1);
        }
        finally
        {
            foreach (KvCache c in caches) c.Dispose();
        }
    }

    [Fact]
    public void Config_RefusesRopeOnFullAttentionLayer()
    {
        string json = File.ReadAllText(Path.Combine(Dir, "muse_spark_tiny_config.json"))
            .Replace("\"full_attention\"", "\"sliding_attention\"", StringComparison.Ordinal);
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => MuseSparkHfConfig.FromJson(json));
        Assert.Contains("RoPE", ex.Message, StringComparison.Ordinal);
    }

    private static void AssertRowsMatch(Fixture f, ReadOnlySpan<float> got, int firstRow, int rows)
    {
        int vocab = f.Config.Transformer.VocabSize;
        ReadOnlySpan<float> want = f.Logits.Slice(firstRow * vocab, rows * vocab);
        Assert.Equal(want.Length, got.Length);
        float worst = 0f;
        int worstAt = 0;
        for (int i = 0; i < want.Length; i++)
        {
            float d = MathF.Abs(want[i] - got[i]);
            if (d > worst) { worst = d; worstAt = i; }
        }
        Assert.True(worst < Tolerance,
            $"rows [{firstRow}, {firstRow + rows}): max |Δ| = {worst} at row {firstRow + worstAt / vocab}, vocab {worstAt % vocab}");
    }

    /// <summary>The fixture's weights mapped and loaded on the CPU backend, with the reference ids and logits.</summary>
    private sealed class Fixture : IDisposable
    {
        private readonly SafeTensorsLoader _weights = new();
        private readonly SafeTensorsLoader _expected = new();
        private readonly Tensor _logits;

        public Fixture()
        {
            _weights.Load(Path.Combine(Dir, "muse_spark_tiny.safetensors"));
            _expected.Load(Path.Combine(Dir, "muse_spark_tiny_expected.safetensors"));
            Config = MuseSparkHfConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "muse_spark_tiny_config.json")));
            Dictionary<string, Tensor> hf = new(StringComparer.Ordinal);
            foreach (string name in _weights.Descriptors.Keys) hf[name] = _weights.GetTensor(name);
            Transformer = new GenericTransformer(Config.Transformer);
            Transformer.LoadWeights(MuseSparkCheckpointMapper.Map(hf, Config), "model");
            Ids = _expected.GetTensor("input_ids").AsReadOnlySpan<int>().ToArray();
            _logits = _expected.GetTensor("logits");
        }

        public MuseSparkHfConfig Config { get; }

        public GenericTransformer Transformer { get; }

        public CpuBackend Backend { get; } = new();

        public int[] Ids { get; }

        public ReadOnlySpan<float> Logits => _logits.AsReadOnlySpan<float>();

        public KvCache NewCache() => new(Config.Transformer.NumLayers, 1, Config.Transformer.NumKvHeads, Config.Transformer.HeadDim);

        public void Dispose()
        {
            Transformer.Dispose();
            Backend.Dispose();
            _weights.Dispose();
            _expected.Dispose();
        }
    }
}
