using System.Text.Json.Nodes;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.LLM.Maxwell;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The engine's Maxwell decoder against the release's own modeling code on a tiny float32 checkpoint (Fixtures/Maxwell, written by
/// tools/maxwell-reference/dump_tiny.py with transformers 4.46.3). The geometry is shrunk so remote block selection fires from position 24: a 40-token
/// prefill, a 30-token prefill followed by ten cached decode steps, per-layer residual streams and the final hidden rows.</summary>
public sealed class MaxwellTinyParityTests(ITestOutputHelper output)
{
    private static readonly string Directory = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Maxwell");

    private static float[] Load(string name) => NpyReader.ReadFloat32(Path.Combine(Directory, name + ".npy")).Data;

    private static int[] Tokens() => NpyReader.ReadInt32(Path.Combine(Directory, "input_ids.npy")).Data;

    private static MaxwellModel Model() => MaxwellModel.Load(Directory, new CpuBackend(), new MaxwellLoadOptions { MaxSequenceTokens = 64, ResidentWeights = true });

    /// <summary>Largest absolute difference relative to the reference's largest magnitude.</summary>
    private static double Relative(ReadOnlySpan<float> actual, ReadOnlySpan<float> reference)
    {
        Assert.Equal(reference.Length, actual.Length);
        double diff = 0, scale = 1e-6;
        for (int i = 0; i < reference.Length; i++)
        {
            diff = Math.Max(diff, Math.Abs(actual[i] - reference[i]));
            scale = Math.Max(scale, Math.Abs(reference[i]));
        }
        return diff / scale;
    }

    [Fact]
    public void Prefill_MatchesReferenceLayerByLayerAndInTheLogits()
    {
        using MaxwellModel model = Model();
        MaxwellSequenceState state = model.CreateState(64);
        int[] ids = Tokens();
        List<double> layers = [];

        float[] stream = model.ForwardStream(ids, state, (layer, s) =>
        {
            // hidden_{i+1} is the 4-stream residual after layer i; the last layer's stream is only dumped read out and normed.
            if (layer < model.Config.NumHiddenLayers - 1) layers.Add(Relative(s, Load($"hidden_{layer + 1}")));
        });
        float[] hidden = model.ReadOut(stream);
        float[] logits = model.Logits(hidden, ids.Length);

        for (int i = 0; i < layers.Count; i++) output.WriteLine($"layer {i}: relative max diff {layers[i]:E2}");
        double finalHidden = Relative(hidden, Load("hidden_5")), final = Relative(logits, Load("logits"));
        output.WriteLine($"final hidden {finalHidden:E2}, logits {final:E2}");
        Assert.All(layers, static d => Assert.True(d < 1e-4, $"layer stream differs by {d:E2}"));
        Assert.True(finalHidden < 1e-4, $"final hidden differs by {finalHidden:E2}");
        Assert.True(final < 1e-4, $"logits differ by {final:E2}");
    }

    [Fact]
    public void CachedDecode_PastTheDenseLimit_MatchesReference()
    {
        using MaxwellModel model = Model();
        MaxwellSequenceState state = model.CreateState(64);
        int[] ids = Tokens();
        const int Prefill = 30;
        int vocab = model.Config.VocabSize;

        float[] prefill = model.Forward(ids.AsSpan(0, Prefill), state, allRows: true);
        float[] decodeReference = Load("decode_logits");
        double worst = Relative(prefill, Load("prefill_logits"));
        for (int step = 0; step < ids.Length - Prefill; step++)
        {
            float[] logits = model.Forward(ids.AsSpan(Prefill + step, 1), state);
            worst = Math.Max(worst, Relative(logits, decodeReference.AsSpan(step * vocab, vocab)));
        }

        output.WriteLine($"prefill + {ids.Length - Prefill} decode steps: worst relative max diff {worst:E2}");
        Assert.Equal(ids.Length, state.Length);
        Assert.True(worst < 1e-4, $"cached decode differs by {worst:E2}");
    }

    [Fact]
    public void RemoteSelection_IsWhatMatchesThePositionsPastTheDenseLimit()
    {
        // With room for every block the same weights attend densely: that must agree with the reference below the dense limit and miss it past it.
        JsonObject config = JsonNode.Parse(File.ReadAllText(Path.Combine(Directory, "config.json")))!.AsObject();
        config["block_sparse_gqa_num_remote_blocks"] = 100;
        config["block_sparse_gqa_total_support_blocks"] = 104;
        MaxwellConfig wide = MaxwellConfig.Parse(config.ToJsonString());
        using BlockFp8CheckpointWeights weights = BlockFp8CheckpointWeights.Open(Directory, "Maxwell");
        using MaxwellModel model = new(new CpuBackend(), wide, weights, new MaxwellLoadOptions { MaxSequenceTokens = 64, ResidentWeights = true });
        // The wide config's dense limit is 112 tokens, so every query takes the dense path; the narrow geometry's limit is 24.
        float[] dense = model.Forward(Tokens(), model.CreateState(64), allRows: true);
        float[] reference = Load("logits");
        int offset = Config().Sparse.DenseCausalLimit * model.Config.VocabSize;

        double below = Relative(dense.AsSpan(0, offset), reference.AsSpan(0, offset));
        double past = Relative(dense.AsSpan(offset), reference.AsSpan(offset));
        output.WriteLine($"all-blocks attention vs routed reference: below the limit {below:E2}, past it {past:E2}");
        Assert.True(below < 1e-4);
        Assert.True(past > 1e-3, "the routed reference equals dense attention past the limit, so the fixture does not test remote selection");
    }

    private static MaxwellConfig Config() => MaxwellConfig.Parse(File.ReadAllText(Path.Combine(Directory, "config.json")));
}
