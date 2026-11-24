using HartsyInference.Core.Backends.Cpu;
using HartsyInference.LLM.Ernie5;
using HartsyInference.LLM.Glm5Next;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The C# decoder against transformers' <c>Glm5NextForConditionalGeneration</c> (commit 52bac5d0, torch 2.7.1 CPU, F32) on a tiny random model
/// with the real key layout: 4 layers (KDA + dense, DSA + MoE, KDA + MoE, DSA + MoE), a 41-token prompt whose indexer keeps 4 of 10 pools plus a tail.
/// Reference: <c>tools/parity/glm5_next_reference.py</c>; the saved seed has a 0.0216 gap between the last kept and first dropped pool score.</summary>
public sealed class Glm5NextReferenceParityTests(ITestOutputHelper output)
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next", "tiny");

    private static float[] Npy(string name) => NpyReader.ReadFloat32(Path.Combine(Dir, name)).Data;

    [Fact]
    public void Prefill_StreamsHiddenAndLogits_MatchTheReference()
    {
        Glm5NextConfig c = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
        int[] ids = [.. NpyReader.ReadInt64(Path.Combine(Dir, "input_ids.npy")).Data.Select(v => (int)v)];
        using Ernie5CheckpointWeights source = Ernie5CheckpointWeights.Open(Dir);
        using Glm5NextModel model = new(new CpuBackend(), c, source);
        using Glm5NextState state = new(c, ids.Length);
        Dictionary<int, float[]> streams = [];
        model.LayerObserver = (layer, stream) => streams[layer] = (float[])stream.Clone();

        float[] embeds = model.Embed(ids);
        float[] hidden = model.ForwardHidden(embeds, state);
        float[] logits = model.Project(hidden);

        Report("embeds", embeds, Npy("embeds.npy"), 1e-6f);
        for (int l = 0; l < c.Layers; l++) Report($"layer {l} stream", streams[l], Npy($"layer_{l}_stream.npy"), 2e-4f);
        Report("final hidden", hidden, Npy("final_hidden.npy"), 2e-4f);
        Report("logits", logits, Npy("logits.npy"), 5e-4f);
    }

    [Fact]
    public void TokenByTokenDecode_MatchesTheReferencesOwnCachedDecode()
    {
        Glm5NextConfig c = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
        int[] ids = [.. NpyReader.ReadInt64(Path.Combine(Dir, "input_ids.npy")).Data.Select(v => (int)v)];
        using Ernie5CheckpointWeights source = Ernie5CheckpointWeights.Open(Dir);
        using Glm5NextModel model = new(new CpuBackend(), c, source);
        using Glm5NextState state = new(c, ids.Length);
        List<float> logits = [];
        foreach (int id in ids) logits.AddRange(model.Forward([id], state));
        Report("decode logits", [.. logits], Npy("decode_logits.npy"), 5e-4f);
    }

    [Fact]
    public void WiderTopK_DoesNotMatchTheReference_SoTheSelectionIsExercised()
    {
        Glm5NextConfig c = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(Dir, "config.json"))) with { IndexTopK = 1000 };
        int[] ids = [.. NpyReader.ReadInt64(Path.Combine(Dir, "input_ids.npy")).Data.Select(v => (int)v)];
        using Ernie5CheckpointWeights source = Ernie5CheckpointWeights.Open(Dir);
        using Glm5NextModel model = new(new CpuBackend(), c, source);
        using Glm5NextState state = new(c, ids.Length);
        float[] logits = model.Forward(ids, state), expected = Npy("logits.npy");
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - logits[i]));
        output.WriteLine($"top-k 1000 (no pool is ever dropped): max abs logit difference {worst:E3}");
        Assert.True(worst > 1e-2f, $"A top-k that keeps every pool left the logits within {worst} of the reference: the selection is not exercised.");
    }

    private void Report(string what, float[] actual, float[] expected, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0f, scale = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
            scale = MathF.Max(scale, MathF.Abs(expected[i]));
        }
        output.WriteLine($"{what}: max abs difference {worst:E3} (reference max magnitude {scale:F3}, {expected.Length} values)");
        Assert.True(worst < tolerance, $"{what}: max abs difference {worst} exceeds {tolerance}");
    }
}
