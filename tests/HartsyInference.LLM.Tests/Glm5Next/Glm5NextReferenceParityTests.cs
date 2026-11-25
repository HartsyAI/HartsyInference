using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
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
        using BlockFp8CheckpointWeights source = BlockFp8CheckpointWeights.Open(Dir, Glm5NextTextModel.ModelType);
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
    public void Bf16Checkpoint_MatchesTheSameWeightsRoundedAndRunInF32()
    {
        // The real checkpoint is BF16, norms included. Stored as BF16, a tensor must read exactly as its rounded values run in F32: a misread
        // dtype (such as the final norm once was) shows as an error of whole units here. The tiny random model is itself very sensitive to
        // rounding (embedding rounding alone moves its logits by about 1), so the comparison is against the rounded F32 run, not the original.
        Glm5NextConfig c = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
        int[] ids = [.. NpyReader.ReadInt64(Path.Combine(Dir, "input_ids.npy")).Data.Select(v => (int)v)];
        float[] rounded = Logits(c, ids, new Bf16Source(BlockFp8CheckpointWeights.Open(Dir, Glm5NextTextModel.ModelType), storeBf16: false));
        float[] stored = Logits(c, ids, new Bf16Source(BlockFp8CheckpointWeights.Open(Dir, Glm5NextTextModel.ModelType), storeBf16: true));
        float worst = 0f;
        for (int i = 0; i < rounded.Length; i++) worst = MathF.Max(worst, MathF.Abs(rounded[i] - stored[i]));
        output.WriteLine($"BF16-stored vs rounded-F32 logits: max abs difference {worst:E3}");
        Assert.True(worst < 1e-4f, $"BF16 storage reads differently from its rounded values: {worst}.");
    }

    [Fact]
    public void WrongEmbeddingShape_IsRefusedByName()
    {
        // The refusal is the observable behaviour; the constructor also releases what it had read, which this test does not measure.
        Glm5NextConfig c = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
        using BlockFp8CheckpointWeights inner = BlockFp8CheckpointWeights.Open(Dir, Glm5NextTextModel.ModelType);
        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => new Glm5NextModel(new CpuBackend(), c, new WrongEmbedding(inner)));
        Assert.Contains("embed_tokens", ex.Message);
    }

    [Fact]
    public void ScaledFp8Embedding_IsRefusedByName()
    {
        Glm5NextConfig c = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
        using BlockFp8CheckpointWeights inner = BlockFp8CheckpointWeights.Open(Dir, Glm5NextTextModel.ModelType);
        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => new Glm5NextModel(new CpuBackend(), c, new ScaledEmbedding(inner, c)));
        Assert.Contains("embed_tokens", ex.Message);
    }

    // Replaces the embedding with a block-scaled FP8 table of the right shape and passes every other read through.
    private sealed class ScaledEmbedding(IModelWeightSource inner, Glm5NextConfig c) : IModelWeightSource
    {
        public IEnumerable<string> Names => inner.Names;

        public Tensor Read(string name)
        {
            if (name != Glm5NextWeightNames.Embedding) return inner.Read(name);
            Tensor table = new(new TensorShape(c.Vocab, c.Hidden), DType.F8E4M3);
            Tensor grid = new(new TensorShape(1, 1), DType.F32);
            grid.AsSpan<float>()[0] = 1f;
            table.QuantInfo = new QuantWeightInfo { Format = "fp8_block_scale_inv", Fp8BlockScaleInv = grid, Fp8BlockSize = 128 };
            return table;
        }
    }

    // Replaces the embedding with a tensor of the wrong shape and passes every other read through.
    private sealed class WrongEmbedding(IModelWeightSource inner) : IModelWeightSource
    {
        public IEnumerable<string> Names => inner.Names;

        public Tensor Read(string name) => name == Glm5NextWeightNames.Embedding
            ? HostMath.Tensor(new float[8], 2, 4)
            : inner.Read(name);
    }

    private static float[] Logits(Glm5NextConfig c, int[] ids, IModelWeightSource source)
    {
        using IDisposable owned = (IDisposable)source;
        using Glm5NextModel model = new(new CpuBackend(), c, source);
        using Glm5NextState state = new(c, ids.Length);
        return model.Forward(ids, state);
    }

    // Presents every F32 tensor of an inner source rounded to BF16: kept as BF16 storage, or read back as F32 holding the rounded values.
    private sealed class Bf16Source(IModelWeightSource inner, bool storeBf16) : IModelWeightSource, IDisposable
    {
        public IEnumerable<string> Names => inner.Names;

        public Tensor Read(string name)
        {
            Tensor raw = inner.Read(name);
            if (raw.DType != DType.F32) return raw;
            Tensor bf16 = raw.CastTo(DType.BF16);
            raw.Dispose();
            if (storeBf16) return bf16;
            Tensor f32 = bf16.CastTo(DType.F32);
            bf16.Dispose();
            return f32;
        }

        public void Dispose() => (inner as IDisposable)?.Dispose();
    }

    [Fact]
    public void TokenByTokenDecode_MatchesTheReferencesOwnCachedDecode()
    {
        Glm5NextConfig c = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
        int[] ids = [.. NpyReader.ReadInt64(Path.Combine(Dir, "input_ids.npy")).Data.Select(v => (int)v)];
        using BlockFp8CheckpointWeights source = BlockFp8CheckpointWeights.Open(Dir, Glm5NextTextModel.ModelType);
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
        using BlockFp8CheckpointWeights source = BlockFp8CheckpointWeights.Open(Dir, Glm5NextTextModel.ModelType);
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
