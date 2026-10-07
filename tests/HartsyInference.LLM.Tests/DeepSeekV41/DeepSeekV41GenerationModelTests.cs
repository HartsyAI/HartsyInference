using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41GenerationModelTests
{
    private static JsonElement Fx => DeepSeekV41ModelFixtureCheckpoint.Fx;

    private static float[] Floats(JsonElement e) => DeepSeekV41ModelFixtureCheckpoint.Floats(e);

    private static int[] Ints(JsonElement e) => DeepSeekV41ModelFixtureCheckpoint.Ints(e);

    private sealed class NumberTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => [];

        public int[] EncodeOrdinary(string text) => [];

        public string Decode(IReadOnlyList<int> ids) => string.Join(",", ids);

        public int? SpecialId(string token) => null;

        public int? BosId => null;

        public int? EosId => null;

        public IReadOnlyList<int> StopIds => [];

        public string? BosToken => null;

        public string? EosToken => null;
    }

    private sealed class ForeignState : ISequenceState
    {
        public int Length => 0;

        public int Capacity => 8;

        public int MaxRollback => 0;

        public void Truncate(int newLength) { }

        public void Reset() { }

        public void Dispose() { }
    }

    // Writes the fixture checkpoint, loads it and wraps it; the model disposes the checkpoint, and the temp directory goes with the test.
    private static void WithModel(Action<DeepSeekV41GenerationModel> body, int maxTokens = 64)
    {
        string dir = Path.Combine(Path.GetTempPath(), "dsv41-gen-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            DeepSeekV41ModelFixtureCheckpoint.Write(dir);
            using CpuBackend cpu = new();
            using DeepSeekV41GenerationModel model = new(DeepSeekV41HostModelLoader.Load(cpu, dir, new DeepSeekV41LoadOptions(maxTokens, 3)), cpu);
            body(model);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    private static void AssertClose(float[] expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    private static float[] Logits(DeepSeekV41GenerationModel model, int[] ids, ISequenceState state)
    {
        using Tensor hidden = model.Prefill(new PrefillChunk(ids.AsMemory(), state.Length, LastRowOnly: true), state);
        using Tensor logits = model.ProjectLogits(hidden, 1);
        return logits.AsReadOnlySpan<float>().ToArray();
    }

    [Fact]
    public void Prefill_And_Decode_Logits_Match_Upstream_And_Shapes_Are_Right()
    {
        WithModel(model =>
        {
            Assert.Equal(Fx.GetProperty("config").GetProperty("vocab_size").GetInt32(), model.Info.VocabSize);
            Assert.Equal(64, model.Info.MaxContextTokens);
            using ISequenceState state = model.CreateSequenceState(new SequenceStateOptions(64));
            int stepNo = 0;
            foreach (JsonElement step in Fx.GetProperty("steps").EnumerateArray())
            {
                AssertClose(Floats(step.GetProperty("logits")), Logits(model, Ints(step.GetProperty("ids")), state), $"step {stepNo} logits");
                stepNo++;
            }
            Assert.Equal(17, state.Length);
        });
    }

    [Fact]
    public void LastRowOnly_Returns_One_Row_And_Otherwise_Every_Row_With_The_Same_Last_Row()
    {
        WithModel(model =>
        {
            int[] ids = Ints(Fx.GetProperty("steps")[0].GetProperty("ids"));
            using ISequenceState a = model.CreateSequenceState(new SequenceStateOptions(64)), b = model.CreateSequenceState(new SequenceStateOptions(64));
            using Tensor last = model.Prefill(new PrefillChunk(ids.AsMemory(), 0, LastRowOnly: true), a);
            using Tensor all = model.Prefill(new PrefillChunk(ids.AsMemory(), 0, LastRowOnly: false), b);
            int dim = model.Info.HiddenSize;
            Assert.Equal(new TensorShape(1, 1, dim), last.Shape);
            Assert.Equal(new TensorShape(1, ids.Length, dim), all.Shape);
            Assert.Equal(last.AsReadOnlySpan<float>().ToArray(), all.AsReadOnlySpan<float>().Slice((ids.Length - 1) * dim, dim).ToArray());
        });
    }

    [Fact]
    public void DecodeBatch_Gives_Each_Sequence_The_Row_It_Would_Get_Alone()
    {
        WithModel(model =>
        {
            int[] first = [1, 2, 3, 4], second = [9, 8, 7];
            using ISequenceState a = model.CreateSequenceState(new SequenceStateOptions(64)), b = model.CreateSequenceState(new SequenceStateOptions(64));
            using ISequenceState soloA = model.CreateSequenceState(new SequenceStateOptions(64)), soloB = model.CreateSequenceState(new SequenceStateOptions(64));
            Logits(model, first, a);
            Logits(model, second, b);
            Logits(model, first, soloA);
            Logits(model, second, soloB);

            using Tensor batched = model.DecodeBatch([5, 6], [a, b]);
            using Tensor one = model.DecodeBatch([5], [soloA]);
            using Tensor two = model.DecodeBatch([6], [soloB]);
            int dim = model.Info.HiddenSize;
            Assert.Equal(new TensorShape(1, 2, dim), batched.Shape);
            Assert.Equal(one.AsReadOnlySpan<float>().ToArray(), batched.AsReadOnlySpan<float>().Slice(0, dim).ToArray());
            Assert.Equal(two.AsReadOnlySpan<float>().ToArray(), batched.AsReadOnlySpan<float>().Slice(dim, dim).ToArray());
            Assert.Equal(first.Length + 1, a.Length);
            Assert.Equal(second.Length + 1, b.Length);
        });
    }

    [Fact]
    public void Truncate_Replays_The_Kept_Prefix_So_The_Next_Token_Matches_Upstream()
    {
        WithModel(model =>
        {
            JsonElement prefill = Fx.GetProperty("steps")[0], next = Fx.GetProperty("steps")[1];
            using ISequenceState state = model.CreateSequenceState(new SequenceStateOptions(64));
            Logits(model, Ints(prefill.GetProperty("ids")), state);
            Logits(model, [3, 3], state);
            Assert.Equal(13, state.Length);
            Assert.Equal(13, state.MaxRollback);

            state.Truncate(11);

            Assert.Equal(11, state.Length);
            AssertClose(Floats(next.GetProperty("logits")), Logits(model, Ints(next.GetProperty("ids")), state), "logits after truncate");
            state.Truncate(0);
            Assert.Equal(0, state.Length);
            Assert.Throws<ArgumentOutOfRangeException>(() => state.Truncate(1));
        });
    }

    [Fact]
    public void Refuses_Embeds_A_Wrong_Start_A_Foreign_State_Cancellation_And_An_Oversized_Capacity_Without_Changing_The_State()
    {
        WithModel(model =>
        {
            using ISequenceState state = model.CreateSequenceState(new SequenceStateOptions(64));
            int[] ids = [1, 2, 3];
            using Tensor embeds = new(new TensorShape(1, 3, model.Info.HiddenSize), DType.F32);
            Assert.Throws<NotSupportedException>(() => model.Prefill(new PrefillChunk(ids.AsMemory(), 0, false, embeds), state));
            Assert.Throws<ArgumentException>(() => model.Prefill(new PrefillChunk(ids.AsMemory(), 2, false), state));
            Assert.Throws<ArgumentException>(() => model.Prefill(new PrefillChunk(ids.AsMemory(), 0, false), new ForeignState()));
            using CancellationTokenSource cancelled = new();
            cancelled.Cancel();
            Assert.Throws<OperationCanceledException>(() => model.Prefill(new PrefillChunk(ids.AsMemory(), 0, false), state, cancelled.Token));
            Assert.Equal(0, state.Length);
            Assert.Throws<ArgumentOutOfRangeException>(() => model.CreateSequenceState(new SequenceStateOptions(65)));
            Assert.Throws<ArgumentException>(() => model.DecodeBatch([1, 2], [state]));
        });
    }

    [Fact]
    public void Estimates_Are_Positive_And_Grow_With_Context()
    {
        WithModel(model =>
        {
            Assert.True(model.EstimateSequenceBytes(64) > 0);
            Assert.True(model.EstimateSequenceBytes(64) >= model.EstimateSequenceBytes(8));
            Assert.Empty(model.EnumerateWeights(includeRedundantSplits: true));
            CapacitySnapshot capacity = model.Capacity();
            Assert.True(capacity.TotalBytes > 0 && capacity.FreeBytes >= 0);
        });
    }

    [Fact]
    public void The_Shared_Pipeline_Drives_It_Greedily_To_The_Same_Tokens_As_A_Manual_Loop()
    {
        WithModel(model =>
        {
            int[] prompt = Ints(Fx.GetProperty("steps")[0].GetProperty("ids"));
            float[] firstLogits = Floats(Fx.GetProperty("steps")[0].GetProperty("logits"));
            int firstToken = Array.IndexOf(firstLogits, firstLogits.Max());

            List<int> manual = [];
            using (ISequenceState state = model.CreateSequenceState(new SequenceStateOptions(64)))
            {
                int[] feed = prompt;
                for (int i = 0; i < 3; i++)
                {
                    float[] logits = Logits(model, feed, state);
                    int token = Array.IndexOf(logits, logits.Max());
                    manual.Add(token);
                    feed = [token];
                }
            }

            TextGenerationPipeline pipeline = new(model, new NumberTokenizer());
            GenerationResult result = pipeline.Generate(new GenerationRequest { RawTokenIds = prompt, MaxTokens = 3, Sampling = SamplingOptions.GreedyPreset });

            Assert.Equal(firstToken, result.TokenIds[0]);
            Assert.Equal(manual, result.TokenIds);
            Assert.Equal(prompt.Length, result.PromptTokens);
        });
    }
}
