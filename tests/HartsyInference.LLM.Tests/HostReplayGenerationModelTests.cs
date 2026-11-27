using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The shared host-reference adapter shell, driven by a toy family whose hidden row is the token id and position and whose logits double the row.</summary>
public sealed class HostReplayGenerationModelTests
{
    private const int Vocab = 10;
    private const int Dim = 2;

    [Fact]
    public void APrefill_ReplaysEveryToken_AndReturnsTheLastRowWhenAsked()
    {
        using CpuBackend backend = new();
        using ToyModel model = new(backend, maxTokens: 8);
        using ISequenceState state = model.CreateSequenceState(new SequenceStateOptions(8));

        using (Tensor all = model.Prefill(new PrefillChunk(new[] { 3, 4 }, 0), state))
            Assert.Equal(new float[] { 3, 0, 4, 1 }, all.AsReadOnlySpan<float>().ToArray());
        using Tensor last = model.Prefill(new PrefillChunk(new[] { 5, 6 }, 2, LastRowOnly: true), state);

        Assert.Equal(new TensorShape(1, 1, Dim), last.Shape);
        Assert.Equal(new float[] { 6, 3 }, last.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void APrefill_IsRefusedBeforeTheSequenceMoves()
    {
        using CpuBackend backend = new();
        using ToyModel model = new(backend, maxTokens: 8);
        using ISequenceState state = model.CreateSequenceState(new SequenceStateOptions(8));
        using Tensor embeds = new(new TensorShape(1, 1, Dim), DType.F32);

        NotSupportedException withEmbeds = Assert.Throws<NotSupportedException>(() => model.Prefill(new PrefillChunk(new[] { 1 }, 0, Embeds: embeds), state));
        Assert.Equal(ToyModel.EmbedsRefusal, withEmbeds.Message);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Prefill(new PrefillChunk(new[] { 1, Vocab }, 0), state));
        Assert.Throws<ArgumentException>(() => model.Prefill(new PrefillChunk(new[] { 1 }, 1), state));
        Assert.Throws<ArgumentException>(() => model.Prefill(new PrefillChunk(Array.Empty<int>(), 0), state));

        Assert.Equal(0, model.Forwards);
    }

    [Fact]
    public void ASequenceLongerThanTheLoad_IsRefused_AndAForeignStateIsNamedByFamily()
    {
        using CpuBackend backend = new();
        using ToyModel model = new(backend, maxTokens: 4);

        Assert.Throws<ArgumentOutOfRangeException>(() => model.CreateSequenceState(new SequenceStateOptions(5)));
        ArgumentException foreign = Assert.Throws<ArgumentException>(() => model.Prefill(new PrefillChunk(new[] { 1 }, 0), new ForeignState()));
        Assert.StartsWith($"{nameof(ForeignState)} is not a Toy sequence state.", foreign.Message);
    }

    [Fact]
    public void ABatchDecode_StepsEachSequenceOnce_AndRefusesARepeatedState()
    {
        using CpuBackend backend = new();
        using ToyModel model = new(backend, maxTokens: 8);
        using ISequenceState a = model.CreateSequenceState(new SequenceStateOptions(8));
        using ISequenceState b = model.CreateSequenceState(new SequenceStateOptions(8));
        model.Prefill(new PrefillChunk(new[] { 1 }, 0), a).Dispose();

        using (Tensor rows = model.DecodeBatch(new[] { 7, 8 }, [a, b]))
            Assert.Equal(new float[] { 7, 1, 8, 0 }, rows.AsReadOnlySpan<float>().ToArray());
        int forwards = model.Forwards;

        Assert.Throws<ArgumentException>(() => model.DecodeBatch(new[] { 2, 3 }, [a, a]));
        Assert.Equal(forwards, model.Forwards);
        Assert.True(model.Capabilities.SupportsBatchDecode);
        Assert.Equal(new GenerationModelInfo("Toy", Vocab, Dim, 1, 8), model.Info);
    }

    [Fact]
    public void Logits_AreTheFamilyProjection_OfTheRequestedRows()
    {
        using CpuBackend backend = new();
        using ToyModel model = new(backend, maxTokens: 8);
        using Tensor hidden = new(new TensorShape(1, 2, Dim), DType.F32);
        new float[] { 1, 2, 3, 4 }.CopyTo(hidden.AsSpan<float>());

        using Tensor logits = model.ProjectLogits(hidden, 2);

        Assert.Equal(new TensorShape(1, 2, Vocab), logits.Shape);
        ReadOnlySpan<float> values = logits.AsReadOnlySpan<float>();
        Assert.Equal(2f, values[0]);
        Assert.Equal(6f, values[Vocab]);
        Assert.Throws<ArgumentException>(() => model.ProjectLogits(hidden, 3));
    }

    [Fact]
    public void DisposingTheAdapter_DisposesTheModel()
    {
        using CpuBackend backend = new();
        ToyModel model = new(backend, maxTokens: 8);

        model.Dispose();

        Assert.True(model.Disposed);
    }

    /// <summary>A family whose forward writes (token id, position) per row and whose projection puts twice a row's first value in its first logit.</summary>
    private sealed class ToyModel(CpuBackend backend, int maxTokens)
        : HostReplayGenerationModel("Toy", Vocab, Dim, 1, backend, maxTokens, "Toy", EmbedsRefusal, batchDecode: true)
    {
        public const string EmbedsRefusal = "The toy model takes text tokens only.";

        public int Forwards { get; private set; }

        public bool Disposed { get; private set; }

        public override long EstimateSequenceBytes(int contextTokens) => contextTokens;

        protected override HostReplaySequenceState CreateReplayState(int capacity)
        {
            int position = 0;
            return new HostReplaySequenceState(capacity, Dim, (ids, hidden) =>
            {
                Forwards++;
                for (int i = 0; i < ids.Length; i++)
                {
                    hidden[i * Dim] = ids[i];
                    hidden[i * Dim + 1] = position++;
                }
            }, () => position = 0);
        }

        protected override void ProjectRows(ReadOnlySpan<float> hidden, int rows, Span<float> logits)
        {
            for (int r = 0; r < rows; r++) logits[r * Vocab] = 2 * hidden[r * Dim];
        }

        protected override void DisposeModel() => Disposed = true;
    }

    private sealed class ForeignState : ISequenceState
    {
        public int Length => 0;

        public int Capacity => 1;

        public int MaxRollback => 0;

        public void Truncate(int newLength)
        {
        }

        public void Reset()
        {
        }

        public void Dispose()
        {
        }
    }
}
