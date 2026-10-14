using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Host-staged graph decode embedding (an untied quantized embed table, which graph decode never uploads):
/// the staged row must be the eager lookup's row bit for bit, and a graph session must write each replay's row before
/// that replay launches, for the token the previous replay sampled. CPU only: the session is driven through a recording
/// <see cref="IBackend"/> proxy, so no device is opened.</summary>
public sealed class GraphDecodeStagedRowTests
{
    private static TransformerConfig Cfg(float embeddingScale) => new()
    {
        HiddenSize = 16, NumLayers = 1, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
        IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 1024, QkNorm = false,
        EmbeddingScale = embeddingScale,
    };

    private static unsafe Tensor Random(int rows, int cols, ref uint state)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            p[i] = ((state & 0xFFFF) / 65535f - 0.5f) * 0.3f;
        }
        return t;
    }

    private static unsafe Tensor Ones(int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        new Span<float>((float*)t.DataPointer, n).Fill(1f);
        return t;
    }

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        uint state = 0x5EED1E55u;
        int h = c.HiddenSize;
        Dictionary<string, Tensor> w = new()
        {
            ["model.embed_tokens.weight"] = Random(c.VocabSize, h, ref state),
            ["model.norm.weight"] = Ones(h),
        };
        string p = "model.layers.0";
        w[$"{p}.input_layernorm.weight"] = Ones(h);
        w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
        w[$"{p}.self_attn.q_proj.weight"] = Random(c.QDim, h, ref state);
        w[$"{p}.self_attn.k_proj.weight"] = Random(c.KvDim, h, ref state);
        w[$"{p}.self_attn.v_proj.weight"] = Random(c.KvDim, h, ref state);
        w[$"{p}.self_attn.o_proj.weight"] = Random(h, c.QDim, ref state);
        w[$"{p}.mlp.gate_proj.weight"] = Random(c.IntermediateSize, h, ref state);
        w[$"{p}.mlp.up_proj.weight"] = Random(c.IntermediateSize, h, ref state);
        w[$"{p}.mlp.down_proj.weight"] = Random(h, c.IntermediateSize, ref state);
        return w;
    }

    [Theory]
    [InlineData(1f)]
    [InlineData(12f)]
    public unsafe void GraphDecodeEmbedRow_MatchesTheEagerLookupBitForBit(float embeddingScale)
    {
        TransformerConfig cfg = Cfg(embeddingScale);
        GenericTransformer transformer = new(cfg);
        transformer.LoadWeights(Weights(cfg), "model");

        foreach (int token in new[] { 0, 1, 17, cfg.VocabSize - 1 })
        {
            using Tensor eager = new(new TensorShape(1, 1, cfg.HiddenSize), DType.F32);
            transformer.EmbedLookup(eager, [token]);
            float* staged = (float*)transformer.GraphDecodeEmbedRow(token);
            float* want = (float*)eager.DataPointer;
            for (int i = 0; i < cfg.HiddenSize; i++)
                Assert.Equal(BitConverter.SingleToInt32Bits(want[i]), BitConverter.SingleToInt32Bits(staged[i]));
        }
    }

    [Fact]
    public void GraphDecodeEmbedRow_RejectsATokenOutsideTheVocabulary()
    {
        TransformerConfig cfg = Cfg(12f);
        GenericTransformer transformer = new(cfg);
        transformer.LoadWeights(Weights(cfg), "model");
        Assert.Throws<ArgumentException>(() => transformer.GraphDecodeEmbedRow(cfg.VocabSize));
        Assert.Throws<ArgumentException>(() => transformer.GraphDecodeEmbedRow(-1));
    }

    /// <summary>Records the backend calls a graph session makes, in order. Token-id replies come from a script; every other
    /// call returns the default for its return type.</summary>
    public class RecordingBackend : DispatchProxy
    {
        public List<string> Calls { get; } = new();
        public Queue<int> Sampled { get; } = new();
        public List<float[]> Rows { get; } = new();

        protected override unsafe object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            string name = targetMethod!.Name;
            switch (name)
            {
                case nameof(IBackend.WriteDecodeEmbedRow):
                {
                    Calls.Add($"write:{(ulong)args![0]!}");
                    nint row = (nint)args[1]!;
                    int count = (int)args[2]!;
                    float[] values = new float[count];
                    for (int i = 0; i < count; i++) values[i] = ((float*)row)[i];
                    Rows.Add(values);
                    return null;
                }
                case nameof(IBackend.LaunchGraph):
                    Calls.Add("launch");
                    return null;
                case nameof(IBackend.ReadDeviceTokenId):
                    Calls.Add("read");
                    return Sampled.Dequeue();
                default:
                    if (targetMethod.ReturnType == typeof(void)) return null;
                    return targetMethod.ReturnType.IsValueType ? Activator.CreateInstance(targetMethod.ReturnType) : null;
            }
        }
    }

    /// <summary>A row source whose row for a token holds recognisable values: <c>token * 100 + index</c>.</summary>
    private sealed unsafe class TokenRows(int width) : IGraphDecodeRowSource, IDisposable
    {
        private readonly float* _row = (float*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)(width * sizeof(float)));

        public nint GraphDecodeEmbedRow(int tokenId)
        {
            for (int i = 0; i < width; i++) _row[i] = tokenId * 100f + i;
            return (nint)_row;
        }

        public void Dispose() => System.Runtime.InteropServices.NativeMemory.Free(_row);
    }

    [Fact]
    public void GraphSession_WritesEachReplaysRowBeforeItsLaunch_ForTheToken_ThePreviousReplaySampled()
    {
        IBackend backend = DispatchProxy.Create<IBackend, RecordingBackend>();
        RecordingBackend recorder = (RecordingBackend)backend;
        recorder.Sampled.Enqueue(6);
        recorder.Sampled.Enqueue(7);

        using TokenRows rows = new(width: 4);
        GraphDecodeSession session = new(backend, new object(), devicePos: 1, deviceTokenId: 2, history: 0,
            historyCount: 0, pos: 0, rng: 0, embedRow: 9, embedWidth: 4, firstToken: 5, rowSource: rows);

        Assert.Equal(6, session.Replay());
        Assert.Equal(7, session.Replay());

        Assert.Equal(new[] { "write:9", "launch", "read", "write:9", "launch", "read" }, recorder.Calls);
        Assert.Equal(new float[] { 500f, 501f, 502f, 503f }, recorder.Rows[0]);   // the first replay embeds the firstToken
        Assert.Equal(new float[] { 600f, 601f, 602f, 603f }, recorder.Rows[1]);   // the second embeds what the first sampled
        session.Dispose();
    }

    [Fact]
    public void GraphSession_WithoutAStagedRow_OnlyLaunchesAndReads()
    {
        IBackend backend = DispatchProxy.Create<IBackend, RecordingBackend>();
        RecordingBackend recorder = (RecordingBackend)backend;
        recorder.Sampled.Enqueue(4);

        GraphDecodeSession session = new(backend, new object(), devicePos: 1, deviceTokenId: 2, history: 0,
            historyCount: 0, pos: 0);

        Assert.Equal(4, session.Replay());
        Assert.Equal(new[] { "launch", "read" }, recorder.Calls);
        Assert.Empty(recorder.Rows);
        session.Dispose();
    }
}
