using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Kyutai/moshi TTS generator assembling the three validated cores (<see cref="MoshiTransformer"/>
/// temporal backbone, <see cref="MoshiDepformer"/> depth transformer, <see cref="MoshiConditioner"/>) plus the
/// input embeddings. Per frame the temporal input is <c>Σ emb[k](audioCode_k) + text_emb(textToken) +
/// sum_condition</c>; the text embedding DEMUXES the multiplexed second stream (<c>out1</c>/<c>out2</c>). The
/// backbone (run over the whole prefix each step — TODO(perf): add a streaming KV cache like upstream) yields
/// the per-frame context that drives the depformer.</summary>
public sealed unsafe class MoshiTtsGenerator : IDisposable
{
    public const int Dim = 2048, TextCard = 8000, AudioCard = 2048, NumCodebooks = 32;
    public readonly MoshiTransformer Backbone = new(layers: 16);
    public readonly MoshiDepformer Depformer = new();
    public readonly MoshiConditioner Conditioner = new();

    private Tensor? _textW, _textOut1, _textOut2, _textLinear;
    private readonly Tensor?[] _emb = new Tensor?[NumCodebooks];
    private int _zeroToken = -1, _disposed;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        Backbone.LoadWeights(w);
        Depformer.LoadWeights(w);
        Conditioner.LoadWeights(w);
        _textW = WhisperOps.EnsureF32(w["text_emb.weight"]);          // [8001,2048]
        _textOut1 = WhisperOps.EnsureF32(w["text_emb.out1.weight"]);  // [2048,2048]
        _textOut2 = WhisperOps.EnsureF32(w["text_emb.out2.weight"]);  // [2048,2048]
        _textLinear = WhisperOps.EnsureF32(w["text_linear.weight"]);  // [8000,2048]
        for (int k = 0; k < NumCodebooks; k++) _emb[k] = WhisperOps.EnsureF32(w[$"emb.{k}.weight"]);
    }

    /// <summary>Demuxing text embedding (the temporal text stream): the input token packs two streams as
    /// <c>(second+1)·card + main</c>; <c>y = out1(emb[main]) + (second≥0 ? out2(emb[second]) : 0)</c>.</summary>
    public Tensor EmbedText(IBackend backend, int token)
    {
        if (token == _zeroToken) return new Tensor(new TensorShape(1, 1, Dim), DType.F32);  // all-zeros
        int card = TextCard + 1;
        int main = token % card, second = token / card - 1;
        Tensor leftRow = Row(_textW!, main);
        Tensor y = WhisperOps.ProjectLinear(backend, leftRow, _textOut1!, null, 1, 1, Dim, Dim);
        leftRow.Dispose();
        if (second >= 0)
        {
            Tensor rightRow = Row(_textW!, second);
            Tensor r = WhisperOps.ProjectLinear(backend, rightRow, _textOut2!, null, 1, 1, Dim, Dim);
            rightRow.Dispose();
            backend.Add(y, y, r); r.Dispose();
        }
        return y;
    }

    /// <summary>Frame input embedding: <c>text_emb(textToken) + Σ emb[k](audioCodes[k])</c> → <c>[1,1,2048]</c>.
    /// Audio codes use the moshi initial/zero token convention; pass the (delayed) previous-frame codes.</summary>
    public Tensor EmbedFrame(IBackend backend, int textToken, ReadOnlySpan<int> audioCodes)
    {
        Tensor x = EmbedText(backend, textToken);
        float* xp = (float*)x.DataPointer;
        for (int k = 0; k < NumCodebooks && k < audioCodes.Length; k++)
        {
            float* row = (float*)_emb[k]!.DataPointer + (long)audioCodes[k] * Dim;
            for (int i = 0; i < Dim; i++) xp[i] += row[i];
        }
        return x;
    }

    /// <summary>Runs the backbone over a full prefix of frame embeddings (each <c>[1,1,2048]</c>, plus the shared
    /// <paramref name="sumCond"/>) with cross conditioning <paramref name="cross"/>. Returns the per-frame context
    /// <c>transformer_out [1,T,2048]</c>; <paramref name="textLogits"/> is <c>[1,T,8000]</c>.</summary>
    public Tensor ForwardText(IBackend backend, IReadOnlyList<Tensor> frameEmbeds, Tensor? sumCond, Tensor cross, out Tensor textLogits)
    {
        int t = frameEmbeds.Count;
        Tensor seq = new(new TensorShape(1, t, Dim), DType.F32);
        float* sp = (float*)seq.DataPointer;
        float* cp = sumCond is null ? null : (float*)sumCond.DataPointer;
        for (int f = 0; f < t; f++)
        {
            float* e = (float*)frameEmbeds[f].DataPointer;
            float* dst = sp + (long)f * Dim;
            for (int i = 0; i < Dim; i++) dst[i] = e[i] + (cp is null ? 0f : cp[i]);
        }
        Tensor transformerOut = Backbone.Forward(backend, seq, cross);   // post out_norm [1,t,2048]
        seq.Dispose();
        textLogits = WhisperOps.ProjectLinear(backend, transformerOut, _textLinear!, null, 1, t, Dim, TextCard);
        return transformerOut;
    }

    /// <summary>Last-frame context slice <c>[1,1,2048]</c> from a <c>[1,T,2048]</c> transformer_out (for the depformer).</summary>
    public static Tensor LastFrame(Tensor transformerOut)
    {
        int t = (int)transformerOut.Shape[1], dim = (int)transformerOut.Shape[2];
        Tensor outT = new(new TensorShape(1, 1, dim), DType.F32);
        Buffer.MemoryCopy((float*)transformerOut.DataPointer + (long)(t - 1) * dim, (void*)outT.DataPointer, (long)dim * 4, (long)dim * 4);
        return outT;
    }

    private static Tensor Row(Tensor table, int row)
    {
        Tensor outT = new(new TensorShape(1, 1, Dim), DType.F32);
        Buffer.MemoryCopy((float*)table.DataPointer + (long)row * Dim, (void*)outT.DataPointer, (long)Dim * 4, (long)Dim * 4);
        return outT;
    }

    public void SetZeroToken(int z) => _zeroToken = z;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Backbone.Dispose(); Depformer.Dispose(); Conditioner.Dispose();
        GC.SuppressFinalize(this);
    }
}
