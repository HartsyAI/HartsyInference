using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Kyutai/moshi TTS generator assembling the three validated cores (<see cref="MoshiTransformer"/>
/// temporal backbone, <see cref="MoshiDepformer"/> depth transformer, <see cref="MoshiConditioner"/>) plus the
/// input embeddings. Per frame the temporal input is <c>Σ emb[k](audioCode_k) + text_emb(textToken) +
/// sum_condition</c>; the text embedding DEMUXES the multiplexed second stream (<c>out1</c>/<c>out2</c>). The
/// backbone runs one token per frame through a streaming <c>FixedKvCache</c> (see <see cref="StepText"/> /
/// <see cref="MoshiTransformer.StepForward"/>) — O(n), not the O(n²) full-prefix re-run — yielding the per-frame
/// context that drives the depformer. Greedy text sampling uses the device-resident
/// <see cref="IBackend.ArgMaxLastDim"/>.</summary>
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
            int code = audioCodes[k];
            // moshi's ScaledEmbedding maps the zero token (-1) to a zero contribution (is_zero); the
            // initial/special token (AudioCard=2048) is a real row. Anything < 0 must NOT index the table.
            if (code == _zeroToken || code < 0) continue;
            float* row = (float*)_emb[k]!.DataPointer + (long)code * Dim;
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

    /// <summary>Streaming single-frame step (the O(n) replacement for <see cref="ForwardText"/> in the generation
    /// loop): adds <paramref name="sumCond"/> to the frame embedding, runs the backbone's
    /// <see cref="MoshiTransformer.StepForward"/> with the persistent <paramref name="selfCache"/> /
    /// <paramref name="crossKv"/> at position <paramref name="pos"/>, and projects the single output row to the
    /// text head. Returns the per-frame context <c>[1,1,2048]</c> (feed straight to the depformer);
    /// <paramref name="textLogits"/> is <c>[1,1,8000]</c>.</summary>
    public Tensor StepText(IBackend backend, Tensor frameEmbed, Tensor? sumCond, MoshiTransformer.CrossKvCache crossKv,
        FixedKvCache selfCache, int pos, out Tensor textLogits)
    {
        Tensor seq;
        if (sumCond is null) { seq = frameEmbed; }
        else { seq = new(new TensorShape(1, 1, Dim), DType.F32); backend.Add(seq, frameEmbed, sumCond); }
        Tensor tout = Backbone.StepForward(backend, seq, crossKv, selfCache, pos);
        if (!ReferenceEquals(seq, frameEmbed)) seq.Dispose();
        textLogits = WhisperOps.ProjectLinear(backend, tout, _textLinear!, null, 1, 1, Dim, TextCard);
        return tout;
    }

    /// <summary>Last-frame context slice <c>[1,1,2048]</c> from a <c>[1,T,2048]</c> transformer_out (for the depformer).</summary>
    public static Tensor LastFrame(Tensor transformerOut)
    {
        int t = (int)transformerOut.Shape[1], dim = (int)transformerOut.Shape[2];
        Tensor outT = new(new TensorShape(1, 1, dim), DType.F32);
        Buffer.MemoryCopy((float*)transformerOut.DataPointer + (long)(t - 1) * dim, (void*)outT.DataPointer, (long)dim * 4, (long)dim * 4);
        return outT;
    }

    /// <summary>Greedy autoregressive generation. Runs the delayed-streams frame loop: per frame it reads the
    /// (delayed) feedback codes from a ring cache, embeds + runs the backbone over the whole prefix, samples the
    /// text token (argmax), lets the <paramref name="scheduler"/> decide the actual text to feed, runs the
    /// depformer for the 32 audio codes (forced silent during the <paramref name="delaySteps"/> text-lead), and
    /// emits acoustic-delay-corrected codes. Returns <c>[32, nFrames]</c> Mimi codes (warmup frames trimmed).
    /// cfg <paramref name="cfgCoef"/>=1.0 only (single forward). O(n) streaming: cross K/V are precomputed once
    /// and self-attention runs through a <see cref="FixedKvCache"/>, so each frame is a single-token backbone
    /// step rather than a full-prefix re-run.</summary>
    /// <param name="onValidFrame">Optional callback invoked with each frame as soon as it's emitted, starting
    /// from the same first valid (non-warmup) frame the post-loop trim below would keep — lets a caller stream
    /// audio incrementally instead of waiting for the whole utterance. Receives the exact same <c>int[32]</c>
    /// arrays that end up in the returned array; null (the default) preserves the original non-streaming
    /// behavior byte-for-byte.</param>
    public int[,] Generate(IBackend backend, KyutaiTextScheduler scheduler, IEnumerable<KyutaiTextScheduler.Entry> entries,
        Tensor cross, Tensor sumCond, int maxFrames = 250, int delaySteps = 16, int finalPadding = 4,
        float audioTemp = 0.6f, int audioTopK = 250, float textTemp = 0.6f, int textTopK = 25, int seed = 0,
        IBackend? depBackend = null, Action<int[]>? onValidFrame = null)
    {
        // The depformer works on tiny (1-token, ≤32-step) tensors whose head-split / KV-cache management runs on
        // host pointers; on a GPU backend that forces a D2H drain per op (~500 syncs/frame). Running the whole
        // depformer cascade on a CPU backend costs a single D2H copy of the per-frame context and then stays on
        // host — far cheaper for these small ops. Falls back to the main backend when not supplied.
        depBackend ??= backend;
        // moshi samples BOTH the text token (temp_text/top_k_text) and the audio codes: the sampled text token's
        // new_word/pad choice is what PACES the words through the scheduler. Greedy argmax over the text head
        // collapses to always-pad (the model is confident to articulate), so words only advance when the scheduler
        // FORCES one every max_padding steps — stretching the frame count and desynchronising the audio (mush).
        Random rng = new(seed);
        // Per-stream delays from the checkpoint config: [text=0, cb0(semantic)=0, cb1..cb31(acoustic)=2].
        // (Codebook 0 must stay delay 0 — it carries the semantic content the acoustic codebooks follow; giving
        // it delay 2 mis-aligns the whole cascade by two frames and scrambles the audio.)
        int[] delays = new int[1 + NumCodebooks];
        for (int k = 2; k < 1 + NumCodebooks; k++) delays[k] = 2;
        int maxDelay = 2, ct = maxFrames + maxDelay + 2;
        int[] initial = new int[1 + NumCodebooks];
        initial[0] = TextCard;                                       // text initial = 8000
        for (int k = 1; k < 1 + NumCodebooks; k++) initial[k] = AudioCard;  // audio initial = 2048
        int[,] cache = new int[1 + NumCodebooks, ct];

        KyutaiTextScheduler.State state = scheduler.NewState(entries);
        List<int[]> emitted = new();
        bool seenValidFrame = false;   // tracks the SAME leading-invalid-run skip the post-loop trim applies below
        using MoshiTransformer.CrossKvCache crossKv = Backbone.PrecomputeCrossKv(backend, cross);
        using FixedKvCache selfCache = new(numLayers: 16, batch: 1,
            numKvHeads: MoshiTransformer.Heads, headDim: MoshiTransformer.HeadDim, maxSequenceLength: maxFrames + 1);
        {
            for (int offset = 0; offset < maxFrames; offset++)
            {
                int textIn = offset <= delays[0] ? initial[0] : cache[0, offset % ct];
                int[] audioIn = new int[NumCodebooks];
                for (int k = 0; k < NumCodebooks; k++)
                    audioIn[k] = offset <= delays[1 + k] ? initial[1 + k] : cache[1 + k, offset % ct];

                using Tensor frameEmbed = EmbedFrame(backend, textIn, audioIn);
                Tensor lastCtx = StepText(backend, frameEmbed, sumCond, crossKv, selfCache, offset, out Tensor textLogits);
                // Sample the text token (moshi sample_token, temp_text/top_k_text) — NOT argmax; the sampled
                // new_word/pad choice paces the words. textLogits is [1,1,TextCard]; sample on the host.
                ReadOnlySpan<float> textSpan = new((void*)textLogits.DataPointer, TextCard);
                int textTok = textTemp > 0f ? SampleTopK(textSpan, textTemp, textTopK, rng) : ArgMax(textSpan);
                textLogits.Dispose();

                int outTok = scheduler.Process(offset, state, textTok, out _);

                int[] audio = new int[NumCodebooks];
                if (offset >= delaySteps)
                {
                    using Tensor logits = Depformer.DecodeFrameGreedy(depBackend, lastCtx, outTok, out audio, audioTemp, audioTopK, rng);
                }
                lastCtx.Dispose();
                for (int q = 0; q < NumCodebooks; q++)
                    if (offset < delays[1 + q] + delaySteps) audio[q] = -1;   // forced silence in the text-lead

                int wpos = (offset + 1) % ct;
                cache[0, wpos] = outTok;
                for (int k = 0; k < NumCodebooks; k++) cache[1 + k, wpos] = audio[k];

                if (offset + 1 > maxDelay)
                {
                    int[] frame = new int[NumCodebooks];
                    for (int k = 0; k < NumCodebooks; k++)
                        frame[k] = cache[1 + k, ((offset + 1 - maxDelay + delays[1 + k]) % ct + ct) % ct];
                    emitted.Add(frame);
                    if (onValidFrame is not null)
                    {
                        if (!seenValidFrame && IsValidFrame(frame)) seenValidFrame = true;
                        if (seenValidFrame) onValidFrame(frame);
                    }
                }

                if (state.EndStep is int es && offset >= es + delaySteps + finalPadding) break;
            }
        }

        // Trim warmup frames whose codes are still the forced-silence / initial tokens (any codebook < 0 or >= card).
        int start = 0;
        while (start < emitted.Count && !IsValidFrame(emitted[start])) start++;
        int n = emitted.Count - start;
        int[,] codes = new int[NumCodebooks, n];
        for (int f = 0; f < n; f++)
            for (int k = 0; k < NumCodebooks; k++) codes[k, f] = emitted[start + f][k];
        return codes;
    }

    private static bool IsValidFrame(int[] frame)
    {
        foreach (int c in frame) if (c < 0 || c >= AudioCard) return false;
        return true;
    }

    private static int ArgMax(ReadOnlySpan<float> v)
    {
        int best = 0; float bv = v[0];
        for (int i = 1; i < v.Length; i++) if (v[i] > bv) { bv = v[i]; best = i; }
        return best;
    }

    // Top-k temperature sampling (moshi sample_token): scale by temp, keep the topK highest, softmax, multinomial.
    private static int SampleTopK(ReadOnlySpan<float> logits, float temp, int topK, Random rng)
    {
        int n = logits.Length;
        int k = topK <= 0 ? n : Math.Min(topK, n);
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        float[] vals = new float[n];
        for (int i = 0; i < n; i++) vals[i] = logits[i];
        Array.Sort(idx, (a, b) => vals[b].CompareTo(vals[a]));   // descending by logit
        float max = vals[idx[0]] / temp;
        double sum = 0;
        double[] p = new double[k];
        for (int j = 0; j < k; j++) { p[j] = Math.Exp(vals[idx[j]] / temp - max); sum += p[j]; }
        double r = rng.NextDouble() * sum, acc = 0;
        for (int j = 0; j < k; j++) { acc += p[j]; if (r <= acc) return idx[j]; }
        return idx[k - 1];
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
