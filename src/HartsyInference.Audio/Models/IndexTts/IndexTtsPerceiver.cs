using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>IndexTTS-1.5's <c>gpt.perceiver_encoder</c> (naturalspeech2-pytorch <c>PerceiverResampler</c>, depth 2):
/// 32 learned latents jointly cross- and self-attend <c>[latents; proj_context(conformerOutput)]</c>, each layer
/// followed by a GEGLU feed-forward, both residual with NO pre-norm; a single RMSNorm closes the stack.</summary>
/// <remarks>Architecturally distinct from <see cref="HartsyInference.Audio.Models.Chatterbox.ChatterboxPerceiver"/>
/// (which does two separate attention passes — cross then self — with LayerNorm and a biased projection):
/// here the context sequence is concatenated with the latents themselves before a single attention call per
/// layer (<c>cross_attn_include_queries</c>), every linear is bias-free, and the closing norm is RMS, not
/// LayerNorm — verified against the real <c>gpt.pth</c> checkpoint's key set and the upstream source.</remarks>
internal sealed unsafe class IndexTtsPerceiver : IDisposable
{
    private readonly int _numLatents;
    private readonly int _dimHead;
    private readonly int _heads;

    private readonly int _dim;        // 1280 (GPT hidden)
    private readonly int _dimContext; // 512 (Conformer output)
    private readonly int _depth;
    private int _disposed;

    private Tensor? _latents;                 // [numLatents, dim]
    private Tensor? _projCtxW, _projCtxB;      // [dim, dimContext]
    private Tensor?[] _toQ = [], _toKv = [], _toOut = [];
    private Tensor?[] _ffW1 = [], _ffB1 = [], _ffW2 = [], _ffB2 = [];
    private Tensor? _normGamma;                // RMSNorm scale [dim]

    /// <summary><paramref name="numLatents"/>/<paramref name="heads"/>/<paramref name="dimHead"/> default to
    /// IndexTTS-1.5's speaker perceiver (32 latents, 8 heads of 64) — IndexTTS-2's emotion perceiver is the same
    /// upstream <c>PerceiverResampler</c> shape instantiated with <c>num_latents=1, heads=4</c> (real
    /// <c>model_v2.py</c>: <c>PerceiverResampler(1024, dim_context=512, ff_mult=2, heads=4, num_latents=1)</c>);
    /// <c>ff_mult</c> needs no parameter since <see cref="FeedForward"/> already infers it from the loaded
    /// weight's own shape.</summary>
    public IndexTtsPerceiver(int dim, int dimContext, int depth = 2, int numLatents = 32, int heads = 8, int dimHead = 64)
    {
        _dim = dim;
        _dimContext = dimContext;
        _depth = depth;
        _numLatents = numLatents;
        _heads = heads;
        _dimHead = dimHead;
        _toQ = new Tensor?[depth]; _toKv = new Tensor?[depth]; _toOut = new Tensor?[depth];
        _ffW1 = new Tensor?[depth]; _ffB1 = new Tensor?[depth]; _ffW2 = new Tensor?[depth]; _ffB2 = new Tensor?[depth];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _latents = WhisperOps.EnsureF32(w[$"{prefix}.latents"]);
        _projCtxW = WhisperOps.EnsureF32(w[$"{prefix}.proj_context.weight"]);
        _projCtxB = WhisperOps.EnsureF32(w[$"{prefix}.proj_context.bias"]);
        for (int i = 0; i < _depth; i++)
        {
            _toQ[i] = WhisperOps.EnsureF32(w[$"{prefix}.layers.{i}.0.to_q.weight"]);
            _toKv[i] = WhisperOps.EnsureF32(w[$"{prefix}.layers.{i}.0.to_kv.weight"]);
            _toOut[i] = WhisperOps.EnsureF32(w[$"{prefix}.layers.{i}.0.to_out.weight"]);
            _ffW1[i] = WhisperOps.EnsureF32(w[$"{prefix}.layers.{i}.1.0.weight"]);
            _ffB1[i] = WhisperOps.EnsureF32(w[$"{prefix}.layers.{i}.1.0.bias"]);
            _ffW2[i] = WhisperOps.EnsureF32(w[$"{prefix}.layers.{i}.1.2.weight"]);
            _ffB2[i] = WhisperOps.EnsureF32(w[$"{prefix}.layers.{i}.1.2.bias"]);
        }
        _normGamma = WhisperOps.EnsureF32(w[$"{prefix}.norm.gamma"]);
    }

    /// <summary>Resamples Conformer output <c>[1, T, dimContext]</c> → <c>[1, 32, dim]</c> conditioning latents.</summary>
    public Tensor Forward(IBackend backend, Tensor conformerOut, int t)
    {
        if (_latents is null) throw new InvalidOperationException("IndexTtsPerceiver weights not loaded.");
        Tensor context = WhisperOps.ProjectLinear(backend, conformerOut, _projCtxW!, _projCtxB, 1, t, _dimContext, _dim);

        Tensor latents = new(new TensorShape(1, _numLatents, _dim), DType.F32);
        float* lp = (float*)_latents!.DataPointer;
        float* l0 = (float*)latents.DataPointer;
        for (long i = 0; i < latents.ElementCount; i++) l0[i] = lp[i];

        for (int layer = 0; layer < _depth; layer++)
        {
            Tensor attnOut = Attention(backend, latents, context, t, layer);
            AddInPlace(latents, attnOut);
            attnOut.Dispose();

            Tensor ffOut = FeedForward(backend, latents, layer);
            AddInPlace(latents, ffOut);
            ffOut.Dispose();
        }
        context.Dispose();

        Tensor normed = new(latents.Shape, DType.F32);
        RmsNorm(normed, latents, _normGamma!);
        latents.Dispose();
        return normed;
    }

    /// <summary>Cross+self attention in one call: queries are the latents; keys/values come from
    /// <c>concat(latents, context)</c> (<c>cross_attn_include_queries=True</c>), so the latents also attend to
    /// each other. All linears bias-free.</summary>
    private Tensor Attention(IBackend backend, Tensor latents, Tensor context, int tContext, int layer)
    {
        int dimInner = _dimHead * _heads;
        int kvLen = _numLatents + tContext;

        Tensor q = WhisperOps.ProjectLinear(backend, latents, _toQ[layer]!, null, 1, _numLatents, _dim, dimInner);

        Tensor kvInput = new(new TensorShape(1, kvLen, _dim), DType.F32);
        float* kvp = (float*)kvInput.DataPointer;
        float* latp = (float*)latents.DataPointer;
        float* ctxp = (float*)context.DataPointer;
        for (long i = 0; i < (long)_numLatents * _dim; i++) kvp[i] = latp[i];
        for (long i = 0; i < (long)tContext * _dim; i++) kvp[(long)_numLatents * _dim + i] = ctxp[i];

        Tensor kv = WhisperOps.ProjectLinear(backend, kvInput, _toKv[layer]!, null, 1, kvLen, _dim, 2 * dimInner);
        kvInput.Dispose();

        float* qp = (float*)q.DataPointer;
        float* kvAll = (float*)kv.DataPointer;   // [1, kvLen, 2*dimInner] — k then v per row

        Tensor outMerged = new(new TensorShape(1, _numLatents, dimInner), DType.F32);
        float* om = (float*)outMerged.DataPointer;
        float scale = 1f / MathF.Sqrt(_dimHead);
        float[] scores = new float[kvLen];

        for (int h = 0; h < _heads; h++)
        {
            int hOff = h * _dimHead;
            for (int i = 0; i < _numLatents; i++)
            {
                float* qi = qp + (long)i * dimInner + hOff;
                float maxS = float.NegativeInfinity;
                for (int j = 0; j < kvLen; j++)
                {
                    float* kj = kvAll + (long)j * 2 * dimInner + hOff;
                    float dot = 0f;
                    for (int e = 0; e < _dimHead; e++) dot += qi[e] * kj[e];
                    float s = dot * scale;
                    scores[j] = s;
                    if (s > maxS) maxS = s;
                }
                float sum = 0f;
                for (int j = 0; j < kvLen; j++) { float e = MathF.Exp(scores[j] - maxS); scores[j] = e; sum += e; }
                float invSum = 1f / sum;
                float* oi = om + (long)i * dimInner + hOff;
                for (int e = 0; e < _dimHead; e++) oi[e] = 0f;
                for (int j = 0; j < kvLen; j++)
                {
                    float a = scores[j] * invSum;
                    float* vj = kvAll + (long)j * 2 * dimInner + dimInner + hOff;
                    for (int e = 0; e < _dimHead; e++) oi[e] += a * vj[e];
                }
            }
        }
        q.Dispose(); kv.Dispose();

        Tensor o = WhisperOps.ProjectLinear(backend, outMerged, _toOut[layer]!, null, 1, _numLatents, dimInner, _dim);
        outMerged.Dispose();
        return o;
    }

    /// <summary>GEGLU feed-forward: <c>Linear(dim→2·inner) → split → gelu(gate)·x → Linear(inner→dim)</c>.</summary>
    private Tensor FeedForward(IBackend backend, Tensor latents, int layer)
    {
        int doubled = (int)_ffW1[layer]!.Shape[0];
        int inner = doubled / 2;
        Tensor h = WhisperOps.ProjectLinear(backend, latents, _ffW1[layer]!, _ffB1[layer], 1, _numLatents, _dim, doubled);
        Tensor gated = new(new TensorShape(1, _numLatents, inner), DType.F32);
        float* hp = (float*)h.DataPointer;
        float* gp = (float*)gated.DataPointer;
        for (int i = 0; i < _numLatents; i++)
        {
            float* row = hp + (long)i * doubled;
            float* outRow = gp + (long)i * inner;
            for (int e = 0; e < inner; e++)
            {
                float x = row[e];
                float gate = row[inner + e];
                // erf-based GELU, matching PyTorch's default F.gelu.
                float geluGate = 0.5f * gate * (1f + Erf(gate / 1.41421356f));
                outRow[e] = geluGate * x;
            }
        }
        h.Dispose();
        Tensor o = WhisperOps.ProjectLinear(backend, gated, _ffW2[layer]!, _ffB2[layer], 1, _numLatents, inner, _dim);
        gated.Dispose();
        return o;
    }

    private void RmsNorm(Tensor output, Tensor input, Tensor gamma)
    {
        // Reference: out = F.normalize(x, dim=-1) * sqrt(dim) * gamma. L2-normalize divides by ||x||_2 =
        // sqrt(dim)*rms(x), so the sqrt(dim) factor exactly cancels that back to plain RMS-normalize * gamma —
        // NOT an extra sqrt(dim) on top of invRms (that would be the ~35.8x over-scale a review caught here).
        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        float* gp = (float*)gamma.DataPointer;
        for (int i = 0; i < _numLatents; i++)
        {
            float* row = ip + (long)i * _dim;
            double sumSq = 0d;
            for (int e = 0; e < _dim; e++) sumSq += (double)row[e] * row[e];
            float invRms = (float)(1.0 / Math.Sqrt(sumSq / _dim + 1e-12));
            float* outRow = op + (long)i * _dim;
            for (int e = 0; e < _dim; e++) outRow[e] = row[e] * invRms * gp[e];
        }
    }

    private static float Erf(float x)
    {
        // Abramowitz-Stegun 7.1.26, max error 1.5e-7 — sufficient for an activation nonlinearity.
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float t = 1f / (1f + p * x);
        float y = 1f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }

    private static void AddInPlace(Tensor dst, Tensor src)
    {
        float* dp = (float*)dst.DataPointer;
        float* sp = (float*)src.DataPointer;
        long n = dst.ElementCount;
        for (long i = 0; i < n; i++) dp[i] += sp[i];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_latents, _projCtxW, _projCtxB, _normGamma];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        for (int i = 0; i < _depth; i++)
        {
            Tensor?[] layer = [_toQ[i], _toKv[i], _toOut[i], _ffW1[i], _ffB1[i], _ffW2[i], _ffB2[i]];
            foreach (Tensor? t in layer) if (t is not null) yield return t;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
