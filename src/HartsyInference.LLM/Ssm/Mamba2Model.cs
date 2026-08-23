using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Ssm;

/// <summary>Mamba-2 (SSD — state-space duality) decoder loaded from a llama.cpp <c>mamba2</c> GGUF (mamba2-370m, Codestral-Mamba); NOT a transformer.</summary>
/// <remarks>Differs from Mamba-1: <c>in_proj</c> is fused into <c>[z | xBC | dt]</c> (no separate <c>x_proj</c>/<c>dt_proj</c>); the causal depthwise Conv1d runs over <c>xBC = [x | B | C]</c>; A is a single scalar <i>per head</i> (not per-channel × state); the scan is grouped into heads of <see cref="HeadDim"/> with B/C shared across each of <see cref="NumGroups"/> groups; and a <b>gated</b> grouped RMSNorm (<c>y · silu(z)</c> then RMSNorm) precedes <c>out_proj</c>. Block: RMSNorm → in_proj → [z, xBC, dt] → Conv1d(xBC)+SiLU → split x/B/C → softplus(dt+bias) → <b>SSD scan</b> → gated-RMSNorm(y, z) → out_proj + residual. Linear projections run through <see cref="IBackend"/>; the conv/softplus/scan/gate run host-side.</remarks>
public sealed unsafe class Mamba2Model : IDisposable, ISsmModel
{
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    // Per-layer recurrent state, carried across calls so ForwardLastLogits can be fed only the NEW tokens each
    // call (true O(1)-per-token decode) instead of recomputing the whole growing sequence every step.
    // _convHistory[i] holds the last (ConvKernel-1) pre-conv xBC rows (the causal conv's left context);
    // _ssmState[i] holds the SSD scan's running per-head state. Both start zeroed (matches the original
    // zero-padding / zero-initial-state at true sequence position 0) and ResetState() re-zeros them for a new
    // generation (the model instance is reused across unrelated chat turns via the provider's device slot).
    private readonly float[][] _convHistory;
    private readonly float[][] _ssmState;
    private int _disposed;

    public GgufMetadata Metadata => _handle.Metadata;
    public int DModel { get; }
    public int NumLayers { get; }
    public int DInner { get; }
    public int DState { get; }
    public int ConvKernel { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int NumGroups { get; }
    public int VocabSize { get; }
    public float Eps { get; }

    private int ConvDim => DInner + 2 * NumGroups * DState;          // xBC width (x + B + C)
    private int InProjDim => 2 * DInner + 2 * NumGroups * DState + NumHeads;   // [z | x | B | C | dt]

    private Mamba2Model(GgufModelLoader.LoadedGgufModel handle, IReadOnlyDictionary<string, Tensor> w,
        int dModel, int layers, int dInner, int dState, int convK, int nHeads, int nGroups, int vocab, float eps)
    {
        _handle = handle; _w = w;
        DModel = dModel; NumLayers = layers; DInner = dInner; DState = dState; ConvKernel = convK;
        NumHeads = nHeads; HeadDim = dInner / nHeads; NumGroups = nGroups; VocabSize = vocab;
        Eps = eps <= 0f ? 1e-5f : eps;
        _convHistory = new float[layers][];
        _ssmState = new float[layers][];
        for (int i = 0; i < layers; i++)
        {
            _convHistory[i] = new float[(convK - 1) * ConvDim];
            _ssmState[i] = new float[NumHeads * HeadDim * dState];
        }
    }

    /// <summary>Zeroes every layer's recurrent state — call before starting a new, unrelated generation, since the model instance persists across chat turns via the provider's device slot.</summary>
    public void ResetState()
    {
        foreach (float[] h in _convHistory) Array.Clear(h);
        foreach (float[] s in _ssmState) Array.Clear(s);
    }

    public static Mamba2Model Load(string ggufPath)
    {
        (Dictionary<string, Tensor> w, GgufModelLoader.LoadedGgufModel handle) = GgufModelLoader.LoadDequantized(ggufPath, DType.F32);
        try
        {
            // Relabel the nn.Linear weights (in_proj/out_proj) + token_embd (tied lm_head) to [out, in]. The raw SSM
            // params (ssm_a, ssm_d, ssm_conv1d, ssm_norm, ssm_dt.bias) are read directly with explicit indexing.
            foreach (string key in w.Keys.ToList())
            {
                if (!key.EndsWith(".weight", StringComparison.Ordinal) || w[key].Shape.Rank != 2) continue;
                if (key == "token_embd.weight" || key.Contains("ssm_in") || key.Contains("ssm_out"))
                    w[key] = TensorCasts.RelabelRank2Copy(w[key]);
            }
            GgufMetadata m = handle.Metadata;
            int dModel = (int)m.GetUInt32("mamba2.embedding_length");
            int layers = (int)m.GetUInt32("mamba2.block_count");
            int dInner = (int)m.GetUInt32("mamba2.ssm.inner_size");
            int dState = (int)m.GetUInt32("mamba2.ssm.state_size");
            int convK = (int)m.GetUInt32("mamba2.ssm.conv_kernel");
            int nHeads = (int)m.GetUInt32("mamba2.ssm.time_step_rank");   // mamba2 reuses time_step_rank for n_heads
            int nGroups = (int)m.GetUInt32("mamba2.ssm.group_count", 1u);
            float eps = m.GetFloat32("mamba2.attention.layer_norm_rms_epsilon", 1e-5f);
            int vocab = (int)w["token_embd.weight"].Shape[0];   // relabeled to [vocab, dModel]
            return new Mamba2Model(handle, w, dModel, layers, dInner, dState, convK, nHeads, nGroups, vocab, eps);
        }
        catch { handle.Dispose(); throw; }
    }

    private Tensor W(string key) => _w[key];

    /// <summary>Runs the stack over <paramref name="ids"/> — the NEW tokens since the last call — advancing each layer's carried recurrent state, and returns the next-token logits (last position); call <see cref="ResetState"/> before the first call of a new generation.</summary>
    public float[] ForwardLastLogits(IBackend backend, IReadOnlyList<int> ids)
    {
        int seq = ids.Count, d = DModel;
        Tensor h = new(new TensorShape(1, seq, d), DType.F32);
        float* hp = (float*)h.DataPointer;
        float* emb = (float*)W("token_embd.weight").DataPointer;
        for (int s = 0; s < seq; s++)
            Buffer.MemoryCopy(emb + (long)ids[s] * d, hp + (long)s * d, (long)d * 4, (long)d * 4);

        for (int i = 0; i < NumLayers; i++)
        {
            Tensor next = Block(backend, h, i, seq);
            h.Dispose(); h = next;
        }
        Tensor normed = new(new TensorShape(1, seq, d), DType.F32);
        backend.RmsNorm(normed, h, W("output_norm.weight"), Eps);
        h.Dispose();
        backend.Sync();
        using Tensor last = new(new TensorShape(1, 1, d), DType.F32);
        Buffer.MemoryCopy((byte*)normed.DataPointer + (long)(seq - 1) * d * 4, (void*)last.DataPointer, (long)d * 4, (long)d * 4);
        normed.Dispose();
        using Tensor logits = new(new TensorShape(1, 1, VocabSize), DType.F32);
        backend.Linear(logits, last, W("token_embd.weight"), null);   // tied head
        backend.Sync();
        float[] outv = new float[VocabSize];
        fixed (float* o = outv) Buffer.MemoryCopy((void*)logits.DataPointer, o, (long)VocabSize * 4, (long)VocabSize * 4);
        return outv;
    }

    private Tensor Block(IBackend backend, Tensor hIn, int i, int seq)
    {
        int d = DModel, di = DInner, ds = DState, k = ConvKernel;
        int nh = NumHeads, hd = HeadDim, ng = NumGroups, convDim = ConvDim;
        int gStateW = ng * ds;          // total B (and C) width across groups
        int hpg = nh / ng;              // heads per group
        string p = $"blk.{i}";

        // RMSNorm → in_proj → [z | xBC | dt].
        Tensor xn = new(new TensorShape(1, seq, d), DType.F32);
        backend.RmsNorm(xn, hIn, W($"{p}.attn_norm.weight"), Eps);
        Tensor proj = new(new TensorShape(1, seq, InProjDim), DType.F32);
        backend.Linear(proj, xn, W($"{p}.ssm_in.weight"), null);
        xn.Dispose();
        backend.Sync();
        float* pp = (float*)proj.DataPointer;
        float[] z = new float[seq * di];          // gate
        float[] xbc = new float[seq * convDim];   // [x | B | C] to be conv'd
        float[] dtIn = new float[seq * nh];
        for (int s = 0; s < seq; s++)
        {
            long b = (long)s * InProjDim;
            for (int c = 0; c < di; c++) z[s * di + c] = pp[b + c];
            for (int c = 0; c < convDim; c++) xbc[s * convDim + c] = pp[b + di + c];
            for (int c = 0; c < nh; c++) dtIn[s * nh + c] = pp[b + di + convDim + c];
        }
        proj.Dispose();

        // Causal depthwise Conv1d over xBC (kernel k, left context = the (k-1) rows carried from the previous
        // call — zero on the very first call, matching the original zero-padding at true position 0).
        float[] history = _convHistory[i];
        float* convW = (float*)W($"{p}.ssm_conv1d.weight").DataPointer;   // [convDim, k]
        float* convB = (float*)W($"{p}.ssm_conv1d.bias").DataPointer;     // [convDim]
        float[] xc = new float[seq * convDim];
        for (int s = 0; s < seq; s++)
            for (int c = 0; c < convDim; c++)
            {
                float acc = convB[c];
                for (int j = 0; j < k; j++)
                {
                    // Combined [history (k-1 rows) ++ xbc (seq rows)] index for this tap.
                    int combined = s + j;
                    acc += combined < k - 1
                        ? convW[c * k + j] * history[combined * convDim + c]
                        : convW[c * k + j] * xbc[(combined - (k - 1)) * convDim + c];
                }
                xc[s * convDim + c] = acc / (1f + MathF.Exp(-acc));   // SiLU
            }
        // Carry the last (k-1) pre-conv rows forward as this layer's conv history for the next call.
        if (k > 1)
        {
            int carry = Math.Min(k - 1, seq);
            for (int r = 0; r < k - 1 - carry; r++)
                Array.Copy(history, (r + carry) * convDim, history, r * convDim, convDim);
            for (int r = 0; r < carry; r++)
                Array.Copy(xbc, (seq - carry + r) * convDim, history, (k - 1 - carry + r) * convDim, convDim);
        }
        // Split conv output into x [di], B [ng*ds], C [ng*ds].
        float[] x = new float[seq * di], Bm = new float[seq * gStateW], Cm = new float[seq * gStateW];
        for (int s = 0; s < seq; s++)
        {
            long b = (long)s * convDim;
            for (int c = 0; c < di; c++) x[s * di + c] = xc[b + c];
            for (int c = 0; c < gStateW; c++) Bm[s * gStateW + c] = xc[b + di + c];
            for (int c = 0; c < gStateW; c++) Cm[s * gStateW + c] = xc[b + di + gStateW + c];
        }

        // dt = softplus(dt + dt_bias) per head.
        float* dtB = (float*)W($"{p}.ssm_dt.bias").DataPointer;   // [nh]
        float[] dt = new float[seq * nh];
        for (int s = 0; s < seq; s++)
            for (int hh = 0; hh < nh; hh++)
            {
                float v = dtIn[s * nh + hh] + dtB[hh];
                dt[s * nh + hh] = v > 20f ? v : MathF.Log(1f + MathF.Exp(v));
            }

        // SSD scan: per head h (group g = h / heads_per_group), state[hd, ds].
        // A[h] is already -exp(A_log) (baked by the converter); dA = exp(dt·A).
        float* A = (float*)W($"{p}.ssm_a").DataPointer;   // [nh]
        float* Dp = (float*)W($"{p}.ssm_d").DataPointer;  // [nh]
        float[] y = new float[seq * di];                  // [seq, nh*hd] == [seq, di]
        float[] state = _ssmState[i];                      // carried across calls — zero only at true position 0
        for (int s = 0; s < seq; s++)
            for (int hh = 0; hh < nh; hh++)
            {
                int g = hh / hpg;
                float dth = dt[s * nh + hh], Ah = A[hh], Dh = Dp[hh];
                float dA = MathF.Exp(dth * Ah);
                long bOff = (long)s * gStateW + g * ds;     // B/C for this group at this step
                long stBase = (long)hh * hd * ds;
                for (int pdim = 0; pdim < hd; pdim++)
                {
                    int xi = s * di + hh * hd + pdim;
                    float xv = x[xi];
                    float dbx = dth * xv;                    // δ·x (B applied per state below)
                    long st = stBase + (long)pdim * ds;
                    float acc = 0f;
                    for (int n = 0; n < ds; n++)
                    {
                        float hs = dA * state[st + n] + dbx * Bm[bOff + n];
                        state[st + n] = hs;
                        acc += Cm[bOff + n] * hs;
                    }
                    y[xi] = acc + Dh * xv;
                }
            }

        // Gated grouped RMSNorm: y = y · silu(z); then RMSNorm within each of ng groups of size di/ng; × weight.
        float* normW = (float*)W($"{p}.ssm_norm.weight").DataPointer;   // [di]
        int gsz = di / ng;
        for (int s = 0; s < seq; s++)
        {
            for (int c = 0; c < di; c++) { float zv = z[s * di + c]; y[s * di + c] *= zv / (1f + MathF.Exp(-zv)); }
            for (int grp = 0; grp < ng; grp++)
            {
                long gb = (long)s * di + grp * gsz;
                float var = 0f;
                for (int c = 0; c < gsz; c++) { float v = y[gb + c]; var += v * v; }
                float inv = 1f / MathF.Sqrt(var / gsz + Eps);
                for (int c = 0; c < gsz; c++) y[gb + c] = y[gb + c] * inv * normW[grp * gsz + c];
            }
        }

        // out_proj + residual.
        Tensor yT = new(new TensorShape(1, seq, di), DType.F32);
        fixed (float* src = y) Buffer.MemoryCopy(src, (void*)yT.DataPointer, (long)seq * di * 4, (long)seq * di * 4);
        Tensor outp = new(new TensorShape(1, seq, d), DType.F32);
        backend.Linear(outp, yT, W($"{p}.ssm_out.weight"), null);
        yT.Dispose();
        Tensor result = new(new TensorShape(1, seq, d), DType.F32);
        backend.Add(result, hIn, outp);
        outp.Dispose();
        return result;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _handle.Dispose();
    }
}
