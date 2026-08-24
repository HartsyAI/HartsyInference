using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Ssm;

/// <summary>Mamba-1 (selective state-space model) decoder loaded from a llama.cpp <c>mamba</c> GGUF (mamba-130m…, Falcon-Mamba); NOT a transformer, there is no attention. Each block: RMSNorm → in_proj → [x, z] → causal depthwise Conv1d(x) → SiLU → x_proj → (dt, B, C) → dt_proj+softplus → δ → <b>selective scan</b> → y·SiLU(z) → out_proj, with a residual. The linear projections run through <see cref="IBackend"/>; the conv/softplus/scan/gate run host-side (the scan is an inherently sequential recurrence).</summary>
public sealed unsafe class MambaModel : IDisposable, ISsmModel
{
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    // Per-layer recurrent state, carried across calls — see Mamba2Model's identical pattern for the full
    // rationale. _convHistory[i]: last (ConvKernel-1) pre-conv x rows. _ssmState[i]: the selective scan state.
    private readonly float[][] _convHistory;
    private readonly float[][] _ssmState;
    private int _disposed;

    public GgufMetadata Metadata => _handle.Metadata;
    public int DModel { get; }
    public int NumLayers { get; }
    public int DInner { get; }
    public int DState { get; }
    public int ConvKernel { get; }
    public int DtRank { get; }
    public int VocabSize { get; }
    public float Eps { get; }

    private MambaModel(GgufModelLoader.LoadedGgufModel handle, IReadOnlyDictionary<string, Tensor> w,
        int dModel, int layers, int dInner, int dState, int convK, int dtRank, int vocab, float eps)
    {
        _handle = handle; _w = w;
        DModel = dModel; NumLayers = layers; DInner = dInner; DState = dState; ConvKernel = convK; DtRank = dtRank;
        VocabSize = vocab; Eps = eps <= 0f ? 1e-5f : eps;
        _convHistory = new float[layers][];
        _ssmState = new float[layers][];
        for (int i = 0; i < layers; i++)
        {
            _convHistory[i] = new float[(convK - 1) * dInner];
            _ssmState[i] = new float[dInner * dState];
        }
    }

    /// <summary>Zeroes every layer's recurrent state — call before starting a new, unrelated generation.</summary>
    public void ResetState()
    {
        foreach (float[] h in _convHistory) Array.Clear(h);
        foreach (float[] s in _ssmState) Array.Clear(s);
    }

    public static MambaModel Load(string ggufPath)
    {
        (Dictionary<string, Tensor> w, GgufModelLoader.LoadedGgufModel handle) = GgufModelLoader.LoadDequantized(ggufPath, DType.F32);
        try
        {
            // Relabel the nn.Linear weights (+ token_embd, used as the tied lm_head) to [out, in]. The raw SSM
            // parameters (ssm_a, ssm_conv1d.weight, ssm_d) are read directly with explicit indexing — no relabel.
            foreach (string key in w.Keys.ToList())
            {
                if (!key.EndsWith(".weight", StringComparison.Ordinal) || w[key].Shape.Rank != 2) continue;
                if (key == "token_embd.weight" || key.Contains("ssm_in") || key.Contains("ssm_x") || key.Contains("ssm_dt") || key.Contains("ssm_out"))
                    w[key] = TensorCasts.RelabelRank2Copy(w[key]);
            }
            GgufMetadata m = handle.Metadata;
            int dModel = (int)m.GetUInt32("mamba.embedding_length");
            int layers = (int)m.GetUInt32("mamba.block_count");
            int dInner = (int)m.GetUInt32("mamba.ssm.inner_size");
            int dState = (int)m.GetUInt32("mamba.ssm.state_size");
            int convK = (int)m.GetUInt32("mamba.ssm.conv_kernel");
            int dtRank = (int)m.GetUInt32("mamba.ssm.time_step_rank");
            float eps = m.GetFloat32("mamba.attention.layer_norm_rms_epsilon", 1e-5f);
            int vocab = (int)w["token_embd.weight"].Shape[0];   // relabeled to [vocab, dModel]
            return new MambaModel(handle, w, dModel, layers, dInner, dState, convK, dtRank, vocab, eps);
        }
        catch { handle.Dispose(); throw; }
    }

    private Tensor W(string key) => _w[key];

    /// <summary>Runs the stack over <paramref name="ids"/> — the NEW tokens since the last call — advancing each layer's carried recurrent state, and returns the next-token logits (last position); call <see cref="ResetState"/> before the first call of a new generation.</summary>
    public float[] ForwardLastLogits(IBackend backend, IReadOnlyList<int> ids)
    {
        int seq = ids.Count, d = DModel;
        // Token embedding (gather rows of the relabeled [vocab, dModel] table).
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
        // Final norm + tied lm_head on the last position.
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
        int d = DModel, di = DInner, ds = DState, k = ConvKernel, dtr = DtRank;
        string p = $"blk.{i}";

        // RMSNorm.
        Tensor xn = new(new TensorShape(1, seq, d), DType.F32);
        backend.RmsNorm(xn, hIn, W($"{p}.attn_norm.weight"), Eps);

        // in_proj → [seq, 2*di]; split x, z.
        Tensor xz = new(new TensorShape(1, seq, 2 * di), DType.F32);
        backend.Linear(xz, xn, W($"{p}.ssm_in.weight"), null);
        xn.Dispose();
        backend.Sync();
        float* xzp = (float*)xz.DataPointer;
        float[] x = new float[seq * di], z = new float[seq * di];
        for (int s = 0; s < seq; s++)
        {
            for (int c = 0; c < di; c++) x[s * di + c] = xzp[(long)s * 2 * di + c];
            for (int c = 0; c < di; c++) z[s * di + c] = xzp[(long)s * 2 * di + di + c];
        }
        xz.Dispose();

        // Causal depthwise Conv1d (kernel k, left context = the (k-1) rows carried from the previous call) + SiLU.
        float[] history = _convHistory[i];
        float* convW = (float*)W($"{p}.ssm_conv1d.weight").DataPointer;   // [di, k]
        float* convB = (float*)W($"{p}.ssm_conv1d.bias").DataPointer;     // [di]
        float[] xc = new float[seq * di];
        for (int s = 0; s < seq; s++)
            for (int c = 0; c < di; c++)
            {
                float acc = convB[c];
                for (int j = 0; j < k; j++)
                {
                    int combined = s + j;
                    acc += combined < k - 1 ? convW[c * k + j] * history[combined * di + c]
                        : convW[c * k + j] * x[(combined - (k - 1)) * di + c];
                }
                xc[s * di + c] = acc / (1f + MathF.Exp(-acc));   // SiLU
            }
        if (k > 1)
        {
            int carry = Math.Min(k - 1, seq);
            for (int r = 0; r < k - 1 - carry; r++)
                Array.Copy(history, (r + carry) * di, history, r * di, di);
            for (int r = 0; r < carry; r++)
                Array.Copy(x, (seq - carry + r) * di, history, (k - 1 - carry + r) * di, di);
        }

        // x_proj → [seq, dtr + 2*ds]; split dt, B, C.
        Tensor xcT = new(new TensorShape(1, seq, di), DType.F32);
        fixed (float* src = xc) Buffer.MemoryCopy(src, (void*)xcT.DataPointer, (long)seq * di * 4, (long)seq * di * 4);
        Tensor xdbl = new(new TensorShape(1, seq, dtr + 2 * ds), DType.F32);
        backend.Linear(xdbl, xcT, W($"{p}.ssm_x.weight"), null);
        backend.Sync();
        float* xd = (float*)xdbl.DataPointer;
        float[] dtIn = new float[seq * dtr], Bm = new float[seq * ds], Cm = new float[seq * ds];
        for (int s = 0; s < seq; s++)
        {
            long b = (long)s * (dtr + 2 * ds);
            for (int c = 0; c < dtr; c++) dtIn[s * dtr + c] = xd[b + c];
            for (int c = 0; c < ds; c++) Bm[s * ds + c] = xd[b + dtr + c];
            for (int c = 0; c < ds; c++) Cm[s * ds + c] = xd[b + dtr + ds + c];
        }
        xdbl.Dispose();

        // dt_proj → [seq, di]; + bias; softplus → δ.
        Tensor dtInT = new(new TensorShape(1, seq, dtr), DType.F32);
        fixed (float* src = dtIn) Buffer.MemoryCopy(src, (void*)dtInT.DataPointer, (long)seq * dtr * 4, (long)seq * dtr * 4);
        Tensor delta = new(new TensorShape(1, seq, di), DType.F32);
        backend.Linear(delta, dtInT, W($"{p}.ssm_dt.weight"), W($"{p}.ssm_dt.bias"));
        backend.Sync();
        dtInT.Dispose(); xcT.Dispose();
        float* dp = (float*)delta.DataPointer;
        for (long n = 0; n < (long)seq * di; n++) { float v = dp[n]; dp[n] = v > 20f ? v : MathF.Log(1f + MathF.Exp(v)); }

        // Selective scan: h[di, ds]; A = -exp(A_log). y[t,i] = Σ_s C[t,s]·h[i,s] + D[i]·x[t,i].
        // GGUF ssm_a already stores A = -exp(A_log) (llama.cpp bakes the -exp at conversion), so use it directly.
        float* A = (float*)W($"{p}.ssm_a").DataPointer;      // [di, ds] = A
        float* dD = (float*)W($"{p}.ssm_d").DataPointer;     // [di]
        float[] state = _ssmState[i];   // carried across calls — zero only at true position 0
        float[] y = new float[seq * di];
        for (int s = 0; s < seq; s++)
            for (int c = 0; c < di; c++)
            {
                float dti = dp[s * di + c], xi = xc[s * di + c], acc = 0f;
                long sb = (long)c * ds;
                for (int st = 0; st < ds; st++)
                {
                    float dA = MathF.Exp(dti * A[c * ds + st]);
                    float dBx = dti * Bm[s * ds + st] * xi;
                    float hs = dA * state[sb + st] + dBx;
                    state[sb + st] = hs;
                    acc += Cm[s * ds + st] * hs;
                }
                y[s * di + c] = acc + dD[c] * xi;
            }
        delta.Dispose();

        // Gate: y *= SiLU(z), then out_proj.
        for (long n = 0; n < (long)seq * di; n++) { float zv = z[n]; y[n] *= zv / (1f + MathF.Exp(-zv)); }
        Tensor yT = new(new TensorShape(1, seq, di), DType.F32);
        fixed (float* src = y) Buffer.MemoryCopy(src, (void*)yT.DataPointer, (long)seq * di * 4, (long)seq * di * 4);
        Tensor outp = new(new TensorShape(1, seq, d), DType.F32);
        backend.Linear(outp, yT, W($"{p}.ssm_out.weight"), null);
        yT.Dispose();

        // Residual.
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
