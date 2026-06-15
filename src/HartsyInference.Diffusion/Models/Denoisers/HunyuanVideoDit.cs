using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>HunyuanVideo MM-DiT — the base offline-video transformer and the backbone the Hunyuan-GameCraft
/// world model finetunes. Reuses <see cref="HunyuanImageBlock"/> (dual-stream) and
/// <see cref="HunyuanImageSingleBlock"/> (single-stream) with a <b>3-axis (T,H,W) RoPE</b> via the generalized
/// <see cref="HunyuanImageRope"/>. Predicts rectified-flow velocity over the 16-channel latent from a
/// patchified input (16-ch plain, or GameCraft's 33-ch noisy+history+mask), Llava text tokens, a pooled CLIP
/// vector, and a timestep. Optional <c>cameraTokens</c> are token-added to the image stream (GameCraft CameraNet).
/// <para><b>Numerics validation-pending</b> — the txt token-refiner is currently a plain projection (the
/// 2-layer refiner is validation-gated) and timestep scaling / block wiring are reconciled at diff time.</para></summary>
public sealed unsafe class HunyuanVideoDit : IDisposable
{
    private readonly HunyuanVideoConfig _cfg;
    private readonly HunyuanImageBlock[] _double;
    private readonly HunyuanImageSingleBlock[] _single;
    private readonly HunyuanImageRope _rope;

    private Tensor? _imgInW, _imgInB;       // [hidden, inCh*pT*pH*pW]
    private Tensor? _txtInW, _txtInB;       // [hidden, textDim]
    private Tensor? _timeW0, _timeB0, _timeW1, _timeB1;
    private Tensor? _vecW0, _vecB0, _vecW1, _vecB1;
    private Tensor? _finalModW, _finalModB; // [2*hidden, hidden]
    private Tensor? _outW, _outB;           // [outCh*pT*pH*pW, hidden]

    public HunyuanVideoConfig Config => _cfg;

    public HunyuanVideoDit(HunyuanVideoConfig cfg)
    {
        _cfg = cfg;
        _double = new HunyuanImageBlock[cfg.NumDoubleBlocks];
        for (int i = 0; i < cfg.NumDoubleBlocks; i++)
            _double[i] = new HunyuanImageBlock(cfg.HiddenSize, cfg.NumHeads, cfg.HeadDim, cfg.MlpDim);
        _single = new HunyuanImageSingleBlock[cfg.NumSingleBlocks];
        for (int i = 0; i < cfg.NumSingleBlocks; i++)
            _single[i] = new HunyuanImageSingleBlock(cfg.HiddenSize, cfg.NumHeads, cfg.HeadDim, cfg.MlpDim);
        _rope = new HunyuanImageRope(cfg.RopeAxesDim, cfg.RopeTheta);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _imgInW = F32(w[$"{p}img_in.weight"]); _imgInB = F32(w[$"{p}img_in.bias"]);
        _txtInW = F32(w[$"{p}txt_in.weight"]); _txtInB = F32(w[$"{p}txt_in.bias"]);
        _timeW0 = F32(w[$"{p}time_in.0.weight"]); _timeB0 = F32(w[$"{p}time_in.0.bias"]);
        _timeW1 = F32(w[$"{p}time_in.2.weight"]); _timeB1 = F32(w[$"{p}time_in.2.bias"]);
        _vecW0 = F32(w[$"{p}vector_in.0.weight"]); _vecB0 = F32(w[$"{p}vector_in.0.bias"]);
        _vecW1 = F32(w[$"{p}vector_in.2.weight"]); _vecB1 = F32(w[$"{p}vector_in.2.bias"]);
        for (int i = 0; i < _double.Length; i++) _double[i].LoadWeights(w, $"{p}double_blocks.{i}");
        for (int i = 0; i < _single.Length; i++) _single[i].LoadWeights(w, $"{p}single_blocks.{i}");
        _finalModW = F32(w[$"{p}final_layer.mod.weight"]); _finalModB = F32(w[$"{p}final_layer.mod.bias"]);
        _outW = F32(w[$"{p}final_layer.proj.weight"]); _outB = F32(w[$"{p}final_layer.proj.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_imgInW, _imgInB, _txtInW, _txtInB, _timeW0, _timeB0, _timeW1, _timeB1,
            _vecW0, _vecB0, _vecW1, _vecB1, _finalModW, _finalModB, _outW, _outB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (HunyuanImageBlock b in _double) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (HunyuanImageSingleBlock b in _single) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Predicts velocity <c>[B, OutChannels, T, H, W]</c> for the latent <c>[B, InChannels, T, H, W]</c>,
    /// conditioned on Llava text tokens <c>[B, L, TextEmbedDim]</c>, a pooled CLIP vector <c>[B, PooledEmbedDim]</c>,
    /// and <paramref name="timestep"/>. <paramref name="cameraTokens"/> (<c>[B, S_img, HiddenSize]</c>), when
    /// supplied, is added to the image tokens after patchify (GameCraft CameraNet fusion).</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor txt, Tensor pooled, float timestep, Tensor? cameraTokens = null)
    {
        int b = (int)latent.Shape[0];
        int hidden = _cfg.HiddenSize;
        (int pT, int pH, int pW) = _cfg.PatchSize;
        int T = (int)latent.Shape[2], H = (int)latent.Shape[3], W = (int)latent.Shape[4];
        int tOut = T / pT, hOut = H / pH, wOut = W / pW;
        int sImg = tOut * hOut * wOut;
        int patchVec = _cfg.InChannels * pT * pH * pW;

        // 1. Patchify → img tokens.
        Tensor patches = new(new TensorShape(b, sImg, patchVec), DType.F32);
        Patchify(latent, patches, b, _cfg.InChannels, T, H, W, pT, pH, pW, tOut, hOut, wOut);
        Tensor img = new(new TensorShape(b, sImg, hidden), DType.F32);
        backend.Linear(img, patches, _imgInW!, _imgInB!); patches.Dispose();
        if (cameraTokens is not null) { Tensor sum = new(img.Shape, DType.F32); backend.Add(sum, img, cameraTokens); img.Dispose(); img = sum; }

        // 2. Text tokens.
        int L = (int)txt.Shape[1];
        Tensor txtTok = new(new TensorShape(b, L, hidden), DType.F32);
        backend.Linear(txtTok, txt, _txtInW!, _txtInB!);

        // 3. temb = time(timestep) + vector(pooled).
        Tensor temb = BuildTemb(backend, b, hidden, timestep, pooled);

        // 4. Blocks (3D rope, packed (tOut,hOut,wOut)).
        for (int i = 0; i < _double.Length; i++)
        {
            (Tensor ni, Tensor nt) = _double[i].Forward(backend, img, txtTok, temb, _rope, hOut, wOut, tOut);
            img.Dispose(); txtTok.Dispose(); img = ni; txtTok = nt;
        }
        for (int i = 0; i < _single.Length; i++)
        {
            (Tensor ni, Tensor nt) = _single[i].Forward(backend, img, txtTok, temb, _rope, hOut, wOut, tOut);
            img.Dispose(); txtTok.Dispose(); img = ni; txtTok = nt;
        }
        txtTok.Dispose();

        // 5. Final AdaLN-continuous → proj → unpatchify.
        Tensor mod = new(new TensorShape(b, 2 * hidden), DType.F32);
        Tensor tAct = new(temb.Shape, DType.F32); backend.Silu(tAct, temb); temb.Dispose();
        backend.Linear(mod, tAct, _finalModW!, _finalModB!); tAct.Dispose();
        Tensor normed = new(img.Shape, DType.F32); DiTUtils.LayerNormNoAffine(normed, img, b, sImg, hidden); img.Dispose();
        Modulate(normed, mod, b, sImg, hidden); mod.Dispose();

        int outVec = _cfg.OutChannels * pT * pH * pW;
        Tensor projected = new(new TensorShape(b, sImg, outVec), DType.F32);
        backend.Linear(projected, normed, _outW!, _outB!); normed.Dispose();

        Tensor velocity = new(new TensorShape([(long)b, _cfg.OutChannels, T, H, W]), DType.F32);
        Unpatchify(projected, velocity, b, _cfg.OutChannels, T, H, W, pT, pH, pW, tOut, hOut, wOut);
        projected.Dispose();
        return velocity;
    }

    private Tensor BuildTemb(IBackend backend, int b, int hidden, float timestep, Tensor pooled)
    {
        Tensor sin = new(new TensorShape(b, 256), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sin, timestep, b, 256);
        Tensor t0 = new(new TensorShape(b, hidden), DType.F32); backend.Linear(t0, sin, _timeW0!, _timeB0!); sin.Dispose();
        Tensor t0a = new(t0.Shape, DType.F32); backend.Silu(t0a, t0); t0.Dispose();
        Tensor tTime = new(new TensorShape(b, hidden), DType.F32); backend.Linear(tTime, t0a, _timeW1!, _timeB1!); t0a.Dispose();

        Tensor v0 = new(new TensorShape(b, hidden), DType.F32); backend.Linear(v0, pooled, _vecW0!, _vecB0!);
        Tensor v0a = new(v0.Shape, DType.F32); backend.Silu(v0a, v0); v0.Dispose();
        Tensor tVec = new(new TensorShape(b, hidden), DType.F32); backend.Linear(tVec, v0a, _vecW1!, _vecB1!); v0a.Dispose();

        Tensor temb = new(new TensorShape(b, hidden), DType.F32); backend.Add(temb, tTime, tVec);
        tTime.Dispose(); tVec.Dispose();
        return temb;
    }

    private static void Patchify(Tensor latent, Tensor patches, int b, int c, int T, int H, int W,
        int pT, int pH, int pW, int tOut, int hOut, int wOut)
    {
        float* lp = (float*)latent.DataPointer; float* dp = (float*)patches.DataPointer;
        int patchVec = c * pT * pH * pW;
        for (int bb = 0; bb < b; bb++)
            for (int ti = 0; ti < tOut; ti++) for (int hi = 0; hi < hOut; hi++) for (int wi = 0; wi < wOut; wi++)
            {
                long token = ((long)ti * hOut + hi) * wOut + wi;
                long dstBase = ((long)bb * (tOut * hOut * wOut) + token) * patchVec;
                int idx = 0;
                for (int cc = 0; cc < c; cc++) for (int kt = 0; kt < pT; kt++) for (int kh = 0; kh < pH; kh++) for (int kw = 0; kw < pW; kw++)
                {
                    long src = ((((long)bb * c + cc) * T + (ti * pT + kt)) * H + (hi * pH + kh)) * W + (wi * pW + kw);
                    dp[dstBase + idx++] = lp[src];
                }
            }
    }

    private static void Unpatchify(Tensor tokens, Tensor outVol, int b, int oc, int T, int H, int W,
        int pT, int pH, int pW, int tOut, int hOut, int wOut)
    {
        float* tp = (float*)tokens.DataPointer; float* op = (float*)outVol.DataPointer;
        int outVec = oc * pT * pH * pW;
        for (int bb = 0; bb < b; bb++)
            for (int ti = 0; ti < tOut; ti++) for (int hi = 0; hi < hOut; hi++) for (int wi = 0; wi < wOut; wi++)
            {
                long token = ((long)ti * hOut + hi) * wOut + wi;
                long srcBase = ((long)bb * (tOut * hOut * wOut) + token) * outVec;
                int idx = 0;
                for (int cc = 0; cc < oc; cc++) for (int kt = 0; kt < pT; kt++) for (int kh = 0; kh < pH; kh++) for (int kw = 0; kw < pW; kw++)
                {
                    long dst = ((((long)bb * oc + cc) * T + (ti * pT + kt)) * H + (hi * pH + kh)) * W + (wi * pW + kw);
                    op[dst] = tp[srcBase + idx++];
                }
            }
    }

    /// <summary>In-place AdaLN-continuous: <c>x = x*(1+scale)+shift</c> from packed mod <c>[B, 2*hidden]</c>.</summary>
    private static void Modulate(Tensor x, Tensor mod, int b, int s, int hidden)
    {
        float* xp = (float*)x.DataPointer; float* mp = (float*)mod.DataPointer;
        for (int bb = 0; bb < b; bb++)
        {
            float* shift = mp + (long)bb * 2 * hidden;
            float* scale = shift + hidden;
            for (int ss = 0; ss < s; ss++)
            {
                float* row = xp + ((long)bb * s + ss) * hidden;
                for (int c = 0; c < hidden; c++) row[c] = row[c] * (1f + scale[c]) + shift[c];
            }
        }
    }

    internal static Tensor F32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;

    public void Dispose() { }
}
