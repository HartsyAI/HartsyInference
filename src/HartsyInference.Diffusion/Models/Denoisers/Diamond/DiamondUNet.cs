using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.Diamond;

/// <summary>Shared DIAMOND block ops: same-shape/strided/1×1 convolution, GroupNorm group count, the
/// AdaGroupNorm conditioning (no-affine group norm then a cond-driven per-channel scale/shift), and the
/// mid-block self-attention. Mirrors <c>eloialonso/diamond</c> <c>models/blocks.py</c>.</summary>
internal static unsafe class DiamondOps
{
    public static int Groups(int channels, DiamondConfig cfg) => Math.Max(1, channels / cfg.GnGroupSize);

    /// <summary>Conv2D with computed output H/W. <paramref name="kPad"/> is the symmetric padding; kernel size is
    /// read from the weight. Stride <paramref name="s"/>.</summary>
    public static Tensor Conv(IBackend backend, Tensor input, Tensor weight, Tensor? bias, int s, int kPad, int outCh)
    {
        int h = (int)input.Shape[2], w = (int)input.Shape[3];
        int k = (int)weight.Shape[2];
        int oh = (h + 2 * kPad - k) / s + 1, ow = (w + 2 * kPad - k) / s + 1;
        Tensor o = new(new TensorShape(1, outCh, oh, ow), DType.F32);
        backend.Conv2D(o, input, weight, bias, s, s, kPad, kPad);
        return o;
    }

    /// <summary>AdaGroupNorm: no-affine group norm of <paramref name="x"/> <c>[1,C,H,W]</c>, then
    /// <c>x·(1+scale)+shift</c> where <c>[scale,shift]=Linear(cond)</c>. Returns a fresh tensor.</summary>
    public static Tensor AdaGroupNorm(IBackend backend, Tensor x, Tensor cond, Tensor linW, Tensor linB, DiamondConfig cfg)
    {
        int c = (int)x.Shape[1], h = (int)x.Shape[2], w = (int)x.Shape[3];
        long hw = (long)h * w;
        int groups = Groups(c, cfg), cpg = c / groups;
        Tensor ss = new(new TensorShape(1, 2 * c), DType.F32);
        backend.Linear(ss, cond, linW, linB);
        Tensor o = new(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer, sp = (float*)ss.DataPointer;
        for (int g = 0; g < groups; g++)
        {
            long baseC = (long)g * cpg;
            double sum = 0, sumSq = 0;
            long n = cpg * hw;
            for (int cc = 0; cc < cpg; cc++)
                for (long i = 0; i < hw; i++) { float v = xp[(baseC + cc) * hw + i]; sum += v; sumSq += (double)v * v; }
            double mean = sum / n;
            double inv = 1.0 / Math.Sqrt(sumSq / n - mean * mean + cfg.GnEps);
            for (int cc = 0; cc < cpg; cc++)
            {
                long ch = baseC + cc;
                float scale = sp[ch], shift = sp[c + ch];
                for (long i = 0; i < hw; i++)
                    op[ch * hw + i] = (float)((xp[ch * hw + i] - mean) * inv) * (1f + scale) + shift;
            }
        }
        ss.Dispose();
        return o;
    }

    /// <summary>SelfAttention2d: affine GroupNorm → 1×1 QKV → multi-head SDPA over the H·W tokens →
    /// 1×1 out-proj; returns <c>norm(x) + out_proj(attn)</c> (the reference adds the residual to the NORMED x).</summary>
    public static Tensor SelfAttention(IBackend backend, Tensor x, Tensor normW, Tensor normB, Tensor qkvW, Tensor qkvB,
        Tensor outW, Tensor outB, DiamondConfig cfg)
    {
        int c = (int)x.Shape[1], h = (int)x.Shape[2], w = (int)x.Shape[3], hw = h * w;
        int nHead = Math.Max(1, c / cfg.AttnHeadDim), hd = c / nHead;
        Tensor xn = new(x.Shape, DType.F32);
        backend.GroupNorm(xn, x, normW, normB, Groups(c, cfg), cfg.GnEps);
        Tensor qkv = Conv(backend, xn, qkvW, qkvB, 1, 0, 3 * c);   // [1,3C,H,W]

        // slice q/k/v channels → [1, nHead, hw, hd]
        Tensor q = SplitHeads(qkv, 0 * c, c, nHead, hd, hw);
        Tensor k = SplitHeads(qkv, 1 * c, c, nHead, hd, hw);
        Tensor v = SplitHeads(qkv, 2 * c, c, nHead, hd, hw);
        qkv.Dispose();
        Tensor att = new(new TensorShape(1, nHead, hw, hd), DType.F32);
        backend.ScaledDotProductAttention(att, q, k, v, null, 1f / MathF.Sqrt(hd));
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor y = MergeHeads(att, c, nHead, hd, h, w); att.Dispose();     // [1,C,H,W]
        Tensor proj = Conv(backend, y, outW, outB, 1, 0, c); y.Dispose();  // out_proj (1x1)
        Tensor o = new(x.Shape, DType.F32); backend.Add(o, xn, proj);
        xn.Dispose(); proj.Dispose();
        return o;
    }

    private static Tensor SplitHeads(Tensor qkv, int chOffset, int c, int nHead, int hd, int hw)
    {
        Tensor o = new(new TensorShape(1, nHead, hw, hd), DType.F32);
        float* src = (float*)qkv.DataPointer + (long)chOffset * hw;
        float* dst = (float*)o.DataPointer;
        for (int head = 0; head < nHead; head++)
            for (int i = 0; i < hw; i++)
                for (int d = 0; d < hd; d++)
                    dst[((long)head * hw + i) * hd + d] = src[((long)head * hd + d) * hw + i];
        return o;
    }

    private static Tensor MergeHeads(Tensor att, int c, int nHead, int hd, int h, int w)
    {
        int hw = h * w;
        Tensor o = new(new TensorShape(1, c, h, w), DType.F32);
        float* src = (float*)att.DataPointer, dst = (float*)o.DataPointer;
        for (int head = 0; head < nHead; head++)
            for (int i = 0; i < hw; i++)
                for (int d = 0; d < hd; d++)
                    dst[((long)head * hd + d) * hw + i] = src[((long)head * hw + i) * hd + d];
        return o;
    }
}

/// <summary>DIAMOND ResBlock: <c>proj(x) + conv2(silu(adaGN2(conv1(silu(adaGN1(x))))))</c>, optionally followed
/// by self-attention. AdaGroupNorm carries the conditioning; conv2 is zero-init at training. The optional 1×1
/// <c>proj</c> matches channel counts (present when in≠out).</summary>
internal sealed class DiamondResBlock
{
    private readonly DiamondConfig _cfg;
    private readonly int _in, _out;
    private readonly bool _attn;
    private Tensor? _norm1LinW, _norm1LinB, _conv1W, _conv1B, _norm2LinW, _norm2LinB, _conv2W, _conv2B, _projW, _projB;
    private Tensor? _attnNormW, _attnNormB, _attnQkvW, _attnQkvB, _attnOutW, _attnOutB;

    public DiamondResBlock(int inCh, int outCh, bool attn, DiamondConfig cfg) { _in = inCh; _out = outCh; _attn = attn; _cfg = cfg; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _norm1LinW = F(w[$"{p}.norm1.linear.weight"]); _norm1LinB = F(w[$"{p}.norm1.linear.bias"]);
        _conv1W = F(w[$"{p}.conv1.weight"]); _conv1B = F(w[$"{p}.conv1.bias"]);
        _norm2LinW = F(w[$"{p}.norm2.linear.weight"]); _norm2LinB = F(w[$"{p}.norm2.linear.bias"]);
        _conv2W = F(w[$"{p}.conv2.weight"]); _conv2B = F(w[$"{p}.conv2.bias"]);
        if (_in != _out) { _projW = F(w[$"{p}.proj.weight"]); _projB = F(w[$"{p}.proj.bias"]); }
        if (_attn)
        {
            _attnNormW = F(w[$"{p}.attn.norm.norm.weight"]); _attnNormB = F(w[$"{p}.attn.norm.norm.bias"]);
            _attnQkvW = F(w[$"{p}.attn.qkv_proj.weight"]); _attnQkvB = F(w[$"{p}.attn.qkv_proj.bias"]);
            _attnOutW = F(w[$"{p}.attn.out_proj.weight"]); _attnOutB = F(w[$"{p}.attn.out_proj.bias"]);
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_norm1LinW, _norm1LinB, _conv1W, _conv1B, _norm2LinW, _norm2LinB, _conv2W, _conv2B, _projW, _projB,
            _attnNormW, _attnNormB, _attnQkvW, _attnQkvB, _attnOutW, _attnOutB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor x, Tensor cond)
    {
        Tensor r = _in != _out ? DiamondOps.Conv(backend, x, _projW!, _projB, 1, 0, _out) : x;

        Tensor n1 = DiamondOps.AdaGroupNorm(backend, x, cond, _norm1LinW!, _norm1LinB!, _cfg);
        Tensor s1 = new(n1.Shape, DType.F32); backend.Silu(s1, n1); n1.Dispose();
        Tensor h1 = DiamondOps.Conv(backend, s1, _conv1W!, _conv1B, 1, 1, _out); s1.Dispose();

        Tensor n2 = DiamondOps.AdaGroupNorm(backend, h1, cond, _norm2LinW!, _norm2LinB!, _cfg); h1.Dispose();
        Tensor s2 = new(n2.Shape, DType.F32); backend.Silu(s2, n2); n2.Dispose();
        Tensor h2 = DiamondOps.Conv(backend, s2, _conv2W!, _conv2B, 1, 1, _out); s2.Dispose();

        Tensor sum = new(h2.Shape, DType.F32); backend.Add(sum, h2, r); h2.Dispose();
        if (_in != _out) r.Dispose();
        if (!_attn) return sum;
        Tensor a = DiamondOps.SelfAttention(backend, sum, _attnNormW!, _attnNormB!, _attnQkvW!, _attnQkvB!, _attnOutW!, _attnOutB!, _cfg);
        sum.Dispose();
        return a;
    }

    private static Tensor F(Tensor t) => DiamondInnerModel.F32(t);
}

/// <summary>DIAMOND U-Net: symmetric encoder/decoder of <see cref="DiamondResBlock"/> groups with strided-conv
/// downsamples, nearest-upsample+conv upsamples, a 2-block attention bottleneck, and skip concatenation
/// (each decoder block consumes its encoder counterpart's outputs in reverse). Atari sizes are divisible by
/// <c>2^(levels-1)</c> so no input padding is needed.</summary>
internal sealed unsafe class DiamondUNet
{
    private readonly DiamondConfig _cfg;
    private readonly int _numDown;
    private readonly DiamondResBlock[][] _dBlocks, _uBlocks;
    private readonly DiamondResBlock[] _midBlocks;
    private Tensor?[] _downW, _downB, _upW, _upB;

    public DiamondUNet(DiamondConfig cfg)
    {
        _cfg = cfg;
        int levels = cfg.Channels.Length;
        _numDown = levels - 1;
        _dBlocks = new DiamondResBlock[levels][];
        _uBlocks = new DiamondResBlock[levels][];
        for (int i = 0; i < levels; i++)
        {
            int n = cfg.Depths[i], c1 = cfg.Channels[Math.Max(0, i - 1)], c2 = cfg.Channels[i];
            bool attn = cfg.AttnDepths[i];
            // down: [c1]+[c2]*(n-1) -> [c2]*n
            DiamondResBlock[] d = new DiamondResBlock[n];
            for (int j = 0; j < n; j++) d[j] = new DiamondResBlock(j == 0 ? c1 : c2, c2, attn, cfg);
            _dBlocks[i] = d;
            // up: [2*c2]*n + [c1+c2] -> [c2]*n + [c1]
            DiamondResBlock[] u = new DiamondResBlock[n + 1];
            for (int j = 0; j < n; j++) u[j] = new DiamondResBlock(2 * c2, c2, attn, cfg);
            u[n] = new DiamondResBlock(c1 + c2, c1, attn, cfg);
            _uBlocks[i] = u;
        }
        // u_blocks stored reversed (deepest first), matching nn.ModuleList(reversed(u_blocks))
        Array.Reverse(_uBlocks);

        _midBlocks = [new DiamondResBlock(cfg.Channels[^1], cfg.Channels[^1], true, cfg),
                      new DiamondResBlock(cfg.Channels[^1], cfg.Channels[^1], true, cfg)];
        _downW = new Tensor?[levels]; _downB = new Tensor?[levels];
        _upW = new Tensor?[levels]; _upB = new Tensor?[levels];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        int levels = _cfg.Channels.Length;
        for (int i = 0; i < levels; i++)
        {
            for (int j = 0; j < _dBlocks[i].Length; j++) _dBlocks[i][j].LoadWeights(w, $"{p}.d_blocks.{i}.resblocks.{j}");
            for (int j = 0; j < _uBlocks[i].Length; j++) _uBlocks[i][j].LoadWeights(w, $"{p}.u_blocks.{i}.resblocks.{j}");
            if (i >= 1)
            {
                _downW[i] = F(w[$"{p}.downsamples.{i}.conv.weight"]); _downB[i] = F(w[$"{p}.downsamples.{i}.conv.bias"]);
                _upW[i] = F(w[$"{p}.upsamples.{i}.conv.weight"]); _upB[i] = F(w[$"{p}.upsamples.{i}.conv.bias"]);
            }
        }
        for (int j = 0; j < _midBlocks.Length; j++) _midBlocks[j].LoadWeights(w, $"{p}.mid_blocks.resblocks.{j}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (DiamondResBlock[] lvl in _dBlocks) foreach (DiamondResBlock b in lvl) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (DiamondResBlock[] lvl in _uBlocks) foreach (DiamondResBlock b in lvl) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (DiamondResBlock b in _midBlocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (Tensor?[] arr in new[] { _downW, _downB, _upW, _upB }) foreach (Tensor? t in arr) if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor xIn, Tensor cond)
    {
        int levels = _cfg.Channels.Length;
        Tensor x = xIn;
        bool ownX = false;
        // encoder: each level downsamples (identity at level 0) then runs its resblocks; keep (x_down, block outs)
        List<Tensor[]> dOutputs = new();
        List<Tensor> toDispose = new();
        for (int i = 0; i < levels; i++)
        {
            Tensor xDown = i == 0 ? x : DiamondOps.Conv(backend, x, _downW[i]!, _downB[i], 2, 1, _cfg.Channels[i]);
            if (ownX) x.Dispose();
            Tensor cur = xDown;
            Tensor[] blockOuts = new Tensor[_dBlocks[i].Length];
            for (int j = 0; j < _dBlocks[i].Length; j++) { cur = _dBlocks[i][j].Forward(backend, cur, cond); blockOuts[j] = cur; }
            // save (xDown, blockOuts...) as skip set
            Tensor[] set = new Tensor[1 + blockOuts.Length];
            set[0] = xDown; Array.Copy(blockOuts, 0, set, 1, blockOuts.Length);
            dOutputs.Add(set);
            x = cur; ownX = false;   // cur is blockOuts[^1], owned by the skip set
        }

        // bottleneck (mid input is the last encoder block's output, kept in its skip set → do not dispose it)
        Tensor mid = x;
        for (int j = 0; j < _midBlocks.Length; j++) { Tensor n = _midBlocks[j].Forward(backend, mid, cond); if (j > 0) mid.Dispose(); mid = n; }
        x = mid; ownX = true;

        // decoder: reversed skips; each level upsamples (identity at deepest) then runs resblocks with skip concat
        for (int idx = 0; idx < levels; idx++)
        {
            Tensor[] skip = dOutputs[levels - 1 - idx];     // reversed(d_outputs)
            Tensor cur;
            bool ownCur;
            if (idx == 0) { cur = x; ownCur = ownX; }        // identity upsample — keep x as cur
            else { cur = Upsample(backend, x, idx); if (ownX) x.Dispose(); ownCur = true; }   // upsamples[idx]
            DiamondResBlock[] u = _uBlocks[idx];
            for (int j = 0; j < u.Length; j++)
            {
                Tensor skipT = skip[skip.Length - 1 - j];   // skip[::-1]
                Tensor catT = ConcatChannels(cur, skipT);
                if (ownCur) cur.Dispose();
                Tensor next = u[j].Forward(backend, catT, cond); catT.Dispose();
                cur = next; ownCur = true;
            }
            x = cur; ownX = true;
        }

        // dispose all skip tensors (level-0 xDown == the U-Net input xIn, owned here — the caller must not dispose it)
        foreach (Tensor[] set in dOutputs) foreach (Tensor t in set) t.Dispose();
        return x;
    }

    private Tensor Upsample(IBackend backend, Tensor x, int level)
    {
        int c = (int)x.Shape[1], h = (int)x.Shape[2], w = (int)x.Shape[3];
        Tensor up = new(new TensorShape(1, c, h * 2, w * 2), DType.F32);
        backend.UpsampleNearest2D(up, x, 2, 2);
        Tensor conv = DiamondOps.Conv(backend, up, _upW[level]!, _upB[level], 1, 1, c); up.Dispose();
        return conv;
    }

    private static Tensor ConcatChannels(Tensor a, Tensor b)
    {
        int ca = (int)a.Shape[1], cb = (int)b.Shape[1], h = (int)a.Shape[2], w = (int)a.Shape[3];
        long hw = (long)h * w;
        Tensor o = new(new TensorShape(1, ca + cb, h, w), DType.F32);
        Buffer.MemoryCopy((float*)a.DataPointer, (float*)o.DataPointer, ca * hw * 4, ca * hw * 4);
        Buffer.MemoryCopy((float*)b.DataPointer, (float*)o.DataPointer + ca * hw, cb * hw * 4, cb * hw * 4);
        return o;
    }

    private static Tensor F(Tensor t) => DiamondInnerModel.F32(t);
}
