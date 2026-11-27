using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Swin;

/// <summary>Shared Swin-Transformer block math (Liu et al. 2021): shifted-window multi-head self-attention with
/// relative-position bias, the post-attention MLP, and the 2× patch merging between stages. Token grids are
/// <c>[1, H·W, C]</c> row-major; the window-attention core runs as one batched SDPA (windows = batch) with the
/// relative-position bias and the cyclic-shift region mask folded into an additive score mask. Weight layouts and
/// key names are owned by each caller; this file only does the arithmetic and the host layout shuffles.</summary>
internal static unsafe class SwinOps
{
    /// <summary>One Swin block's projections: the separate q/k/v layout (callers split fused qkv at load).</summary>
    internal sealed class BlockWeights
    {
        public Tensor? LnBeforeW, LnBeforeB, QW, QB, KW, KB, VW, VB, RelBiasTable, OW, OB, LnAfterW, LnAfterB, Fc1W, Fc1B, Fc2W, Fc2B;
    }

    /// <summary>One Swin block: pre-norm windowed attention with residual, then a post-norm MLP with residual.
    /// Input and output are <c>[1, h·w, dim]</c>; <paramref name="shift"/> is 0 for regular windows or ws/2 for shifted.</summary>
    public static Tensor Block(IBackend backend, Tensor input, int h, int wdt, int dim, int heads, int ws, int shift, float eps, BlockWeights blk)
    {
        // layernorm_before
        Tensor normed = new(input.Shape, DType.F32);
        backend.LayerNorm(normed, input, blk.LnBeforeW!, blk.LnBeforeB!, eps);

        // reshape [1,HW,C] -> host [H,W,C], pad to multiple of window, cyclic shift(-shift)
        int padR = (ws - wdt % ws) % ws;
        int padB = (ws - h % ws) % ws;
        int hp = h + padB, wp = wdt + padR;
        float[] shifted = new float[(long)hp * wp * dim];
        {
            float* np = (float*)normed.DataPointer;
            for (int y = 0; y < hp; y++)
                for (int xx = 0; xx < wp; xx++)
                {
                    // cyclic shift: source position = (y+shift, x+shift) mod, only within valid region (pad=0)
                    int sy = y, sx = xx;
                    if (shift > 0) { sy = (y + shift) % hp; sx = (xx + shift) % wp; }
                    long dst = ((long)y * wp + xx) * dim;
                    if (sy < h && sx < wdt)
                    {
                        long src = ((long)sy * wdt + sx) * dim;
                        for (int c = 0; c < dim; c++) shifted[dst + c] = np[src + c];
                    }
                }
        }
        normed.Dispose();

        // q,k,v projections on the shifted, padded token grid [1, hp*wp, C]
        int nTok = hp * wp;
        Tensor xs = new(new TensorShape(1, nTok, dim), DType.F32);
        fixed (float* sp = shifted) Buffer.MemoryCopy(sp, (void*)xs.DataPointer, (long)nTok * dim * 4, (long)nTok * dim * 4);
        Tensor q = Project(backend, xs, blk.QW!, blk.QB!, nTok, dim);
        Tensor k = Project(backend, xs, blk.KW!, blk.KB!, nTok, dim);
        Tensor v = Project(backend, xs, blk.VW!, blk.VB!, nTok, dim);
        xs.Dispose();

        Tensor context = WindowAttention(backend, q, k, v, hp, wp, dim, heads, ws, shift, blk.RelBiasTable!);
        q.Dispose(); k.Dispose(); v.Dispose();

        // reverse shift + unpad -> [H,W,C] -> [1,HW,C]
        Tensor attnOut = new(new TensorShape(1, (long)h * wdt, dim), DType.F32);
        {
            float* cp = (float*)context.DataPointer;
            float* ap = (float*)attnOut.DataPointer;
            for (int y = 0; y < h; y++)
                for (int xx = 0; xx < wdt; xx++)
                {
                    int sy = y, sx = xx;
                    if (shift > 0) { sy = ((y - shift) % hp + hp) % hp; sx = ((xx - shift) % wp + wp) % wp; }
                    long src = ((long)sy * wp + sx) * dim;
                    long dst = ((long)y * wdt + xx) * dim;
                    for (int c = 0; c < dim; c++) ap[dst + c] = cp[src + c];
                }
        }
        context.Dispose();

        // o_proj + residual (fresh output tensor — backend.Add does not support aliasing an input)
        Tensor oProj = Project(backend, attnOut, blk.OW!, blk.OB!, h * wdt, dim);
        attnOut.Dispose();
        Tensor attnRes = new(oProj.Shape, DType.F32);
        backend.Add(attnRes, oProj, input);
        oProj.Dispose();

        // MLP branch (post-norm residual)
        Tensor lnA = new(attnRes.Shape, DType.F32);
        backend.LayerNorm(lnA, attnRes, blk.LnAfterW!, blk.LnAfterB!, eps);
        int inner = (int)blk.Fc1W!.Shape[0];
        Tensor fc1 = Project(backend, lnA, blk.Fc1W!, blk.Fc1B!, h * wdt, dim, inner);
        lnA.Dispose();
        backend.GeluErf(fc1, fc1);
        Tensor fc2 = Project(backend, fc1, blk.Fc2W!, blk.Fc2B!, h * wdt, inner, dim);
        fc1.Dispose();
        Tensor outT = new(fc2.Shape, DType.F32);
        backend.Add(outT, fc2, attnRes);
        fc2.Dispose();
        attnRes.Dispose();
        return outT;
    }

    /// <summary>Swin patch merging: 2×2 neighbour concat in the order (row0,col0),(row1,col0),(row0,col1),(row1,col1),
    /// then LayerNorm(4C) and a bias-free linear 4C→2C. Odd grids are zero-padded bottom/right first.</summary>
    public static Tensor PatchMerging(IBackend backend, Tensor x, int h, int w, int dim, Tensor normW, Tensor normB, Tensor reductionW, float eps)
    {
        int hp = h + h % 2, wp = w + w % 2;
        int nh = hp / 2, nw = wp / 2;
        int fourC = 4 * dim;
        float* xp = (float*)x.DataPointer;
        Tensor cat = new(new TensorShape(1, (long)nh * nw, fourC), DType.F32);
        float* cp = (float*)cat.DataPointer;
        int[][] rc = [[0, 0], [1, 0], [0, 1], [1, 1]];
        for (int oy = 0; oy < nh; oy++)
            for (int ox = 0; ox < nw; ox++)
            {
                long dstBase = ((long)oy * nw + ox) * fourC;
                for (int g = 0; g < 4; g++)
                {
                    int sy = oy * 2 + rc[g][0], sx = ox * 2 + rc[g][1];
                    long dst = dstBase + (long)g * dim;
                    if (sy < h && sx < w)
                    {
                        long src = ((long)sy * w + sx) * dim;
                        for (int c = 0; c < dim; c++) cp[dst + c] = xp[src + c];
                    }
                    else
                        for (int c = 0; c < dim; c++) cp[dst + c] = 0f;
                }
            }
        Tensor normed = new(cat.Shape, DType.F32);
        backend.LayerNorm(normed, cat, normW, normB, eps);
        cat.Dispose();
        Tensor reduced = new(new TensorShape(1, (long)nh * nw, 2 * dim), DType.F32);
        backend.Linear(reduced, normed, reductionW, null);
        normed.Dispose();
        return reduced;
    }

    /// <summary>LayerNorm over the channel axis of a token grid, then the host transpose to <c>[1, C, H, W]</c>.</summary>
    public static Tensor LayerNormToNchw(IBackend backend, Tensor tokens, int h, int w, int dim, Tensor normW, Tensor normB, float eps)
    {
        Tensor normed = new(tokens.Shape, DType.F32);
        backend.LayerNorm(normed, tokens, normW, normB, eps);
        Tensor map = new(new TensorShape(1, dim, h, w), DType.F32);
        float* np = (float*)normed.DataPointer;
        float* mp = (float*)map.DataPointer;
        int hw = h * w;
        for (int t = 0; t < hw; t++)
            for (int c = 0; c < dim; c++)
                mp[(long)c * hw + t] = np[(long)t * dim + c];
        normed.Dispose();
        return map;
    }

    /// <summary>Host transpose <c>[1, C, H, W]</c> → token grid <c>[1, H·W, C]</c>.</summary>
    public static Tensor ChannelsToTokens(Tensor chw, int c, int hw)
    {
        Tensor outp = new(new TensorShape(1, hw, c), DType.F32);
        float* ip = (float*)chw.DataPointer;
        float* op = (float*)outp.DataPointer;
        for (int cc = 0; cc < c; cc++)
            for (int t = 0; t < hw; t++)
                op[(long)t * c + cc] = ip[(long)cc * hw + t];
        return outp;
    }

    /// <summary>Zero-pads a <c>[1, C, H, W]</c> image bottom/right to multiples of <paramref name="ps"/>; returns the input itself when already aligned.</summary>
    public static Tensor PadToMultiple(Tensor img, int ps, out int pH, out int pW)
    {
        int c = (int)img.Shape[1], h = (int)img.Shape[2], w = (int)img.Shape[3];
        int padB = (ps - h % ps) % ps, padR = (ps - w % ps) % ps;
        pH = h + padB; pW = w + padR;
        if (padB == 0 && padR == 0) return img;
        Tensor outp = new(new TensorShape(1, c, pH, pW), DType.F32);
        float* ip = (float*)img.DataPointer;
        float* op = (float*)outp.DataPointer;
        for (int cc = 0; cc < c; cc++)
            for (int y = 0; y < h; y++)
                Buffer.MemoryCopy(ip + ((long)cc * h + y) * w, op + ((long)cc * pH + y) * pW, (long)w * 4, (long)w * 4);
        return outp;
    }

    /// <summary>Flattened <c>(2·ws−1)²</c>-table index for every query/key pair of one window (row-major <c>[ws², ws²]</c>).</summary>
    public static int[] RelativePositionIndex(int ws)
    {
        int n = ws * ws;
        int[] coordsY = new int[n], coordsX = new int[n];
        for (int y = 0; y < ws; y++)
            for (int x = 0; x < ws; x++)
            {
                coordsY[y * ws + x] = y;
                coordsX[y * ws + x] = x;
            }
        int[] index = new int[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                int ry = coordsY[i] - coordsY[j] + ws - 1;
                int rx = coordsX[i] - coordsX[j] + ws - 1;
                index[i * n + j] = ry * (2 * ws - 1) + rx;
            }
        return index;
    }

    private static Tensor Project(IBackend backend, Tensor input, Tensor weight, Tensor? bias, int rows, int inDim, int outDim)
    {
        Tensor output = new(new TensorShape(1, rows, outDim), DType.F32);
        backend.Linear(output, input, weight, bias);
        return output;
    }

    private static Tensor Project(IBackend backend, Tensor input, Tensor weight, Tensor? bias, int rows, int dim) =>
        Project(backend, input, weight, bias, rows, dim, (int)weight.Shape[0]);

    /// <summary>Batched windowed multi-head self-attention: partitions the (already shifted) grid into non-overlapping
    /// windows — the SDPA batch dimension — runs the attention with a fully materialized additive mask
    /// <c>[nW, heads, wa, wa]</c> (relative-position bias per head, plus a −100 term across cyclic-shift region
    /// boundaries for shifted blocks), merges the heads, and stitches the windows back. Returns the context on the
    /// shifted grid <c>[1, hp*wp, dim]</c>.</summary>
    private static Tensor WindowAttention(IBackend backend, Tensor q, Tensor k, Tensor v, int hp, int wp, int dim, int heads, int ws, int shift, Tensor relTable)
    {
        int hd = dim / heads;
        int wa = ws * ws;
        int nWh = hp / ws, nWw = wp / ws, nW = nWh * nWw;

        Tensor qWin = PartitionWindows(q, hp, wp, ws, dim);
        Tensor kWin = PartitionWindows(k, hp, wp, ws, dim);
        Tensor vWin = PartitionWindows(v, hp, wp, ws, dim);
        Tensor qh = ToHeads(backend, qWin, nW, wa, heads, hd);
        Tensor kh = ToHeads(backend, kWin, nW, wa, heads, hd);
        Tensor vh = ToHeads(backend, vWin, nW, wa, heads, hd);
        qWin.Dispose(); kWin.Dispose(); vWin.Dispose();

        Tensor mask = BuildWindowMask(nW, heads, wa, ws, nWw, shift, relTable, hp, wp);
        Tensor attn = AttentionLayer.Shared(heads, hd, computeDType: DType.F32).AttendHeadMajorOperands(backend, qh, kh, vh, new AttentionCallContext { Bias = mask });
        mask.Dispose();

        // Merge heads [nW, heads, wa, hd] -> [nW, wa, heads*hd], then stitch windows back to the grid.
        Tensor merged = new(new TensorShape(nW, wa, heads, hd), DType.F32);
        backend.Permute0213(merged, attn, heads, wa, hd);
        attn.Dispose();
        Tensor mergedFlat = merged.Reshape(new TensorShape(nW, wa, dim));
        Tensor context = UnpartitionWindows(mergedFlat, hp, wp, ws, dim);
        merged.Dispose();
        return context;
    }

    /// <summary><c>[nW, wa, heads*hd]</c> -> <c>[nW, heads, wa, hd]</c> (reshape to heads, swap the seq/head axes).</summary>
    private static Tensor ToHeads(IBackend backend, Tensor win, int nW, int wa, int heads, int hd)
    {
        Tensor reshaped = win.Reshape(new TensorShape(nW, wa, heads, hd));
        Tensor outT = new(new TensorShape(nW, heads, wa, hd), DType.F32);
        backend.Permute0213(outT, reshaped, wa, heads, hd);
        return outT;
    }

    /// <summary>Builds the additive attention mask <c>[nW, heads, wa, wa]</c>: per-head relative-position bias plus,
    /// for shifted blocks, a −100 term wherever two window positions fall in different cyclic-shift regions. Fully
    /// materialized (element count == score matrix) so the CPU/CUDA SDPA add it directly.</summary>
    private static Tensor BuildWindowMask(int nW, int heads, int wa, int ws, int nWw, int shift, Tensor relTable, int hp, int wp)
    {
        int[] relIndex = RelativePositionIndex(ws);
        int[] regionId = shift > 0 ? ShiftRegionIds(hp, wp, ws, shift) : Array.Empty<int>();
        Tensor mask = new(new TensorShape(nW, heads, wa, wa), DType.F32);
        float* rt = (float*)relTable.DataPointer;   // [(2ws-1)^2, heads]
        float* mp = (float*)mask.DataPointer;
        for (int w = 0; w < nW; w++)
        {
            int baseY = (w / nWw) * ws, baseX = (w % nWw) * ws;
            for (int hh = 0; hh < heads; hh++)
                for (int i = 0; i < wa; i++)
                {
                    int ri = 0;
                    if (shift > 0)
                    {
                        int iy = baseY + i / ws, ix = baseX + i % ws;
                        ri = regionId[(long)iy * wp + ix];
                    }
                    long rowOff = (((long)w * heads + hh) * wa + i) * wa;
                    for (int j = 0; j < wa; j++)
                    {
                        float bias = rt[(long)relIndex[i * wa + j] * heads + hh];
                        if (shift > 0)
                        {
                            int jy = baseY + j / ws, jx = baseX + j % ws;
                            if (regionId[(long)jy * wp + jx] != ri) bias += -100f;
                        }
                        mp[rowOff + j] = bias;
                    }
                }
        }
        return mask;
    }

    /// <summary>Partitions a row-major grid <c>[1, hp*wp, dim]</c> (hp/wp already multiples of <paramref name="ws"/>)
    /// into <c>[nW, ws*ws, dim]</c> windows. Host layout shuffle (no math).</summary>
    private static Tensor PartitionWindows(Tensor grid, int hp, int wp, int ws, int dim)
    {
        int nWw = wp / ws, nWh = hp / ws, nW = nWh * nWw, wa = ws * ws;
        Tensor outT = new(new TensorShape(nW, wa, dim), DType.F32);
        float* src = (float*)grid.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int py = 0; py < hp; py++)
            for (int px = 0; px < wp; px++)
            {
                int win = (py / ws) * nWw + (px / ws);
                int local = (py % ws) * ws + (px % ws);
                long s = ((long)py * wp + px) * dim;
                long d = ((long)win * wa + local) * dim;
                for (int c = 0; c < dim; c++) dst[d + c] = src[s + c];
            }
        return outT;
    }

    /// <summary>Inverse of <see cref="PartitionWindows"/>: gathers <c>[nW, ws*ws, dim]</c> windows back into the
    /// row-major grid <c>[1, hp*wp, dim]</c>.</summary>
    private static Tensor UnpartitionWindows(Tensor windows, int hp, int wp, int ws, int dim)
    {
        int nWw = wp / ws, wa = ws * ws;
        Tensor outT = new(new TensorShape(1, (long)hp * wp, dim), DType.F32);
        float* src = (float*)windows.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int py = 0; py < hp; py++)
            for (int px = 0; px < wp; px++)
            {
                int win = (py / ws) * nWw + (px / ws);
                int local = (py % ws) * ws + (px % ws);
                long s = ((long)win * wa + local) * dim;
                long d = ((long)py * wp + px) * dim;
                for (int c = 0; c < dim; c++) dst[d + c] = src[s + c];
            }
        return outT;
    }

    /// <summary>Region id (0..8) per grid cell for the Swin shift mask, using the reference slice boundaries
    /// <c>[0,-ws) / [-ws,-shift) / [-shift,end)</c> on each axis.</summary>
    private static int[] ShiftRegionIds(int hp, int wp, int ws, int shift)
    {
        int[] hRegion = new int[hp], wRegion = new int[wp];
        for (int y = 0; y < hp; y++) hRegion[y] = (y >= hp - ws ? 1 : 0) + (y >= hp - shift ? 1 : 0);
        for (int x = 0; x < wp; x++) wRegion[x] = (x >= wp - ws ? 1 : 0) + (x >= wp - shift ? 1 : 0);
        int[] ids = new int[hp * wp];
        for (int y = 0; y < hp; y++)
            for (int x = 0; x < wp; x++)
                ids[y * wp + x] = hRegion[y] * 3 + wRegion[x];
        return ids;
    }
}
