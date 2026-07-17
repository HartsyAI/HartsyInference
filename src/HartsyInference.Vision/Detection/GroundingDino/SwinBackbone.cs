using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Faithful Swin-Transformer-Tiny backbone for Grounding DINO (<c>model.backbone.conv_encoder.model.*</c>):
/// 4x4 patch embed → 96-dim, four stages (depths [2,2,6,2], heads [3,6,12,24]) of shifted-window multi-head
/// self-attention with relative-position bias and patch merging between stages. Exports the stage2/3/4
/// pre-downsampling feature maps (channels 192/384/768), each passed through its <c>hidden_states_norms</c> LayerNorm
/// and returned as <c>[1, C, H, W]</c>.
///
/// <para>Windowing, cyclic shift, padding and patch merging run as host loops (matching the engine's Whisper/CLIP
/// helper idiom); the per-token linear projections, LayerNorms and the patch-embed convolution route through
/// <see cref="IBackend"/>. Window attention (scores + relative bias + shift mask + softmax) is computed on the host
/// over the small window token blocks.</para></summary>
public sealed unsafe class SwinBackbone : IDisposable
{
    private readonly GroundingDinoConfig _cfg;
    private readonly int _numStages;
    private Tensor? _patchProjW, _patchProjB, _embNormW, _embNormB;
    private readonly SwinStageWeights[] _stages;
    private readonly Dictionary<string, Tensor> _outNorms = new();   // "stage2/3/4" -> {W,B} via suffix
    private int _disposed;

    public SwinBackbone(GroundingDinoConfig cfg)
    {
        _cfg = cfg;
        _numStages = cfg.SwinDepths.Length;
        _stages = new SwinStageWeights[_numStages];
        for (int s = 0; s < _numStages; s++)
            _stages[s] = new SwinStageWeights();
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model.backbone.conv_encoder.model")
    {
        _patchProjW = w[$"{prefix}.embeddings.patch_embeddings.projection.weight"];
        _patchProjB = w[$"{prefix}.embeddings.patch_embeddings.projection.bias"];
        _embNormW = w[$"{prefix}.embeddings.norm.weight"];
        _embNormB = w[$"{prefix}.embeddings.norm.bias"];
        for (int s = 0; s < _numStages; s++)
        {
            string sp = $"{prefix}.encoder.layers.{s}";
            SwinStageWeights sw = _stages[s];
            int depth = _cfg.SwinDepths[s];
            sw.Blocks = new SwinBlockWeights[depth];
            for (int b = 0; b < depth; b++)
            {
                string bp = $"{sp}.blocks.{b}";
                sw.Blocks[b] = new SwinBlockWeights
                {
                    LnBeforeW = w[$"{bp}.layernorm_before.weight"], LnBeforeB = w[$"{bp}.layernorm_before.bias"],
                    QW = w[$"{bp}.attention.self.query.weight"], QB = w[$"{bp}.attention.self.query.bias"],
                    KW = w[$"{bp}.attention.self.key.weight"], KB = w[$"{bp}.attention.self.key.bias"],
                    VW = w[$"{bp}.attention.self.value.weight"], VB = w[$"{bp}.attention.self.value.bias"],
                    RelBiasTable = w[$"{bp}.attention.self.relative_position_bias_table"],
                    OW = w[$"{bp}.attention.output.dense.weight"], OB = w[$"{bp}.attention.output.dense.bias"],
                    LnAfterW = w[$"{bp}.layernorm_after.weight"], LnAfterB = w[$"{bp}.layernorm_after.bias"],
                    Fc1W = w[$"{bp}.intermediate.dense.weight"], Fc1B = w[$"{bp}.intermediate.dense.bias"],
                    Fc2W = w[$"{bp}.output.dense.weight"], Fc2B = w[$"{bp}.output.dense.bias"],
                };
            }
            if (s < _numStages - 1)
            {
                sw.DownReductionW = w[$"{sp}.downsample.reduction.weight"];
                sw.DownNormW = w[$"{sp}.downsample.norm.weight"];
                sw.DownNormB = w[$"{sp}.downsample.norm.bias"];
            }
        }
        for (int i = 0; i < _cfg.BackboneOutChannels.Length; i++)
        {
            string stageName = $"stage{i + 2}";
            _outNorms[$"{stageName}.weight"] = w[$"{prefix}.hidden_states_norms.{stageName}.weight"];
            _outNorms[$"{stageName}.bias"] = w[$"{prefix}.hidden_states_norms.{stageName}.bias"];
        }
    }

    /// <summary>Runs the backbone. Returns the three exported feature maps as <c>[1, C, H, W]</c> (stage2/3/4).</summary>
    public Tensor[] Forward(IBackend backend, Tensor pixelValues)
    {
        int ps = _cfg.SwinPatchSize;
        int inH = (int)pixelValues.Shape[2], inW = (int)pixelValues.Shape[3];
        // maybe_pad pixel_values so H,W divisible by patch size (bottom/right, zeros).
        Tensor padded = PadImage(pixelValues, ps, out int pH, out int pW);
        int gH = pH / ps, gW = pW / ps, embed = _cfg.SwinEmbedDim;

        Tensor conv = new(new TensorShape(1, embed, gH, gW), DType.F32);
        backend.Conv2D(conv, padded, _patchProjW!, _patchProjB, ps, ps, 0, 0);
        if (!ReferenceEquals(padded, pixelValues)) padded.Dispose();

        // [1,embed,gH,gW] -> [1, gH*gW, embed]
        int tokens = gH * gW;
        Tensor x = ChannelsToTokens(backend, conv, embed, tokens);
        conv.Dispose();
        Tensor xn = new(x.Shape, DType.F32);
        backend.LayerNorm(xn, x, _embNormW!, _embNormB!, _cfg.SwinLayerNormEps);
        x.Dispose();
        x = xn;

        int curH = gH, curW = gW, dim = embed;
        List<Tensor> outputs = new();
        for (int s = 0; s < _numStages; s++)
        {
            int heads = _cfg.SwinNumHeads[s];
            SwinBlockWeights[] blocks = _stages[s].Blocks;
            for (int b = 0; b < blocks.Length; b++)
            {
                int shift = (b % 2 == 1) ? _cfg.SwinWindowSize / 2 : 0;
                Tensor next = Block(backend, x, curH, curW, dim, heads, blocks[b], shift);
                x.Dispose();
                x = next;
            }

            // Export before-downsampling output for stages 1,2,3 (named stage2/3/4).
            if (s >= 1)
            {
                string stageName = $"stage{s + 1}";
                outputs.Add(ExportFeatureMap(backend, x, curH, curW, dim, stageName));
            }

            if (s < _numStages - 1)
            {
                Tensor merged = PatchMerging(backend, x, curH, curW, dim, _stages[s]);
                x.Dispose();
                x = merged;
                curH = (curH + 1) / 2;
                curW = (curW + 1) / 2;
                dim *= 2;
            }
        }
        x.Dispose();
        return outputs.ToArray();
    }

    // One Swin block. Even blocks use regular windows (shift 0); odd blocks shift by window/2.
    private Tensor Block(IBackend backend, Tensor input, int h, int wdt, int dim, int heads, SwinBlockWeights blk, int shift)
    {
        int ws = _cfg.SwinWindowSize;

        // layernorm_before
        Tensor normed = new(input.Shape, DType.F32);
        backend.LayerNorm(normed, input, blk.LnBeforeW!, blk.LnBeforeB!, _cfg.SwinLayerNormEps);

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
                    // else padding rows/cols stay 0
                }
        }
        normed.Dispose();

        // q,k,v projections on the shifted, padded token grid [1, hp*wp, C]
        int nTok = hp * wp;
        Tensor xs = new(new TensorShape(1, nTok, dim), DType.F32);
        fixed (float* sp = shifted) Buffer.MemoryCopy(sp, (void*)xs.DataPointer, (long)nTok * dim * 4, (long)nTok * dim * 4);
        Tensor q = Linear(backend, xs, blk.QW!, blk.QB!, nTok, dim, dim);
        Tensor k = Linear(backend, xs, blk.KW!, blk.KB!, nTok, dim, dim);
        Tensor v = Linear(backend, xs, blk.VW!, blk.VB!, nTok, dim, dim);
        xs.Dispose();

        // window attention on host, producing context [hp*wp, C]
        float[] context = WindowAttention(q, k, v, hp, wp, dim, heads, ws, shift, blk.RelBiasTable!);
        q.Dispose(); k.Dispose(); v.Dispose();

        // reverse shift + unpad -> [H,W,C] -> [1,HW,C]
        Tensor attnOut = new(new TensorShape(1, (long)h * wdt, dim), DType.F32);
        {
            float* ap = (float*)attnOut.DataPointer;
            for (int y = 0; y < h; y++)
                for (int xx = 0; xx < wdt; xx++)
                {
                    int sy = y, sx = xx;
                    if (shift > 0) { sy = ((y - shift) % hp + hp) % hp; sx = ((xx - shift) % wp + wp) % wp; }
                    long src = ((long)sy * wp + sx) * dim;
                    long dst = ((long)y * wdt + xx) * dim;
                    for (int c = 0; c < dim; c++) ap[dst + c] = context[src + c];
                }
        }

        // o_proj + residual
        Tensor o = Linear(backend, attnOut, blk.OW!, blk.OB!, h * wdt, dim, dim);
        attnOut.Dispose();
        AddInPlace(o, input);

        // MLP branch (post-norm residual)
        Tensor lnA = new(o.Shape, DType.F32);
        backend.LayerNorm(lnA, o, blk.LnAfterW!, blk.LnAfterB!, _cfg.SwinLayerNormEps);
        int inner = (int)(_cfg.SwinMlpRatio * dim);
        Tensor fc1 = Linear(backend, lnA, blk.Fc1W!, blk.Fc1B!, h * wdt, dim, inner);
        lnA.Dispose();
        GdMath.ErfGelu(fc1);
        Tensor fc2 = Linear(backend, fc1, blk.Fc2W!, blk.Fc2B!, h * wdt, inner, dim);
        fc1.Dispose();
        AddInPlace(fc2, o);
        o.Dispose();
        return fc2;
    }

    private float[] WindowAttention(Tensor q, Tensor k, Tensor v, int hp, int wp, int dim, int heads, int ws, int shift, Tensor relTable)
    {
        int hd = dim / heads;
        int wa = ws * ws;
        int nWh = hp / ws, nWw = wp / ws, nW = nWh * nWw;
        float scale = 1f / MathF.Sqrt(hd);
        int[] relIndex = GdMath.SwinRelativePositionIndex(ws);
        float* rt = (float*)relTable.DataPointer;   // [ (2ws-1)^2, heads ]
        // shift attn mask region ids per padded position
        int[] regionId = shift > 0 ? ShiftRegionIds(hp, wp, ws, shift) : Array.Empty<int>();

        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        float[] context = new float[(long)hp * wp * dim];

        float[] scores = new float[wa];
        for (int wy = 0; wy < nWh; wy++)
            for (int wx = 0; wx < nWw; wx++)
            {
                // token global index for (local i) in this window
                int baseY = wy * ws, baseX = wx * ws;
                for (int h = 0; h < heads; h++)
                {
                    for (int i = 0; i < wa; i++)
                    {
                        int iy = baseY + i / ws, ix = baseX + i % ws;
                        long qOff = ((long)iy * wp + ix) * dim + (long)h * hd;
                        float maxv = float.NegativeInfinity;
                        for (int j = 0; j < wa; j++)
                        {
                            int jy = baseY + j / ws, jx = baseX + j % ws;
                            long kOff = ((long)jy * wp + jx) * dim + (long)h * hd;
                            float dot = 0f;
                            for (int c = 0; c < hd; c++) dot += qp[qOff + c] * kp[kOff + c];
                            dot *= scale;
                            dot += rt[(long)relIndex[i * wa + j] * heads + h];
                            if (shift > 0 && regionId[(long)iy * wp + ix] != regionId[(long)jy * wp + jx]) dot += -100f;
                            scores[j] = dot;
                            if (dot > maxv) maxv = dot;
                        }
                        float sum = 0f;
                        for (int j = 0; j < wa; j++) { float e = MathF.Exp(scores[j] - maxv); scores[j] = e; sum += e; }
                        float inv = 1f / sum;
                        long cOff = ((long)iy * wp + ix) * dim + (long)h * hd;
                        for (int c = 0; c < hd; c++)
                        {
                            float acc = 0f;
                            for (int j = 0; j < wa; j++)
                            {
                                int jy = baseY + j / ws, jx = baseX + j % ws;
                                acc += scores[j] * vp[((long)jy * wp + jx) * dim + (long)h * hd + c];
                            }
                            context[cOff + c] = acc * inv;
                        }
                    }
                }
            }
        return context;
    }

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

    private Tensor PatchMerging(IBackend backend, Tensor x, int h, int w, int dim, SwinStageWeights sw)
    {
        // maybe_pad to even
        int padB = h % 2, padR = w % 2;
        int hp = h + padB, wp = w + padR;
        int nh = hp / 2, nw = wp / 2;
        int fourC = 4 * dim;
        float* xp = (float*)x.DataPointer;
        // Concat order: for col in 0..1, for row in 0..1 -> [row::2, col::2]. HF: [feat[row::2,col::2] for col for row]
        // groups appended: (row0,col0),(row1,col0),(row0,col1),(row1,col1)
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
        backend.LayerNorm(normed, cat, sw.DownNormW!, sw.DownNormB!, _cfg.SwinLayerNormEps);
        cat.Dispose();
        Tensor reduced = new(new TensorShape(1, (long)nh * nw, 2 * dim), DType.F32);
        backend.Linear(reduced, normed, sw.DownReductionW!, null);   // reduction has no bias
        normed.Dispose();
        return reduced;
    }

    private Tensor ExportFeatureMap(IBackend backend, Tensor x, int h, int w, int dim, string stageName)
    {
        Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, _outNorms[$"{stageName}.weight"], _outNorms[$"{stageName}.bias"], 1e-5f);
        // [1, HW, C] -> [1, C, H, W]
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

    private static Tensor PadImage(Tensor img, int ps, out int pH, out int pW)
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

    private static Tensor ChannelsToTokens(IBackend backend, Tensor chw, int c, int hw)
    {
        Tensor outp = new(new TensorShape(1, hw, c), DType.F32);
        float* ip = (float*)chw.DataPointer;
        float* op = (float*)outp.DataPointer;
        for (int cc = 0; cc < c; cc++)
            for (int t = 0; t < hw; t++)
                op[(long)t * c + cc] = ip[(long)cc * hw + t];
        return outp;
    }

    private static Tensor Linear(IBackend backend, Tensor input, Tensor weight, Tensor? bias, int rows, int inDim, int outDim)
    {
        Tensor output = new(new TensorShape(1, rows, outDim), DType.F32);
        backend.Linear(output, input, weight, bias);
        return output;
    }

    private static void AddInPlace(Tensor dst, Tensor src)
    {
        float* d = (float*)dst.DataPointer, s = (float*)src.DataPointer;
        for (long i = 0; i < dst.ElementCount; i++) d[i] += s[i];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private sealed class SwinStageWeights
    {
        public SwinBlockWeights[] Blocks = Array.Empty<SwinBlockWeights>();
        public Tensor? DownReductionW, DownNormW, DownNormB;
    }

    private sealed class SwinBlockWeights
    {
        public Tensor? LnBeforeW, LnBeforeB, QW, QB, KW, KB, VW, VB, RelBiasTable, OW, OB, LnAfterW, LnAfterB, Fc1W, Fc1B, Fc2W, Fc2B;
    }
}
