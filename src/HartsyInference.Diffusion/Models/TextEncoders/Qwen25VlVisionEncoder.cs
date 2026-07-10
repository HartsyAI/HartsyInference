using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Qwen2.5-VL vision tower loaded from HF safetensors (<c>visual.*</c> keys). Same architecture and math as
/// the GGUF-backed <c>HartsyInference.LLM.Multimodal.Qwen25VlEncoder</c>: temporal-summed Conv3D patch embed, 2D
/// rotary position embeddings (rotate-half, no learned position table), window attention (full attention only on
/// <see cref="Qwen25VlVisionConfig.FullAttentionLayers"/>), RMSNorm blocks, SwiGLU MLP, and a 2×2 patch-merger.
/// Patches are processed in merge-block order and reordered into window-contiguous groups for the windowed layers.
///
/// <para>The 2D-RoPE application, window partitioning, and merge-block patchify run host-side (one-time, off any
/// hot path); the heavy matmuls/attention run through <see cref="IBackend"/>. All weights are materialized to F32
/// host copies at load (fp8_scaled companions dequantized) — the tower runs once per image, so F32 is fine.</para></summary>
public sealed unsafe class Qwen25VlVisionEncoder : IDisposable
{
    private readonly Qwen25VlVisionConfig _config;
    private readonly Block[] _blocks;
    private readonly HashSet<int> _fullAttention;

    private Tensor? _patchEmbedWeight;
    private Tensor? _mergerNormWeight;
    private Tensor? _mergerFc1Weight, _mergerFc1Bias, _mergerFc2Weight, _mergerFc2Bias;
    private int _disposed;

    public Qwen25VlVisionEncoder(Qwen25VlVisionConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _blocks = new Block[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new Block(config);
        _fullAttention = new HashSet<int>(config.FullAttentionLayers);
    }

    /// <summary>Loads vision-tower weights from HF <c>visual.*</c> keys, materializing everything to owned F32 host
    /// copies. The Conv3D patch embed's two temporal slices are summed (single images duplicate the temporal frame,
    /// so summing the conv weights is numerically identical); fused <c>attn.qkv</c> weights are split into Q/K/V;
    /// fp8_scaled tensors are dequantized via their <c>.scale_weight</c> companions.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _patchEmbedWeight = SumTemporalPatchEmbed(weights, "visual.patch_embed.proj.weight");
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, $"visual.blocks.{i}");
        _mergerNormWeight = MaterializeF32(weights, "visual.merger.ln_q.weight");
        _mergerFc1Weight = MaterializeF32(weights, "visual.merger.mlp.0.weight");
        _mergerFc1Bias = MaterializeF32(weights, "visual.merger.mlp.0.bias");
        _mergerFc2Weight = MaterializeF32(weights, "visual.merger.mlp.2.weight");
        _mergerFc2Bias = MaterializeF32(weights, "visual.merger.mlp.2.bias");
    }

    /// <summary>Enumerates all weight tensors for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchEmbedWeight, _mergerNormWeight, _mergerFc1Weight, _mergerFc1Bias, _mergerFc2Weight, _mergerFc2Bias })
            if (t is not null) yield return t;
        foreach (Block b in _blocks)
            foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the vision tower on a preprocessed image <c>[3, gridH·14, gridW·14]</c> (F32, already resized +
    /// CLIP-normalized, see <see cref="Qwen25VlImageProcessor"/>).</summary>
    /// <returns>Merged vision tokens <c>[1, (gridH/2)·(gridW/2), OutHiddenSize]</c> in spatial (raster) merge-unit order.</returns>
    public Tensor Forward(IBackend backend, Tensor pixelValues, int gridH, int gridW)
    {
        ThrowIfDisposed();
        if (_patchEmbedWeight is null)
            throw new InvalidOperationException("LoadWeights must be called before Forward.");
        int hidden = _config.HiddenSize, patch = _config.PatchSize, m = _config.SpatialMergeSize, hd = _config.HeadDim;
        if (pixelValues.Shape.Rank != 3 || pixelValues.Shape[0] != 3
            || pixelValues.Shape[1] != (long)gridH * patch || pixelValues.Shape[2] != (long)gridW * patch)
            throw new ArgumentException(
                $"Expected pixelValues [3, {gridH * patch}, {gridW * patch}] for grid ({gridH}, {gridW}), got {pixelValues.Shape}.",
                nameof(pixelValues));
        if (gridH % m != 0 || gridW % m != 0)
            throw new ArgumentException($"Grid ({gridH}, {gridW}) must be divisible by spatial merge size {m}.");
        int H = gridH * patch, W = gridW * patch;
        int np = gridH * gridW, pin = 3 * patch * patch;

        // 1. Patchify into merge-block order: [np, 3*patch*patch], each patch flattened [c,h,w].
        Tensor patches = new Tensor(new TensorShape(np, pin), DType.F32);
        float* px = (float*)pixelValues.DataPointer;   // [3,H,W] (D2H sync)
        backend.Sync();
        float* pp = (float*)patches.DataPointer;
        int idx = 0;
        for (int bh = 0; bh < gridH / m; bh++)
            for (int bw = 0; bw < gridW / m; bw++)
                for (int mh = 0; mh < m; mh++)
                    for (int mw = 0; mw < m; mw++)
                    {
                        int ph = (bh * m + mh) * patch, pw = (bw * m + mw) * patch;
                        float* dst = pp + (long)idx * pin;
                        int o = 0;
                        for (int c = 0; c < 3; c++)
                            for (int yy = 0; yy < patch; yy++)
                                for (int xx = 0; xx < patch; xx++)
                                    dst[o++] = px[((long)c * H + (ph + yy)) * W + (pw + xx)];
                        idx++;
                    }

        // 2. Patch embed: [np, hidden] (temporal-summed conv kernel, no bias).
        Tensor embed = new Tensor(new TensorShape(np, hidden), DType.F32);
        backend.Linear(embed, patches, _patchEmbedWeight!, null);
        patches.Dispose();

        // 3. 2D RoPE cos/sin [np, headDim] + window index, then reorder hidden + cos/sin into window order.
        (float[] cos, float[] sin) = BuildRope(gridH, gridW);
        (int[] winIdx, int[] cuWin) = WindowIndex(gridH, gridW);
        int[] mergeOrder = ExpandToPatches(winIdx, m);   // patch-level permutation (np)
        Tensor h = ReorderRows(embed, mergeOrder, hidden);
        embed.Dispose();
        float[] cosW = ReorderVec(cos, mergeOrder, hd), sinW = ReorderVec(sin, mergeOrder, hd);

        // Build the two attention masks (full / windowed block-diagonal) once.
        using Tensor fullMask = BuildMask(np, cuWin, full: true);
        using Tensor winMask = BuildMask(np, cuWin, full: false);

        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, h, cosW, sinW, np, _fullAttention.Contains(i) ? fullMask : winMask);
            h.Dispose();
            h = next;
        }

        // post-blocks RMSNorm (merger.ln_q).
        Tensor post = new Tensor(new TensorShape(1, np, hidden), DType.F32);
        backend.RmsNorm(post, h.Reshape(new TensorShape(1, np, hidden)), _mergerNormWeight!, _config.RmsNormEps);
        h.Dispose();

        // 4. Merger: group 4 consecutive (merge-unit) patches → [np/4, hidden*4] → mlp.0 → GELU → mlp.2.
        int nTok = np / (m * m), mergeDim = hidden * m * m;
        Tensor grouped = post.Reshape(new TensorShape(1, nTok, mergeDim));
        Tensor mid = new Tensor(new TensorShape(1, nTok, mergeDim), DType.F32);
        backend.Linear(mid, grouped, _mergerFc1Weight!, _mergerFc1Bias);
        post.Dispose();
        Tensor act = new Tensor(new TensorShape(1, nTok, mergeDim), DType.F32);
        backend.Gelu(act, mid);
        mid.Dispose();
        Tensor merged = new Tensor(new TensorShape(1, nTok, _config.OutHiddenSize), DType.F32);
        backend.Linear(merged, act, _mergerFc2Weight!, _mergerFc2Bias);
        act.Dispose();

        // 5. Un-reorder tokens (merge-units) back to spatial order.
        int[] tokIdx = new int[nTok];
        for (int i = 0; i < nTok; i++) tokIdx[winIdx[i]] = i;   // reverse permutation
        Tensor img = ReorderRows(merged, tokIdx, _config.OutHiddenSize);
        merged.Dispose();
        return img;
    }

    /// <summary>2D rotary cos/sin tables [np, headDim] in raster (pre-window) order.</summary>
    private (float[] cos, float[] sin) BuildRope(int gh, int gw)
    {
        int hd = _config.HeadDim, m = _config.SpatialMergeSize, np = gh * gw, ropeDim = hd / 2;   // ropeDim=40
        int freqN = ropeDim / 2;                                                                  // 20 inv-freqs
        float[] inv = new float[freqN];
        for (int i = 0; i < freqN; i++) inv[i] = 1f / MathF.Pow(10000f, (2f * i) / ropeDim);
        // merge-permuted (h,w) position ids, matching the patchify order.
        int[] hpos = new int[np], wpos = new int[np];
        int idx = 0;
        for (int bh = 0; bh < gh / m; bh++)
            for (int bw = 0; bw < gw / m; bw++)
                for (int mh = 0; mh < m; mh++)
                    for (int mw = 0; mw < m; mw++)
                    {
                        hpos[idx] = bh * m + mh;
                        wpos[idx] = bw * m + mw;
                        idx++;
                    }
        float[] cos = new float[(long)np * hd];
        float[] sin = new float[(long)np * hd];
        for (int pos = 0; pos < np; pos++)
        {
            // rot = [h*inv (20) ++ w*inv (20)] = 40; emb = [rot, rot] = 80.
            for (int j = 0; j < freqN; j++)
            {
                float fh = hpos[pos] * inv[j], fw = wpos[pos] * inv[j];
                // rot index j → h, freqN+j → w; then duplicated at +ropeDim.
                SetCosSin(cos, sin, pos, hd, j, fh);
                SetCosSin(cos, sin, pos, hd, freqN + j, fw);
                SetCosSin(cos, sin, pos, hd, ropeDim + j, fh);
                SetCosSin(cos, sin, pos, hd, ropeDim + freqN + j, fw);
            }
        }
        return (cos, sin);
    }

    private static void SetCosSin(float[] cos, float[] sin, int pos, int hd, int e, float ang)
    {
        cos[pos * hd + e] = MathF.Cos(ang);
        sin[pos * hd + e] = MathF.Sin(ang);
    }

    /// <summary>Window index in merge-units (matches HF Qwen2.5-VL get_window_index): returns the merge-unit
    /// permutation and the cumulative patch-level window boundaries (cu_seqlens) for the block-diagonal mask.</summary>
    private (int[] winIdx, int[] cuWin) WindowIndex(int gh, int gw)
    {
        int m = _config.SpatialMergeSize, vw = _config.WindowPatches;   // merged units per window side (4)
        int lh = gh / m, lw = gw / m;
        int padH = (vw - lh % vw) % vw, padW = (vw - lw % vw) % vw;
        int nwh = (lh + padH) / vw, nww = (lw + padW) / vw;
        List<int> winIdx = new List<int>();
        List<int> cu = new List<int> { 0 };
        for (int wh = 0; wh < nwh; wh++)
            for (int ww = 0; ww < nww; ww++)
            {
                int count = 0;
                for (int a = 0; a < vw; a++)
                    for (int b = 0; b < vw; b++)
                    {
                        int r = wh * vw + a, c = ww * vw + b;
                        if (r < lh && c < lw)
                        {
                            winIdx.Add(r * lw + c);
                            count++;
                        }
                    }
                cu.Add(cu[^1] + count * m * m);   // patch count
            }
        return (winIdx.ToArray(), cu.ToArray());
    }

    /// <summary>Expands a merge-unit permutation to a patch-level permutation (each unit = m*m consecutive patches).</summary>
    private static int[] ExpandToPatches(int[] winIdx, int m)
    {
        int u = m * m;
        int[] o = new int[winIdx.Length * u];
        for (int i = 0; i < winIdx.Length; i++)
            for (int j = 0; j < u; j++)
                o[i * u + j] = winIdx[i] * u + j;
        return o;
    }

    private static Tensor ReorderRows(Tensor src, int[] rowIdx, int dim)
    {
        int n = rowIdx.Length;
        Tensor o = new Tensor(new TensorShape(1, n, dim), DType.F32);
        float* s = (float*)src.DataPointer;
        float* d = (float*)o.DataPointer;
        for (int i = 0; i < n; i++)
            Buffer.MemoryCopy(s + (long)rowIdx[i] * dim, d + (long)i * dim, (long)dim * 4, (long)dim * 4);
        return o;
    }

    private static float[] ReorderVec(float[] src, int[] rowIdx, int dim)
    {
        float[] o = new float[(long)rowIdx.Length * dim];
        for (int i = 0; i < rowIdx.Length; i++)
            Array.Copy(src, (long)rowIdx[i] * dim, o, (long)i * dim, dim);
        return o;
    }

    /// <summary>Builds a [1,1,np,np] additive attention mask (0 visible, -inf masked). Full attention if
    /// <paramref name="full"/>, else block-diagonal over the window boundaries <paramref name="cuWin"/>.</summary>
    private static Tensor BuildMask(int np, int[] cuWin, bool full)
    {
        Tensor mask = new Tensor(new TensorShape(1, 1, np, np), DType.F32);
        float* mp = (float*)mask.DataPointer;
        const float NEG = -1e30f;
        if (full)
        {
            for (long i = 0; i < (long)np * np; i++) mp[i] = 0f;
            return mask;
        }
        for (long i = 0; i < (long)np * np; i++) mp[i] = NEG;
        for (int w = 0; w < cuWin.Length - 1; w++)
            for (int r = cuWin[w]; r < cuWin[w + 1]; r++)
                for (int c = cuWin[w]; c < cuWin[w + 1]; c++)
                    mp[(long)r * np + c] = 0f;
        return mask;
    }

    /// <summary>Applies 2D RoPE in place to <paramref name="t"/> = <c>[1, np, heads, headDim]</c>, host-side.
    /// <c>out = x·cos + rotate_half(x)·sin</c> where rotate_half splits headDim in two halves.</summary>
    private static void ApplyRope(IBackend backend, Tensor t, float[] cosW, float[] sinW, int np, int heads, int hd)
    {
        backend.Sync();
        float* p = (float*)t.DataPointer;
        int half = hd / 2;
        float[] tmp = new float[hd];
        for (int pos = 0; pos < np; pos++)
            for (int h = 0; h < heads; h++)
            {
                float* row = p + ((long)pos * heads + h) * hd;
                for (int e = 0; e < hd; e++) tmp[e] = row[e];
                for (int e = 0; e < hd; e++)
                {
                    float rot = e < half ? -tmp[e + half] : tmp[e - half];
                    row[e] = tmp[e] * cosW[pos * hd + e] + rot * sinW[pos * hd + e];
                }
            }
    }

    /// <summary>Materializes a checkpoint tensor as an owned F32 host copy. fp8 tensors dequantize as
    /// <c>float(w) · scale</c> — <see cref="Tensor.CastTo"/> folds an already-set <see cref="Tensor.Fp8ScaleFactor"/>;
    /// an unfolded <c>.scale_weight</c> / <c>.weight_scale</c> companion key is applied here instead.</summary>
    private static Tensor MaterializeF32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!weights.TryGetValue(key, out Tensor? src))
            throw new KeyNotFoundException($"Missing Qwen2.5-VL vision weight '{key}'.");
        Tensor f32 = src.CastTo(DType.F32);
        if (src.DType.IsFp8 && src.Fp8ScaleFactor == 1f)
        {
            float scale = FindCompanionScale(weights, key);
            if (scale != 1f)
            {
                float* p = (float*)f32.DataPointer;
                long n = f32.ElementCount;
                for (long i = 0; i < n; i++) p[i] *= scale;
            }
        }
        return f32;
    }

    private static float FindCompanionScale(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!key.EndsWith(".weight", StringComparison.Ordinal)) return 1f;
        string baseKey = key[..^".weight".Length];
        if (!weights.TryGetValue($"{baseKey}.scale_weight", out Tensor? scaleTensor)
            && !weights.TryGetValue($"{baseKey}.weight_scale", out scaleTensor))
            return 1f;
        if (scaleTensor.DType == DType.F32) return *(float*)scaleTensor.DataPointer;
        using Tensor f32 = scaleTensor.CastTo(DType.F32);
        return *(float*)f32.DataPointer;
    }

    /// <summary>Sums the two temporal slices of the Conv3D patch embed [hidden, 3, 2, patch, patch] into
    /// [hidden, 3·patch²]: single images duplicate the temporal frame, so w0+w1 is numerically identical.</summary>
    private Tensor SumTemporalPatchEmbed(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        int hidden = _config.HiddenSize, temporal = _config.TemporalPatchSize;
        int pp = _config.PatchSize * _config.PatchSize;
        int pin = _config.PatchEmbedInDim;
        using Tensor conv = MaterializeF32(weights, key);
        if (conv.ElementCount != (long)hidden * 3 * temporal * pp)
            throw new HartsyInferenceException(
                $"Unexpected patch embed shape {conv.Shape}; expected [{hidden}, 3, {temporal}, {_config.PatchSize}, {_config.PatchSize}].");
        Tensor wSum = new Tensor(new TensorShape(hidden, pin), DType.F32);
        float* src = (float*)conv.DataPointer;
        float* dst = (float*)wSum.DataPointer;
        for (int o = 0; o < hidden; o++)
            for (int c = 0; c < 3; c++)
            {
                long srcBase = (((long)o * 3 + c) * temporal) * pp;
                long dstBase = ((long)o * 3 + c) * pp;
                for (int s = 0; s < pp; s++)
                {
                    float sum = 0f;
                    for (int t = 0; t < temporal; t++) sum += src[srcBase + (long)t * pp + s];
                    dst[dstBase + s] = sum;
                }
            }
        return wSum;
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor? t in new[] { _patchEmbedWeight, _mergerNormWeight, _mergerFc1Weight, _mergerFc1Bias, _mergerFc2Weight, _mergerFc2Bias })
            t?.Dispose();
        _patchEmbedWeight = _mergerNormWeight = _mergerFc1Weight = _mergerFc1Bias = _mergerFc2Weight = _mergerFc2Bias = null;
        foreach (Block b in _blocks) b.Dispose();
        GC.SuppressFinalize(this);
    }

    // ── vision transformer block ──
    private sealed unsafe class Block : IDisposable
    {
        private readonly Qwen25VlVisionConfig _c;
        private Tensor? _norm1W, _norm2W;
        private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _projW, _projB;
        private Tensor? _gateW, _gateB, _upW, _upB, _downW, _downB;

        public Block(Qwen25VlVisionConfig c) => _c = c;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _norm1W = MaterializeF32(w, $"{p}.norm1.weight");
            _norm2W = MaterializeF32(w, $"{p}.norm2.weight");
            // HF fused qkv row order: rows [0,H) = Q, [H,2H) = K, [2H,3H) = V.
            int hidden = _c.HiddenSize;
            using (Tensor qkvW = MaterializeF32(w, $"{p}.attn.qkv.weight"))
            {
                _qW = SliceRows(qkvW, 0, hidden, hidden);
                _kW = SliceRows(qkvW, hidden, hidden, hidden);
                _vW = SliceRows(qkvW, 2 * hidden, hidden, hidden);
            }
            using (Tensor qkvB = MaterializeF32(w, $"{p}.attn.qkv.bias"))
            {
                _qB = SliceRows(qkvB, 0, hidden, 1);
                _kB = SliceRows(qkvB, hidden, hidden, 1);
                _vB = SliceRows(qkvB, 2 * hidden, hidden, 1);
            }
            _projW = MaterializeF32(w, $"{p}.attn.proj.weight");
            _projB = MaterializeF32(w, $"{p}.attn.proj.bias");
            _gateW = MaterializeF32(w, $"{p}.mlp.gate_proj.weight");
            _gateB = MaterializeF32(w, $"{p}.mlp.gate_proj.bias");
            _upW = MaterializeF32(w, $"{p}.mlp.up_proj.weight");
            _upB = MaterializeF32(w, $"{p}.mlp.up_proj.bias");
            _downW = MaterializeF32(w, $"{p}.mlp.down_proj.weight");
            _downB = MaterializeF32(w, $"{p}.mlp.down_proj.bias");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor? t in AllWeights())
                if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x, float[] cosW, float[] sinW, int np, Tensor mask)
        {
            int hidden = _c.HiddenSize, heads = _c.NumHeads, hd = _c.HeadDim;
            TensorShape flat3 = new TensorShape(1, np, hidden);

            Tensor n1 = new Tensor(flat3, DType.F32);
            backend.RmsNorm(n1, x.Reshape(flat3), _norm1W!, _c.RmsNormEps);

            Tensor q = new Tensor(new TensorShape(1, np, heads, hd), DType.F32);
            Tensor k = new Tensor(new TensorShape(1, np, heads, hd), DType.F32);
            Tensor v = new Tensor(new TensorShape(1, np, heads, hd), DType.F32);
            backend.Linear(q, n1, _qW!, _qB);
            backend.Linear(k, n1, _kW!, _kB);
            backend.Linear(v, n1, _vW!, _vB);
            n1.Dispose();
            ApplyRope(backend, q, cosW, sinW, np, heads, hd);
            ApplyRope(backend, k, cosW, sinW, np, heads, hd);

            Tensor qM = new Tensor(new TensorShape(1, heads, np, hd), DType.F32);
            Tensor kM = new Tensor(new TensorShape(1, heads, np, hd), DType.F32);
            Tensor vM = new Tensor(new TensorShape(1, heads, np, hd), DType.F32);
            backend.Permute0213(qM, q, np, heads, hd);
            backend.Permute0213(kM, k, np, heads, hd);
            backend.Permute0213(vM, v, np, heads, hd);
            q.Dispose(); k.Dispose(); v.Dispose();

            Tensor attn = new Tensor(new TensorShape(1, heads, np, hd), DType.F32);
            backend.ScaledDotProductAttention(attn, qM, kM, vM, mask, 1f / MathF.Sqrt(hd));
            qM.Dispose(); kM.Dispose(); vM.Dispose();

            Tensor attnFlat = new Tensor(flat3, DType.F32);
            backend.Permute0213(attnFlat, attn, heads, np, hd);
            attn.Dispose();
            Tensor attnOut = new Tensor(flat3, DType.F32);
            backend.Linear(attnOut, attnFlat, _projW!, _projB);
            attnFlat.Dispose();
            Tensor afterAttn = new Tensor(flat3, DType.F32);
            backend.Add(afterAttn, x.Reshape(flat3), attnOut);
            attnOut.Dispose();

            // SwiGLU MLP: down(silu(gate)·up).
            Tensor n2 = new Tensor(flat3, DType.F32);
            backend.RmsNorm(n2, afterAttn, _norm2W!, _c.RmsNormEps);
            Tensor up = new Tensor(new TensorShape(1, np, _c.IntermediateSize), DType.F32);
            backend.Linear(up, n2, _upW!, _upB);
            Tensor gate = new Tensor(new TensorShape(1, np, _c.IntermediateSize), DType.F32);
            backend.Linear(gate, n2, _gateW!, _gateB);
            n2.Dispose();
            backend.Silu(gate, gate);
            Tensor gu = new Tensor(new TensorShape(1, np, _c.IntermediateSize), DType.F32);
            backend.Mul(gu, gate, up);
            gate.Dispose(); up.Dispose();
            Tensor down = new Tensor(flat3, DType.F32);
            backend.Linear(down, gu, _downW!, _downB);
            gu.Dispose();
            Tensor result = new Tensor(flat3, DType.F32);
            backend.Add(result, afterAttn, down);
            afterAttn.Dispose(); down.Dispose();
            // Barrier once per block: the ViT runs a single time per image (off the decode hot path), and at large
            // patch counts (e.g. 560px → 1600 patches) the stream-ordered activation pool can otherwise recycle a
            // buffer still in flight across the next block's Reshape. A per-block sync is negligible here and serializes.
            backend.Sync();
            return result;
        }

        private static Tensor SliceRows(Tensor src, int startRow, int numRows, int rowElems)
        {
            TensorShape shape = rowElems == 1 ? new TensorShape(numRows) : new TensorShape(numRows, rowElems);
            Tensor outp = new Tensor(shape, DType.F32);
            float* s = (float*)src.DataPointer;
            long bytes = (long)numRows * rowElems * sizeof(float);
            Buffer.MemoryCopy(s + (long)startRow * rowElems, (void*)outp.DataPointer, bytes, bytes);
            return outp;
        }

        private Tensor?[] AllWeights() =>
            [_norm1W, _norm2W, _qW, _qB, _kW, _kB, _vW, _vB, _projW, _projB, _gateW, _gateB, _upW, _upB, _downW, _downB];

        public void Dispose()
        {
            foreach (Tensor? t in AllWeights()) t?.Dispose();
            _norm1W = _norm2W = _qW = _qB = _kW = _kB = _vW = _vB = _projW = _projB = null;
            _gateW = _gateB = _upW = _upB = _downW = _downB = null;
        }
    }
}
