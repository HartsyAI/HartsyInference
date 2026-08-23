using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Qwen3-VL vision tower (<c>Qwen3VLVisionModel</c>). Turns preprocessed image patches into merged image
/// tokens for the Qwen3-VL language stream, and produces the Qwen3-VL <b>deepstack</b> feature taps.
///
/// <para>Pipeline: patch-embed (<c>Linear</c> over the flattened conv kernel) + bilinear-interpolated learned position
/// embedding → <see cref="Qwen3VlVisionConfig.Depth"/> full-attention blocks (LayerNorm, 2D split-half RoPE, gelu MLP)
/// → at the deepstack layers, a post-shuffle merger produces a deepstack feature set → a final merger projects each
/// <c>spatial_merge²</c> block to the LM hidden size.</para>
///
/// <para>Single still image per call (the documented Boogu edit scope): attention is full over all patches (no
/// windowing — Qwen3-VL uses per-image full attention). Structural; numeric parity pending (docs/Research/BOOGU_IMAGE.md §10).</para></summary>
public sealed unsafe class Qwen3VlVisionEncoder : IDisposable
{
    private readonly Qwen3VlVisionConfig _config;
    private readonly VisionBlock[] _blocks;
    private readonly Merger _merger;
    private readonly Merger[] _deepstackMergers;

    private Tensor? _patchEmbedWeight, _patchEmbedBias;
    private Tensor? _posEmbedWeight;
    private int _disposed;

    /// <summary>Result of a vision forward.</summary>
    public sealed class VisionOutput
    {
        /// <summary>Merged image tokens <c>[numMergedTokens, OutHiddenSize]</c> for the LM image positions.</summary>
        public required Tensor MergedTokens { get; init; }
        /// <summary>One deepstack feature tensor per deepstack layer, each <c>[numMergedTokens, OutHiddenSize]</c>.</summary>
        public required Tensor[] DeepstackFeatures { get; init; }
        /// <summary>Number of merged image tokens (= numPatches / spatial_merge²).</summary>
        public required int NumMergedTokens { get; init; }
    }

    public Qwen3VlVisionEncoder(Qwen3VlVisionConfig config)
    {
        _config = config;
        _blocks = new VisionBlock[config.Depth];
        for (int i = 0; i < config.Depth; i++)
            _blocks[i] = new VisionBlock(config);
        _merger = new Merger(config, postShuffleNorm: false);
        _deepstackMergers = new Merger[config.DeepstackVisualIndexes.Length];
        for (int i = 0; i < _deepstackMergers.Length; i++)
            _deepstackMergers[i] = new Merger(config, postShuffleNorm: true);
    }

    /// <summary>Loads vision-tower weights (bare keys after the converter strips the <c>visual.</c> prefix).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        Tensor proj = w["patch_embed.proj.weight"]; // [hidden, in_ch, temporal, patch, patch]
        _patchEmbedWeight = ReshapeConvToLinear(proj, _config.HiddenSize, _config.PatchEmbedInDim);
        _patchEmbedBias = w["patch_embed.proj.bias"];
        _posEmbedWeight = TensorCasts.EnsureF32(w["pos_embed.weight"]);

        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"blocks.{i}");
        _merger.LoadWeights(w, "merger");
        for (int i = 0; i < _deepstackMergers.Length; i++)
            _deepstackMergers[i].LoadWeights(w, $"deepstack_merger_list.{i}");
    }

    /// <summary>Enumerates all weight tensors for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_patchEmbedWeight is not null) yield return _patchEmbedWeight;
        if (_patchEmbedBias is not null) yield return _patchEmbedBias;
        if (_posEmbedWeight is not null) yield return _posEmbedWeight;
        foreach (VisionBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (Tensor t in _merger.EnumerateWeights()) yield return t;
        foreach (Merger m in _deepstackMergers) foreach (Tensor t in m.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the vision tower on preprocessed patches <c>[numPatches, patchEmbedInDim]</c> with grid
    /// <c>(t, h, w)</c> in patch units (single image, t = 1).</summary>
    public VisionOutput Forward(IBackend backend, Tensor pixelValues, int gridT, int gridH, int gridW)
    {
        ThrowIfDisposed();
        int seq = (int)pixelValues.Shape[0];
        int hidden = _config.HiddenSize;
        int merge = _config.SpatialMergeSize;

        // 1. patch embed (Linear) — treat as [1, seq, hidden].
        Tensor h0 = new Tensor(new TensorShape(1, seq, hidden), DType.F32);
        backend.Linear(h0, Reshape2dTo3d(pixelValues), _patchEmbedWeight!, _patchEmbedBias);

        // 2. + bilinear-interpolated learned position embedding.
        AddPositionEmbedding(h0, gridT, gridH, gridW);

        // 3. 2D RoPE table (split-half), per token in merge-block order.
        (float[] ropeCos, float[] ropeSin) = BuildVisionRope(gridT, gridH, gridW);

        // 4. blocks + deepstack taps.
        Tensor hidden3d = h0;
        Tensor[] deepstack = new Tensor[_deepstackMergers.Length];
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, hidden3d, ropeCos, ropeSin, seq);
            hidden3d.Dispose();
            hidden3d = next;
            int tapPos = Array.IndexOf(_config.DeepstackVisualIndexes, i);
            if (tapPos >= 0)
                deepstack[tapPos] = _deepstackMergers[tapPos].Forward(backend, hidden3d, seq, merge);
        }

        Tensor merged = _merger.Forward(backend, hidden3d, seq, merge);
        hidden3d.Dispose();

        return new VisionOutput
        {
            MergedTokens = merged,
            DeepstackFeatures = deepstack,
            NumMergedTokens = seq / (merge * merge),
        };
    }

    private void AddPositionEmbedding(Tensor h, int gridT, int gridH, int gridW)
    {
        int hidden = _config.HiddenSize;
        int side = _config.NumGridPerSide;
        int merge = _config.SpatialMergeSize;
        float* dst = (float*)h.DataPointer;
        float* pos = (float*)_posEmbedWeight!.DataPointer;

        // linspace(0, side-1, n)
        static double Lin(int i, int n, int side) => n == 1 ? 0.0 : i * (double)(side - 1) / (n - 1);

        int token = 0;
        for (int t = 0; t < gridT; t++)
            for (int hb = 0; hb < gridH / merge; hb++)
                for (int wb = 0; wb < gridW / merge; wb++)
                    for (int mh = 0; mh < merge; mh++)
                        for (int mw = 0; mw < merge; mw++)
                        {
                            int r = hb * merge + mh;
                            int c = wb * merge + mw;
                            double hy = Lin(r, gridH, side);
                            double wx = Lin(c, gridW, side);
                            int hf = (int)Math.Floor(hy), wf = (int)Math.Floor(wx);
                            int hc = Math.Min(hf + 1, side - 1), wc = Math.Min(wf + 1, side - 1);
                            hf = Math.Clamp(hf, 0, side - 1); wf = Math.Clamp(wf, 0, side - 1);
                            double fh = hy - Math.Floor(hy), fw = wx - Math.Floor(wx);
                            long i00 = (long)(hf * side + wf), i01 = (long)(hf * side + wc);
                            long i10 = (long)(hc * side + wf), i11 = (long)(hc * side + wc);
                            float w00 = (float)((1 - fh) * (1 - fw)), w01 = (float)((1 - fh) * fw);
                            float w10 = (float)(fh * (1 - fw)), w11 = (float)(fh * fw);
                            long dstOff = (long)token * hidden;
                            for (int d = 0; d < hidden; d++)
                                dst[dstOff + d] +=
                                    w00 * pos[i00 * hidden + d] + w01 * pos[i01 * hidden + d] +
                                    w10 * pos[i10 * hidden + d] + w11 * pos[i11 * hidden + d];
                            token++;
                        }
    }

    private (float[] cos, float[] sin) BuildVisionRope(int gridT, int gridH, int gridW)
    {
        int headDim = _config.HeadDim;          // 72
        int ropeDim = headDim / 2;              // 36 (the dim passed to the vision rotary)
        int half = ropeDim / 2;                 // 18 freqs per axis
        int seq = gridT * gridH * gridW;
        int merge = _config.SpatialMergeSize;
        const double theta = 10000.0;

        double[] invFreq = new double[half];
        for (int j = 0; j < half; j++)
            invFreq[j] = 1.0 / Math.Pow(theta, (double)(2 * j) / ropeDim);

        // Split-half table sized [seq * (headDim/2)] = [seq * ropeDim]; cos[token*ropeDim + k].
        float[] cos = new float[(long)seq * ropeDim];
        float[] sin = new float[(long)seq * ropeDim];
        int token = 0;
        for (int t = 0; t < gridT; t++)
            for (int hb = 0; hb < gridH / merge; hb++)
                for (int wb = 0; wb < gridW / merge; wb++)
                    for (int mh = 0; mh < merge; mh++)
                        for (int mw = 0; mw < merge; mw++)
                        {
                            int row = hb * merge + mh;
                            int col = wb * merge + mw;
                            long baseOff = (long)token * ropeDim;
                            for (int j = 0; j < half; j++)
                            {
                                double ar = row * invFreq[j];
                                cos[baseOff + j] = (float)Math.Cos(ar);
                                sin[baseOff + j] = (float)Math.Sin(ar);
                                double ac = col * invFreq[j];
                                cos[baseOff + half + j] = (float)Math.Cos(ac);
                                sin[baseOff + half + j] = (float)Math.Sin(ac);
                            }
                            token++;
                        }
        return (cos, sin);
    }

    private static Tensor Reshape2dTo3d(Tensor t)
    {
        // [N, D] viewed as [1, N, D] without copy semantics — allocate a view-like 3D tensor sharing data is not
        // available, so wrap by constructing a 3D tensor over the same data via a shallow copy.
        Tensor outp = new Tensor(new TensorShape(1, t.Shape[0], t.Shape[1]), DType.F32);
        long bytes = t.Shape.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)outp.DataPointer, bytes, bytes);
        return outp;
    }

    private static Tensor ReshapeConvToLinear(Tensor conv, int outDim, int inDim)
    {
        Tensor src = TensorCasts.EnsureF32(conv);
        Tensor outp = new Tensor(new TensorShape(outDim, inDim), DType.F32);
        long bytes = (long)outDim * inDim * sizeof(float);
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)outp.DataPointer, bytes, bytes);
        return outp;
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchEmbedWeight = _patchEmbedBias = _posEmbedWeight = null;
        }
        GC.SuppressFinalize(this);
    }

    // ── vision transformer block ──
    private sealed unsafe class VisionBlock
    {
        private readonly Qwen3VlVisionConfig _c;
        private Tensor? _norm1W, _norm1B, _norm2W, _norm2B;
        private Tensor? _qkvW, _qkvB, _projW, _projB;
        private Tensor? _fc1W, _fc1B, _fc2W, _fc2B;

        public VisionBlock(Qwen3VlVisionConfig c) => _c = c;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _norm1W = TensorCasts.EnsureF32(w[$"{p}.norm1.weight"]); _norm1B = TensorCasts.EnsureF32(w[$"{p}.norm1.bias"]);
            _norm2W = TensorCasts.EnsureF32(w[$"{p}.norm2.weight"]); _norm2B = TensorCasts.EnsureF32(w[$"{p}.norm2.bias"]);
            _qkvW = w[$"{p}.attn.qkv.weight"]; _qkvB = w[$"{p}.attn.qkv.bias"];
            _projW = w[$"{p}.attn.proj.weight"]; _projB = w[$"{p}.attn.proj.bias"];
            _fc1W = w[$"{p}.mlp.linear_fc1.weight"]; _fc1B = w[$"{p}.mlp.linear_fc1.bias"];
            _fc2W = w[$"{p}.mlp.linear_fc2.weight"]; _fc2B = w[$"{p}.mlp.linear_fc2.bias"];
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor? t in new[] { _norm1W, _norm1B, _norm2W, _norm2B, _qkvW, _qkvB, _projW, _projB, _fc1W, _fc1B, _fc2W, _fc2B })
                if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor hidden, float[] ropeCos, float[] ropeSin, int seq)
        {
            int H = _c.HiddenSize, heads = _c.NumHeads, hd = _c.HeadDim;
            TensorShape hShape = new TensorShape(1, seq, H);

            // attn = proj(SDPA(rope(qkv(norm1(x)))))
            Tensor n1 = new Tensor(hShape, DType.F32);
            backend.LayerNorm(n1, hidden, _norm1W!, _norm1B!, _c.NormEps);
            Tensor qkv = new Tensor(new TensorShape(1, seq, 3 * H), DType.F32);
            backend.Linear(qkv, n1, _qkvW!, _qkvB);
            n1.Dispose();

            Tensor q = new Tensor(hShape, DType.F32), k = new Tensor(hShape, DType.F32), v = new Tensor(hShape, DType.F32);
            SplitQkv(qkv, q, k, v, seq, H);
            qkv.Dispose();

            Tensor qMh = DiTUtils.ReshapeToMultiHead(q, 1, seq, heads, hd);
            Tensor kMh = DiTUtils.ReshapeToMultiHead(k, 1, seq, heads, hd);
            Tensor vMh = DiTUtils.ReshapeToMultiHead(v, 1, seq, heads, hd);
            q.Dispose(); k.Dispose(); v.Dispose();
            ApplyRopeSplitHalf(qMh, ropeCos, ropeSin, heads, seq, hd);
            ApplyRopeSplitHalf(kMh, ropeCos, ropeSin, heads, seq, hd);

            float scale = 1.0f / MathF.Sqrt(hd);
            Tensor attn = new Tensor(new TensorShape(1, heads, seq, hd), DType.F32);
            backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, null, scale);
            qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

            Tensor attnFlat = DiTUtils.ReshapeFromMultiHead(attn, 1, seq, heads, hd);
            attn.Dispose();
            Tensor proj = new Tensor(hShape, DType.F32);
            backend.Linear(proj, attnFlat, _projW!, _projB);
            attnFlat.Dispose();

            Tensor afterAttn = new Tensor(hShape, DType.F32);
            backend.Add(afterAttn, hidden, proj);
            proj.Dispose();

            // mlp = fc2(gelu(fc1(norm2(x))))
            Tensor n2 = new Tensor(hShape, DType.F32);
            backend.LayerNorm(n2, afterAttn, _norm2W!, _norm2B!, _c.NormEps);
            Tensor fc1 = new Tensor(new TensorShape(1, seq, _c.IntermediateSize), DType.F32);
            backend.Linear(fc1, n2, _fc1W!, _fc1B);
            n2.Dispose();
            Tensor act = new Tensor(new TensorShape(1, seq, _c.IntermediateSize), DType.F32);
            backend.Gelu(act, fc1);
            fc1.Dispose();
            Tensor fc2 = new Tensor(hShape, DType.F32);
            backend.Linear(fc2, act, _fc2W!, _fc2B);
            act.Dispose();

            Tensor outp = new Tensor(hShape, DType.F32);
            backend.Add(outp, afterAttn, fc2);
            afterAttn.Dispose(); fc2.Dispose();
            return outp;
        }

        private static void SplitQkv(Tensor qkv, Tensor q, Tensor k, Tensor v, int seq, int H)
        {
            float* src = (float*)qkv.DataPointer;
            float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
            for (int s = 0; s < seq; s++)
            {
                long b = (long)s * 3 * H;
                Buffer.MemoryCopy(src + b, qp + (long)s * H, H * sizeof(float), H * sizeof(float));
                Buffer.MemoryCopy(src + b + H, kp + (long)s * H, H * sizeof(float), H * sizeof(float));
                Buffer.MemoryCopy(src + b + 2 * H, vp + (long)s * H, H * sizeof(float), H * sizeof(float));
            }
        }

        private static void ApplyRopeSplitHalf(Tensor x, float[] cos, float[] sin, int heads, int seq, int headDim)
        {
            int half = headDim / 2;
            float* p = (float*)x.DataPointer;
            for (int h = 0; h < heads; h++)
                for (int s = 0; s < seq; s++)
                {
                    long off = ((long)h * seq + s) * headDim;
                    int posOff = s * half;
                    for (int i = 0; i < half; i++)
                    {
                        float c = cos[posOff + i], si = sin[posOff + i];
                        float a = p[off + i], b2 = p[off + i + half];
                        p[off + i] = a * c - b2 * si;
                        p[off + i + half] = b2 * c + a * si;
                    }
                }
        }
    }

    // ── patch merger (final + deepstack post-shuffle variants) ──
    private sealed unsafe class Merger
    {
        private readonly Qwen3VlVisionConfig _c;
        private readonly bool _postShuffle;
        private readonly int _mergedDim;
        private Tensor? _normW, _normB, _fc1W, _fc1B, _fc2W, _fc2B;

        public Merger(Qwen3VlVisionConfig c, bool postShuffleNorm)
        {
            _c = c;
            _postShuffle = postShuffleNorm;
            _mergedDim = c.HiddenSize * c.SpatialMergeSize * c.SpatialMergeSize;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _normW = TensorCasts.EnsureF32(w[$"{p}.norm.weight"]); _normB = TensorCasts.EnsureF32(w[$"{p}.norm.bias"]);
            _fc1W = w[$"{p}.linear_fc1.weight"]; _fc1B = w[$"{p}.linear_fc1.bias"];
            _fc2W = w[$"{p}.linear_fc2.weight"]; _fc2B = w[$"{p}.linear_fc2.bias"];
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor? t in new[] { _normW, _normB, _fc1W, _fc1B, _fc2W, _fc2B })
                if (t is not null) yield return t;
        }

        /// <summary>Merges <c>spatial_merge²</c> consecutive patch tokens → one <c>OutHiddenSize</c> token.</summary>
        public Tensor Forward(IBackend backend, Tensor hidden3d, int seq, int merge)
        {
            int H = _c.HiddenSize;
            int numMerged = seq / (merge * merge);

            Tensor grouped;
            if (_postShuffle)
            {
                // group first, then LayerNorm over the merged dim.
                Tensor g = GroupBlocks(hidden3d, seq, H, numMerged);
                Tensor normed = new Tensor(new TensorShape(1, numMerged, _mergedDim), DType.F32);
                backend.LayerNorm(normed, g, _normW!, _normB!, 1e-6f);
                g.Dispose();
                grouped = normed;
            }
            else
            {
                // LayerNorm over hidden per patch, then group.
                Tensor normed = new Tensor(new TensorShape(1, seq, H), DType.F32);
                backend.LayerNorm(normed, hidden3d, _normW!, _normB!, 1e-6f);
                grouped = GroupBlocks(normed, seq, H, numMerged);
                normed.Dispose();
            }

            Tensor fc1 = new Tensor(new TensorShape(1, numMerged, _mergedDim), DType.F32);
            backend.Linear(fc1, grouped, _fc1W!, _fc1B);
            grouped.Dispose();
            Tensor act = new Tensor(new TensorShape(1, numMerged, _mergedDim), DType.F32);
            backend.Gelu(act, fc1);
            fc1.Dispose();
            Tensor outp = new Tensor(new TensorShape(numMerged, _c.OutHiddenSize), DType.F32);
            Tensor outp3d = new Tensor(new TensorShape(1, numMerged, _c.OutHiddenSize), DType.F32);
            backend.Linear(outp3d, act, _fc2W!, _fc2B);
            act.Dispose();
            long bytes = (long)numMerged * _c.OutHiddenSize * sizeof(float);
            Buffer.MemoryCopy((void*)outp3d.DataPointer, (void*)outp.DataPointer, bytes, bytes);
            outp3d.Dispose();
            return outp;
        }

        private static Tensor GroupBlocks(Tensor hidden3d, int seq, int H, int numMerged)
        {
            int blockSize = seq / numMerged; // = merge²
            int mergedDim = H * blockSize;
            Tensor outp = new Tensor(new TensorShape(1, numMerged, mergedDim), DType.F32);
            // consecutive blockSize patch tokens are one 2×2 block (merge-block order) → concat their hidden dims.
            long bytes = (long)mergedDim * sizeof(float);
            Buffer.MemoryCopy((void*)hidden3d.DataPointer, (void*)outp.DataPointer, (long)numMerged * bytes, (long)numMerged * bytes);
            return outp;
        }
    }
}
