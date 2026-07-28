using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Multimodal;

/// <summary>The Qwen3-VL / Qwen3.5-VL vision tower (<c>clip.projector_type = qwen3vl_merger</c>), loaded from
/// llama.cpp's <c>mmproj-*.gguf</c>. Structurally close to <see cref="Qwen25VlEncoder"/>'s Qwen2-VL path —
/// LayerNorm (with bias), non-gated tanh-GELU MLP, FULL attention on every block (no window schedule), 2D-RoPE,
/// 2×2 patch-merger — but with two Qwen3-specific additions:
/// <list type="bullet">
/// <item>a <b>fused QKV</b> projection (<c>v.blk.N.attn_qkv</c>), split into per-head q/k/v at load; and</item>
/// <item>a <b>learned absolute position embedding</b> (<c>v.position_embd</c>, a √P×√P grid) bilinearly
/// interpolated to the actual patch grid and ADDED to the patch embeddings before the blocks (in addition to the
/// 2D-RoPE).</item>
/// </list>
/// <para><b>DeepStack</b> (multi-decoder-layer visual injection) is NOT implemented here: the mmproj this was
/// built against (<c>unsloth/Qwen3.5-0.8B-GGUF</c>) carries no <c>deepstack_{norm,fc1,fc2}</c> merger tensors, so
/// llama.cpp runs it with <c>n_deepstack_layers = 0</c>. If a future mmproj carries those tensors, the encoder
/// must emit the extra streams and the decoder must add them at layers 0..n-1 — see the phased plan.</para></summary>
public sealed unsafe class Qwen3VlEncoder : IVlmImageEncoder
{
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    private readonly Tensor _patchWSum;   // w0 + w1 (temporal frames summed) → [hidden, 3*patch*patch]
    private readonly Tensor? _patchBias;  // [hidden]
    private readonly Tensor? _posEmbd;    // [P, hidden] learned grid; null if absent
    private readonly int _posGrid;        // √P (e.g. 48)
    private int _disposed;

    public int Hidden { get; }
    public int NumLayers { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int Intermediate { get; }
    public int PatchSize { get; }
    public int Merge { get; }
    public int ImageSizeVal { get; }
    public int ProjectionDim { get; }
    public float[] ImageMean { get; }
    public float[] ImageStd { get; }
    public float Eps { get; }

    public int TokensPerImage { get; private set; }
    public int ImageSize => ImageSizeVal;
    public string Family => "qwen3vl";

    private Qwen3VlEncoder(GgufModelLoader.LoadedGgufModel handle, IReadOnlyDictionary<string, Tensor> w,
        Tensor patchWSum, Tensor? patchBias, Tensor? posEmbd, int posGrid,
        int hidden, int layers, int heads, int inter, int patch, int merge, int image, int projDim,
        float[] mean, float[] std, float eps)
    {
        _handle = handle; _w = w; _patchWSum = patchWSum; _patchBias = patchBias; _posEmbd = posEmbd; _posGrid = posGrid;
        Hidden = hidden; NumLayers = layers; NumHeads = heads; HeadDim = hidden / heads; Intermediate = inter;
        PatchSize = patch; Merge = merge; ImageSizeVal = image; ProjectionDim = projDim;
        ImageMean = mean; ImageStd = std; Eps = eps <= 0f ? 1e-6f : eps;
        int g = image / patch; TokensPerImage = (g * g) / (merge * merge);
    }

    /// <summary>Shape-relabels a GGUF nn.Linear weight [in,out] → [out,in] via a copy.</summary>
    private static Tensor Relabel(Tensor t)
    {
        Tensor outp = new(new TensorShape((int)t.Shape[1], (int)t.Shape[0]), DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)outp.DataPointer, outp.ElementCount * 4, t.ElementCount * 4);
        return outp;
    }

    /// <summary>Copies rows [<paramref name="row0"/>, row0+rows) of a 2D [R,C] tensor into a fresh [rows,C].</summary>
    private static Tensor SliceRows(Tensor t, int row0, int rows)
    {
        int cols = (int)t.Shape[1];
        Tensor outp = new(new TensorShape(rows, cols), DType.F32);
        Buffer.MemoryCopy((float*)t.DataPointer + (long)row0 * cols, (void*)outp.DataPointer,
            (long)rows * cols * 4, (long)rows * cols * 4);
        return outp;
    }

    /// <summary>Copies elements [<paramref name="e0"/>, e0+n) of a rank-1 tensor into a fresh [n].</summary>
    private static Tensor SliceVec(Tensor t, int e0, int n)
    {
        Tensor outp = new(new TensorShape(n), DType.F32);
        Buffer.MemoryCopy((float*)t.DataPointer + (long)e0, (void*)outp.DataPointer, (long)n * 4, (long)n * 4);
        return outp;
    }

    public static Qwen3VlEncoder Load(string mmprojPath)
    {
        (Dictionary<string, Tensor> w, GgufModelLoader.LoadedGgufModel handle) = GgufModelLoader.LoadDequantized(mmprojPath, DType.F32);
        try
        {
            GgufMetadata m = handle.Metadata;
            int hidden = (int)m.GetUInt32("clip.vision.embedding_length");
            int layers = (int)m.GetUInt32("clip.vision.block_count");
            int heads = (int)m.GetUInt32("clip.vision.attention.head_count");
            int inter = (int)m.GetUInt32("clip.vision.feed_forward_length");
            int patch = (int)m.GetUInt32("clip.vision.patch_size");
            int image = (int)m.GetUInt32("clip.vision.image_size");
            int projDim = (int)m.GetUInt32("clip.vision.projection_dim");
            int merge = (int)m.GetUInt32("clip.vision.spatial_merge_size", 2u);
            float eps = m.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
            float[] mean = m.GetFloatArray("clip.vision.image_mean") ?? [0.48145467f, 0.45782750f, 0.40821072f];
            float[] std = m.GetFloatArray("clip.vision.image_std") ?? [0.26862955f, 0.26130259f, 0.27577710f];

            // Split the fused per-block QKV ([3*hidden, hidden] post-relabel; bias [3*hidden]) into separate
            // q/k/v tensors so the block math matches the standard separate-projection path. Concatenation order is
            // [q | k | v] along the output (row) axis.
            for (int i = 0; i < layers; i++)
            {
                string p = $"v.blk.{i}";
                if (!w.ContainsKey($"{p}.attn_qkv.weight")) continue;
                Tensor qkvW = Relabel(w[$"{p}.attn_qkv.weight"]);   // [3H, H]
                w[$"{p}.attn_q.weight"] = SliceRows(qkvW, 0, hidden);
                w[$"{p}.attn_k.weight"] = SliceRows(qkvW, hidden, hidden);
                w[$"{p}.attn_v.weight"] = SliceRows(qkvW, 2 * hidden, hidden);
                qkvW.Dispose();
                if (w.TryGetValue($"{p}.attn_qkv.bias", out Tensor? qkvB))
                {
                    w[$"{p}.attn_q.bias"] = SliceVec(qkvB, 0, hidden);
                    w[$"{p}.attn_k.bias"] = SliceVec(qkvB, hidden, hidden);
                    w[$"{p}.attn_v.bias"] = SliceVec(qkvB, 2 * hidden, hidden);
                }
            }

            // Relabel remaining nn.Linear weights (attn_out, ffn up/down, merger mm.0/mm.2) to [out, in].
            foreach (string key in w.Keys.ToList())
            {
                if (!key.EndsWith(".weight", StringComparison.Ordinal) || w[key].Shape.Rank != 2) continue;
                if (key.Contains("attn_out") || key.Contains("ffn_up") || key.Contains("ffn_down")
                    || key == "mm.0.weight" || key == "mm.2.weight")
                    w[key] = Relabel(w[key]);
            }

            // Patch embed: two temporal conv frames [hidden,3,patch,patch] → summed [hidden, 3*patch*patch]
            // (a single image duplicates the two temporal frames).
            if (inter <= 0) inter = (int)(w["v.blk.0.ffn_up.weight"].ElementCount / hidden);
            int pin = 3 * patch * patch;
            Tensor w0 = w["v.patch_embd.weight"];
            Tensor? w1 = w.TryGetValue("v.patch_embd.weight.1", out Tensor? tw1) ? tw1 : null;
            Tensor wSum = new(new TensorShape(hidden, pin), DType.F32);
            float* d = (float*)wSum.DataPointer; float* a = (float*)w0.DataPointer;
            if (w1 is not null) { float* b = (float*)w1.DataPointer; for (long i = 0; i < (long)hidden * pin; i++) d[i] = a[i] + b[i]; }
            else new ReadOnlySpan<float>(a, hidden * pin).CopyTo(new Span<float>(d, hidden * pin));
            Tensor? patchBias = w.TryGetValue("v.patch_embd.bias", out Tensor? pb) ? pb : null;

            // Learned absolute position embedding: [P, hidden] over a √P×√P grid, bilinearly interpolated per image.
            Tensor? posEmbd = w.TryGetValue("v.position_embd.weight", out Tensor? pe) ? pe : null;
            int posGrid = posEmbd is not null ? (int)MathF.Round(MathF.Sqrt(posEmbd.Shape[0])) : 0;

            return new Qwen3VlEncoder(handle, w, wSum, patchBias, posEmbd, posGrid,
                hidden, layers, heads, inter, patch, merge, image, projDim, mean, std, eps);
        }
        catch { handle.Dispose(); throw; }
    }

    private Tensor W(string key) => _w[key];
    private bool Has(string key) => _w.ContainsKey(key);

    /// <summary>Encodes a preprocessed image <c>[1, 3, H, W]</c> into <c>[1, tokens, textHidden]</c>.</summary>
    public Tensor Encode(IBackend backend, Tensor pixelValues)
    {
        int hidden = Hidden, patch = PatchSize, m = Merge, heads = NumHeads, hd = HeadDim;
        int H = (int)pixelValues.Shape[2], Wd = (int)pixelValues.Shape[3];
        int gh = H / patch, gw = Wd / patch, np = gh * gw, pin = 3 * patch * patch;
        TokensPerImage = np / (m * m);

        // 1. Patchify into merge-block order (bh,bw,mh,mw): groups of m*m consecutive patches form a 2×2 block, so
        //    the merger later reads consecutive-4, and merged tokens come out row-major over the (gh/m, gw/m) grid.
        Tensor patches = new(new TensorShape(np, pin), DType.F32);
        backend.Sync();
        float* px = (float*)pixelValues.DataPointer;
        float* pp = (float*)patches.DataPointer;
        int idx = 0;
        for (int bh = 0; bh < gh / m; bh++)
            for (int bw = 0; bw < gw / m; bw++)
                for (int mh = 0; mh < m; mh++)
                    for (int mw = 0; mw < m; mw++)
                    {
                        int ph = (bh * m + mh) * patch, pw = (bw * m + mw) * patch;
                        float* dst = pp + (long)idx * pin;
                        int o = 0;
                        for (int c = 0; c < 3; c++)
                            for (int yy = 0; yy < patch; yy++)
                                for (int xx = 0; xx < patch; xx++)
                                    dst[o++] = px[((long)c * H + (ph + yy)) * Wd + (pw + xx)];
                        idx++;
                    }

        // 2. Patch embed (+ bias) → [np, hidden].
        Tensor embed = new(new TensorShape(np, hidden), DType.F32);
        backend.Linear(embed, patches, _patchWSum, _patchBias);
        patches.Dispose();

        // 3. Add the bilinearly-interpolated learned position embedding (merge-block order), if present.
        if (_posEmbd is not null)
        {
            backend.Sync();
            AddInterpolatedPosEmbed(embed, gh, gw, m);
        }

        // 4. 2D-RoPE cos/sin tables [np, headDim] in the same merge-block order.
        (float[] cos, float[] sin) = BuildRope(gh, gw);
        using Tensor cosT = new(new TensorShape(1, np, hd), DType.F32);
        using Tensor sinT = new(new TensorShape(1, np, hd), DType.F32);
        new ReadOnlySpan<float>(cos).CopyTo(new Span<float>((float*)cosT.DataPointer, np * hd));
        new ReadOnlySpan<float>(sin).CopyTo(new Span<float>((float*)sinT.DataPointer, np * hd));

        // Full (non-windowed) attention: a single all-zeros additive mask.
        using Tensor fullMask = new(new TensorShape(1, 1, np, np), DType.F32);
        new Span<float>((float*)fullMask.DataPointer, np * np).Clear();

        Tensor h = embed;
        for (int i = 0; i < NumLayers; i++)
        {
            Tensor next = Block(backend, h, i, cosT, sinT, np, fullMask);
            h.Dispose(); h = next;
        }

        // post-LN.
        Tensor post = new(new TensorShape(1, np, hidden), DType.F32);
        backend.LayerNorm(post, h.Reshape(new TensorShape(1, np, hidden)), W("v.post_ln.weight"), W("v.post_ln.bias"), Eps);
        h.Dispose();

        // 5. Merger: group m*m consecutive patches → [nTok, hidden*m*m] → mm.0 → GELU → mm.2 → [nTok, projDim].
        int nTok = np / (m * m), mergeDim = hidden * m * m;
        Tensor grouped = post.Reshape(new TensorShape(1, nTok, mergeDim));
        Tensor mid = new(new TensorShape(1, nTok, mergeDim), DType.F32);
        backend.Linear(mid, grouped, W("mm.0.weight"), Has("mm.0.bias") ? W("mm.0.bias") : null);
        post.Dispose();
        Tensor act = new(new TensorShape(1, nTok, mergeDim), DType.F32);
        backend.Gelu(act, mid); mid.Dispose();
        Tensor merged = new(new TensorShape(1, nTok, ProjectionDim), DType.F32);
        backend.Linear(merged, act, W("mm.2.weight"), Has("mm.2.bias") ? W("mm.2.bias") : null);
        act.Dispose();
        // Merged tokens are already in row-major order over the (gh/m, gw/m) grid — no un-reorder needed.
        return merged;
    }

    private Tensor Block(IBackend backend, Tensor x, int i, Tensor cosT, Tensor sinT, int np, Tensor mask)
    {
        int hidden = Hidden, heads = NumHeads, hd = HeadDim;
        string p = $"v.blk.{i}";
        TensorShape flat3 = new(1, np, hidden);

        Tensor ln1 = new(flat3, DType.F32);
        backend.LayerNorm(ln1, x.Reshape(flat3), W($"{p}.ln1.weight"), W($"{p}.ln1.bias"), Eps);

        Tensor q = new(new TensorShape(1, np, heads, hd), DType.F32);
        Tensor k = new(new TensorShape(1, np, heads, hd), DType.F32);
        Tensor v = new(new TensorShape(1, np, heads, hd), DType.F32);
        backend.Linear(q, ln1, W($"{p}.attn_q.weight"), Has($"{p}.attn_q.bias") ? W($"{p}.attn_q.bias") : null);
        backend.Linear(k, ln1, W($"{p}.attn_k.weight"), Has($"{p}.attn_k.bias") ? W($"{p}.attn_k.bias") : null);
        backend.Linear(v, ln1, W($"{p}.attn_v.weight"), Has($"{p}.attn_v.bias") ? W($"{p}.attn_v.bias") : null);
        ln1.Dispose();
        backend.ApplyRopeSingle(q, cosT, sinT);
        backend.ApplyRopeSingle(k, cosT, sinT);

        Tensor qM = new(new TensorShape(1, heads, np, hd), DType.F32);
        Tensor kM = new(new TensorShape(1, heads, np, hd), DType.F32);
        Tensor vM = new(new TensorShape(1, heads, np, hd), DType.F32);
        backend.Permute0213(qM, q, np, heads, hd);
        backend.Permute0213(kM, k, np, heads, hd);
        backend.Permute0213(vM, v, np, heads, hd);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor attn = new(new TensorShape(1, heads, np, hd), DType.F32);
        backend.ScaledDotProductAttention(attn, qM, kM, vM, mask, 1f / MathF.Sqrt(hd));
        qM.Dispose(); kM.Dispose(); vM.Dispose();

        Tensor attnFlat = new(flat3, DType.F32);
        backend.Permute0213(attnFlat, attn, heads, np, hd);
        attn.Dispose();
        Tensor attnOut = new(flat3, DType.F32);
        backend.Linear(attnOut, attnFlat, W($"{p}.attn_out.weight"), Has($"{p}.attn_out.bias") ? W($"{p}.attn_out.bias") : null);
        attnFlat.Dispose();
        Tensor afterAttn = new(flat3, DType.F32);
        backend.Add(afterAttn, x.Reshape(flat3), attnOut);
        attnOut.Dispose();

        // Non-gated GELU MLP.
        Tensor ln2 = new(flat3, DType.F32);
        backend.LayerNorm(ln2, afterAttn, W($"{p}.ln2.weight"), W($"{p}.ln2.bias"), Eps);
        Tensor up = new(new TensorShape(1, np, Intermediate), DType.F32);
        backend.Linear(up, ln2, W($"{p}.ffn_up.weight"), Has($"{p}.ffn_up.bias") ? W($"{p}.ffn_up.bias") : null);
        Tensor gu = new(new TensorShape(1, np, Intermediate), DType.F32);
        backend.Gelu(gu, up);
        ln2.Dispose(); up.Dispose();
        Tensor down = new(flat3, DType.F32);
        backend.Linear(down, gu, W($"{p}.ffn_down.weight"), Has($"{p}.ffn_down.bias") ? W($"{p}.ffn_down.bias") : null);
        gu.Dispose();
        Tensor result = new(flat3, DType.F32);
        backend.Add(result, afterAttn, down);
        afterAttn.Dispose(); down.Dispose();
        backend.Sync();
        return result;
    }

    /// <summary>Adds the learned position embedding to <paramref name="embed"/> (host [np, hidden] in merge-block
    /// order). The stored √P×√P grid is bilinearly interpolated (align_corners=false) to the (gh, gw) patch grid.</summary>
    private void AddInterpolatedPosEmbed(Tensor embed, int gh, int gw, int m)
    {
        int hidden = Hidden, bg = _posGrid;
        float* e = (float*)embed.DataPointer;
        float* pe = (float*)_posEmbd!.DataPointer;   // [bg*bg, hidden]
        float sh = bg / (float)gh, sw = bg / (float)gw;
        int idx = 0;
        for (int bh = 0; bh < gh / m; bh++)
            for (int bw = 0; bw < gw / m; bw++)
                for (int mh = 0; mh < m; mh++)
                    for (int mw = 0; mw < m; mw++)
                    {
                        int th = bh * m + mh, tw = bw * m + mw;
                        float fy = (th + 0.5f) * sh - 0.5f, fx = (tw + 0.5f) * sw - 0.5f;
                        int y0 = (int)MathF.Floor(fy), x0 = (int)MathF.Floor(fx);
                        float dy = fy - y0, dx = fx - x0;
                        int y0c = Math.Clamp(y0, 0, bg - 1), y1c = Math.Clamp(y0 + 1, 0, bg - 1);
                        int x0c = Math.Clamp(x0, 0, bg - 1), x1c = Math.Clamp(x0 + 1, 0, bg - 1);
                        float w00 = (1 - dy) * (1 - dx), w01 = (1 - dy) * dx, w10 = dy * (1 - dx), w11 = dy * dx;
                        float* r00 = pe + (long)(y0c * bg + x0c) * hidden;
                        float* r01 = pe + (long)(y0c * bg + x1c) * hidden;
                        float* r10 = pe + (long)(y1c * bg + x0c) * hidden;
                        float* r11 = pe + (long)(y1c * bg + x1c) * hidden;
                        float* dstr = e + (long)idx * hidden;
                        for (int c = 0; c < hidden; c++)
                            dstr[c] += w00 * r00[c] + w01 * r01[c] + w10 * r10[c] + w11 * r11[c];
                        idx++;
                    }
    }

    /// <summary>2D rotary cos/sin tables [np, headDim] in merge-block order (h-freqs on the first quarter, w-freqs
    /// on the second, mirrored on the upper half — the standard Qwen-VL vision 2D-RoPE with rotate_half layout).</summary>
    private (float[] cos, float[] sin) BuildRope(int gh, int gw)
    {
        int hd = HeadDim, m = Merge, np = gh * gw, ropeDim = hd / 2;
        int freqN = ropeDim / 2;
        float[] inv = new float[freqN];
        for (int i = 0; i < freqN; i++) inv[i] = 1f / MathF.Pow(10000f, (2f * i) / ropeDim);
        int[] hpos = new int[np], wpos = new int[np];
        int idx = 0;
        for (int bh = 0; bh < gh / m; bh++)
            for (int bw = 0; bw < gw / m; bw++)
                for (int mh = 0; mh < m; mh++)
                    for (int mw = 0; mw < m; mw++)
                    { hpos[idx] = bh * m + mh; wpos[idx] = bw * m + mw; idx++; }
        float[] cos = new float[np * hd], sin = new float[np * hd];
        for (int pos = 0; pos < np; pos++)
            for (int j = 0; j < freqN; j++)
            {
                float fh = hpos[pos] * inv[j], fw = wpos[pos] * inv[j];
                SetCosSin(cos, sin, pos, hd, j, fh);
                SetCosSin(cos, sin, pos, hd, freqN + j, fw);
                SetCosSin(cos, sin, pos, hd, ropeDim + j, fh);
                SetCosSin(cos, sin, pos, hd, ropeDim + freqN + j, fw);
            }
        return (cos, sin);
    }

    private static void SetCosSin(float[] cos, float[] sin, int pos, int hd, int e, float ang)
    {
        cos[pos * hd + e] = MathF.Cos(ang);
        sin[pos * hd + e] = MathF.Sin(ang);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _patchWSum.Dispose();
        _handle.Dispose();
    }
}
