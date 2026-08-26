using HartsyInference.Core.Configuration;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Multimodal;

/// <summary>The Qwen2.5-VL vision tower + patch-merger, loaded from llama.cpp's <c>mmproj-*.gguf</c>; unlike the SigLIP/CLIP towers, it uses a Conv3D patch embed, 2D rotary position embeddings (no learned position table), window attention (full attention only on layers 7/15/23/31), RMSNorm, a SwiGLU MLP, and a 2×2 patch-merger.</summary>
/// <remarks>Patches are processed in merge-block order and reordered into window-contiguous groups for the windowed layers. The 2D-RoPE application, window partitioning, and merge-block patchify run host-side (one-time, off the decode hot path); the heavy matmuls/attention run through <see cref="IBackend"/>.</remarks>
public sealed unsafe class Qwen25VlEncoder : IVlmImageEncoder
{
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    private readonly Tensor _patchWSum;   // w0 + w1 (temporal frames identical for a single image) → [hidden, 3*patch*patch]
    private int _disposed;

    public int Hidden { get; }
    public int NumLayers { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int Intermediate { get; }
    public int PatchSize { get; }
    public int Merge { get; }            // spatial_merge_size (2)
    public int WindowPatches { get; }    // window_size / patch / merge (merged units per window side, =4)
    public int ImageSize { get; }
    public int ProjectionDim { get; }    // text hidden
    public float[] ImageMean { get; }
    public float[] ImageStd { get; }
    public float Eps { get; }
    public bool UseRmsNorm { get; }      // Qwen2.5-VL = RMSNorm; Qwen2-VL = LayerNorm (with bias)
    public bool GatedMlp { get; }        // Qwen2.5-VL = SwiGLU (ffn_gate); Qwen2-VL = non-gated GELU
    private readonly HashSet<int> _fullAtt;

    public int TokensPerImage { get; private set; }
    public string Family => "qwen25vl";

    private Qwen25VlEncoder(GgufModelLoader.LoadedGgufModel handle, IReadOnlyDictionary<string, Tensor> w, Tensor patchWSum,
        int hidden, int layers, int heads, int inter, int patch, int merge, int windowPatches, int image, int projDim,
        float[] mean, float[] std, float eps, HashSet<int> fullAtt, bool useRmsNorm, bool gatedMlp)
    {
        _handle = handle; _w = w; _patchWSum = patchWSum;
        UseRmsNorm = useRmsNorm; GatedMlp = gatedMlp;
        Hidden = hidden; NumLayers = layers; NumHeads = heads; HeadDim = hidden / heads; Intermediate = inter;
        PatchSize = patch; Merge = merge; WindowPatches = windowPatches; ImageSize = image; ProjectionDim = projDim;
        ImageMean = mean; ImageStd = std; Eps = eps <= 0f ? 1e-6f : eps; _fullAtt = fullAtt;
        int g = image / patch; TokensPerImage = (g * g) / (merge * merge);
    }

    public static Qwen25VlEncoder Load(string mmprojPath)
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
            int merge = 2;
            int windowPx = 112;   // Qwen2.5-VL window size
            int windowPatches = windowPx / patch / merge;
            float eps = m.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
            float[] mean = m.GetFloatArray("clip.vision.image_mean") ?? [0.48145467f, 0.45782750f, 0.40821072f];
            float[] std = m.GetFloatArray("clip.vision.image_std") ?? [0.26862955f, 0.26130259f, 0.27577710f];
            // Qwen2-VL vs Qwen2.5-VL: 2-VL uses LayerNorm (ln1.bias present) + non-gated GELU MLP (no ffn_gate) +
            // FULL attention everywhere (no n_wa_pattern); 2.5-VL uses RMSNorm + SwiGLU + windowed attention.
            bool useRmsNorm = !w.ContainsKey("v.blk.0.ln1.bias");
            bool gatedMlp = w.ContainsKey("v.blk.0.ffn_gate.weight");
            bool windowed = m.ContainsKey("clip.vision.n_wa_pattern");
            int waPattern = (int)m.GetUInt32("clip.vision.n_wa_pattern", 8u);
            // full-attention layers: every waPattern-th (2.5-VL); ALL layers when not windowed (2-VL).
            HashSet<int> fullAtt = new();
            for (int i = 0; i < layers; i++)
                if (!windowed || (i + 1) % waPattern == 0) fullAtt.Add(i);

            // Relabel nn.Linear weights (attn q/k/v/out, ffn gate/up/down, merger mm.0/mm.2) to [out, in].
            foreach (string key in w.Keys.ToList())
            {
                if (!key.EndsWith(".weight", StringComparison.Ordinal) || w[key].Shape.Rank != 2) continue;
                if (key.Contains("attn_q") || key.Contains("attn_k") || key.Contains("attn_v") || key.Contains("attn_out")
                    || key.Contains("ffn_gate") || key.Contains("ffn_up") || key.Contains("ffn_down")
                    || key == "mm.0.weight" || key == "mm.2.weight")
                    w[key] = TensorCasts.RelabelRank2Copy(w[key]);
            }
            // Patch embed: two temporal conv weights [hidden,3,patch,patch] → [hidden, 3*patch*patch]; summed because
            // the single image fills both temporal frames identically.
            // Qwen2-VL stores feed_forward_length=0; derive the ViT intermediate from the ffn_up tensor (order-agnostic).
            if (inter <= 0) inter = (int)(w["v.blk.0.ffn_up.weight"].ElementCount / hidden);
            int pin = 3 * patch * patch;
            Tensor w0 = w["v.patch_embd.weight"], w1 = w["v.patch_embd.weight.1"];
            Tensor wSum = new(new TensorShape(hidden, pin), DType.F32);
            float* d = (float*)wSum.DataPointer; float* a = (float*)w0.DataPointer; float* b = (float*)w1.DataPointer;
            for (long i = 0; i < (long)hidden * pin; i++) d[i] = a[i] + b[i];

            return new Qwen25VlEncoder(handle, w, wSum, hidden, layers, heads, inter, patch, merge, windowPatches, image, projDim, mean, std, eps, fullAtt, useRmsNorm, gatedMlp);
        }
        catch { handle.Dispose(); throw; }
    }

    private Tensor W(string key) => _w[key];

    private static void Dbg(IBackend backend, string tag, Tensor t)
    {
        if (!EngineKnobs.VlmDebug.Value && EngineKnobs.VlmDump.Value is null) return;
        backend.Sync();
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount; double sum = 0, max = 0;
        for (long i = 0; i < n; i++) { float v = p[i]; sum += v; max = Math.Max(max, Math.Abs(v)); }
        if (EngineKnobs.VlmDebug.Value)
            Console.Error.WriteLine($"[qvis:{tag}] mean={sum / n:F4} maxabs={max:F4} [0..2]=[{p[0]:F3},{p[1]:F3},{p[2]:F3}]");
        string? dir = EngineKnobs.VlmDump.Value;
        if (dir is not null)
        {
            Directory.CreateDirectory(dir);
            byte[] bytes = new byte[n * 4];
            fixed (byte* bb = bytes) Buffer.MemoryCopy(p, bb, bytes.Length, bytes.Length);
            File.WriteAllBytes(Path.Combine(dir, $"cs_{tag}.f32"), bytes);
        }
    }

    /// <summary>Encodes a preprocessed image <c>[1, 3, H, W]</c> into <c>[1, tokens, textHidden]</c>.</summary>
    public Tensor Encode(IBackend backend, Tensor pixelValues)
    {
        int hidden = Hidden, patch = PatchSize, m = Merge, heads = NumHeads, hd = HeadDim;
        int H = (int)pixelValues.Shape[2], Wd = (int)pixelValues.Shape[3];
        int gh = H / patch, gw = Wd / patch, np = gh * gw;
        TokensPerImage = np / (m * m);

        Tensor patches = QwenVlOps.Patchify(backend, pixelValues, gh, gw, m, patch);
        Dbg(backend, "patches", patches);

        // 2. Patch embed: [np, hidden].
        Tensor embed = new(new TensorShape(np, hidden), DType.F32);
        backend.Linear(embed, patches, _patchWSum, null);
        patches.Dispose();
        Dbg(backend, "embed", embed);

        // 3. 2D RoPE cos/sin [np, headDim] + window index, then reorder hidden + cos/sin into window order.
        (float[] cos, float[] sin) = QwenVlOps.BuildRope(gh, gw, hd, m);
        (int[] winIdx, int[] cuWin) = WindowIndex(gh, gw);
        int[] mergeOrder = ExpandToPatches(winIdx, m);   // patch-level permutation (np)
        Tensor h = ReorderRows(embed, mergeOrder, hidden); embed.Dispose();
        float[] cosW = ReorderVec(cos, mergeOrder, hd), sinW = ReorderVec(sin, mergeOrder, hd);
        // Upload once for the whole tower (same table every layer) instead of re-deriving on the host per Block —
        // IBackend.ApplyRopeSingle already runs this exact rotate-half math as a GPU kernel (GenericTransformer's
        // own RoPE uses it identically); the host loop + backend.Sync() this replaced ran twice per layer.
        using Tensor cosT = new(new TensorShape(1, np, hd), DType.F32);
        using Tensor sinT = new(new TensorShape(1, np, hd), DType.F32);
        new ReadOnlySpan<float>(cosW).CopyTo(new Span<float>((float*)cosT.DataPointer, np * hd));
        new ReadOnlySpan<float>(sinW).CopyTo(new Span<float>((float*)sinT.DataPointer, np * hd));
        Dbg(backend, "embed_win", h);

        // Build the two attention masks (full / windowed block-diagonal) once.
        using Tensor fullMask = BuildMask(np, cuWin, full: true);
        using Tensor winMask = BuildMask(np, cuWin, full: false);

        for (int i = 0; i < NumLayers; i++)
        {
            Tensor next = Block(backend, h, i, cosT, sinT, np, _fullAtt.Contains(i) ? fullMask : winMask);
            h.Dispose(); h = next;
            if (i == 0) Dbg(backend, "blk0", h);
        }
        // post-LN (RMSNorm).
        Tensor post = new(new TensorShape(1, np, hidden), DType.F32);
        Norm(backend, post, h.Reshape(new TensorShape(1, np, hidden)), "v.post_ln");
        h.Dispose();
        Dbg(backend, "postln", post);

        // 4. Merger: group 4 consecutive (merge-unit) patches → [np/4, hidden*4] → mm.0 → GELU → mm.2.
        int nTok = np / (m * m), mergeDim = hidden * m * m;
        Tensor grouped = post.Reshape(new TensorShape(1, nTok, mergeDim));
        Tensor mid = new(new TensorShape(1, nTok, mergeDim), DType.F32);
        backend.Linear(mid, grouped, W("mm.0.weight"), W("mm.0.bias"));
        post.Dispose();
        Tensor act = new(new TensorShape(1, nTok, mergeDim), DType.F32);
        backend.Gelu(act, mid); mid.Dispose();
        Tensor merged = new(new TensorShape(1, nTok, ProjectionDim), DType.F32);
        backend.Linear(merged, act, W("mm.2.weight"), W("mm.2.bias")); act.Dispose();

        // 5. Un-reorder tokens (merge-units) back to spatial order.
        int[] tokIdx = new int[nTok];
        for (int i = 0; i < nTok; i++) tokIdx[winIdx[i]] = i;   // reverse permutation
        Tensor img = ReorderRows(merged, tokIdx, ProjectionDim); merged.Dispose();
        Dbg(backend, "embeds", img);
        return img;
    }

    /// <summary>Vision-tower norm: RMSNorm (Qwen2.5-VL) or LayerNorm with bias (Qwen2-VL), keyed by <paramref name="prefix"/> (<c>.weight</c>/<c>.bias</c>).</summary>
    private void Norm(IBackend backend, Tensor outp, Tensor inp, string prefix)
    {
        if (UseRmsNorm) backend.RmsNorm(outp, inp, W($"{prefix}.weight"), Eps);
        else backend.LayerNorm(outp, inp, W($"{prefix}.weight"), W($"{prefix}.bias"), Eps);
    }

    private Tensor Block(IBackend backend, Tensor x, int i, Tensor cosT, Tensor sinT, int np, Tensor mask)
    {
        int hidden = Hidden, heads = NumHeads, hd = HeadDim;
        string p = $"v.blk.{i}";
        TensorShape flat3 = new(1, np, hidden);

        Tensor ln1 = new(flat3, DType.F32);
        Norm(backend, ln1, x.Reshape(flat3), $"{p}.ln1");

        Tensor q = new(new TensorShape(1, np, heads, hd), DType.F32);
        Tensor k = new(new TensorShape(1, np, heads, hd), DType.F32);
        Tensor v = new(new TensorShape(1, np, heads, hd), DType.F32);
        backend.Linear(q, ln1, W($"{p}.attn_q.weight"), W($"{p}.attn_q.bias"));
        backend.Linear(k, ln1, W($"{p}.attn_k.weight"), W($"{p}.attn_k.bias"));
        backend.Linear(v, ln1, W($"{p}.attn_v.weight"), W($"{p}.attn_v.bias"));
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
        backend.Linear(attnOut, attnFlat, W($"{p}.attn_out.weight"), W($"{p}.attn_out.bias"));
        attnFlat.Dispose();
        Tensor afterAttn = new(flat3, DType.F32);
        backend.Add(afterAttn, x.Reshape(flat3), attnOut);
        attnOut.Dispose();

        // MLP: Qwen2.5-VL = SwiGLU (down(silu(gate)·up)); Qwen2-VL = non-gated GELU (down(gelu(up))).
        Tensor ln2 = new(flat3, DType.F32);
        Norm(backend, ln2, afterAttn, $"{p}.ln2");
        Tensor up = new(new TensorShape(1, np, Intermediate), DType.F32);
        backend.Linear(up, ln2, W($"{p}.ffn_up.weight"), W($"{p}.ffn_up.bias"));
        Tensor gu = new(new TensorShape(1, np, Intermediate), DType.F32);
        if (GatedMlp)
        {
            Tensor gate = new(new TensorShape(1, np, Intermediate), DType.F32);
            backend.Linear(gate, ln2, W($"{p}.ffn_gate.weight"), W($"{p}.ffn_gate.bias"));
            backend.Silu(gate, gate);
            backend.Mul(gu, gate, up); gate.Dispose();
        }
        else backend.Gelu(gu, up);
        ln2.Dispose(); up.Dispose();
        Tensor down = new(flat3, DType.F32);
        backend.Linear(down, gu, W($"{p}.ffn_down.weight"), W($"{p}.ffn_down.bias")); gu.Dispose();
        Tensor result = new(flat3, DType.F32);
        backend.Add(result, afterAttn, down);
        afterAttn.Dispose(); down.Dispose();
        // Barrier once per block: the ViT runs a single time per image (off the decode hot path), and at large patch
        // counts (e.g. Qwen2-VL 560px → 1600 patches) the stream-ordered activation pool can otherwise recycle a
        // buffer still in flight across the next block's Reshape. A per-block sync is negligible here and serializes.
        backend.Sync();
        return result;
    }

    /// <summary>Window index in merge-units (matches HF Qwen2.5-VL get_window_index): returns the merge-unit permutation and the cumulative patch-level window boundaries (cu_seqlens) for the block-diagonal mask.</summary>
    private (int[] winIdx, int[] cuWin) WindowIndex(int gh, int gw)
    {
        int m = Merge, vw = WindowPatches;     // merged units per window side (4)
        int lh = gh / m, lw = gw / m;
        int padH = (vw - lh % vw) % vw, padW = (vw - lw % vw) % vw;
        int nwh = (lh + padH) / vw, nww = (lw + padW) / vw;
        List<int> winIdx = new();
        List<int> cu = new() { 0 };
        for (int wh = 0; wh < nwh; wh++)
            for (int ww = 0; ww < nww; ww++)
            {
                int count = 0;
                for (int a = 0; a < vw; a++)
                    for (int b = 0; b < vw; b++)
                    {
                        int r = wh * vw + a, c = ww * vw + b;
                        if (r < lh && c < lw) { winIdx.Add(r * lw + c); count++; }
                    }
                cu.Add(cu[^1] + count * m * m);   // patch count
            }
        return (winIdx.ToArray(), cu.ToArray());
    }

    /// <summary>Expands a merge-unit permutation to a patch-level permutation (each unit = m*m consecutive patches).</summary>
    private int[] ExpandToPatches(int[] winIdx, int m)
    {
        int u = m * m;
        int[] o = new int[winIdx.Length * u];
        for (int i = 0; i < winIdx.Length; i++)
            for (int j = 0; j < u; j++)
                o[i * u + j] = winIdx[i] * u + j;
        return o;
    }

    private Tensor ReorderRows(Tensor src, int[] rowIdx, int dim)
    {
        int n = rowIdx.Length;
        Tensor o = new(new TensorShape(1, n, dim), DType.F32);
        float* s = (float*)src.DataPointer; float* d = (float*)o.DataPointer;
        for (int i = 0; i < n; i++)
            Buffer.MemoryCopy(s + (long)rowIdx[i] * dim, d + (long)i * dim, (long)dim * 4, (long)dim * 4);
        return o;
    }

    private float[] ReorderVec(float[] src, int[] rowIdx, int dim)
    {
        float[] o = new float[(long)rowIdx.Length * dim];
        for (int i = 0; i < rowIdx.Length; i++)
            Array.Copy(src, (long)rowIdx[i] * dim, o, (long)i * dim, dim);
        return o;
    }

    /// <summary>Builds a [1,1,np,np] additive attention mask (0 visible, -inf masked): full attention if <paramref name="full"/>, else block-diagonal over the window boundaries <paramref name="cuWin"/>.</summary>
    private static Tensor BuildMask(int np, int[] cuWin, bool full)
    {
        Tensor mask = new(new TensorShape(1, 1, np, np), DType.F32);
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

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _patchWSum.Dispose();
        _handle.Dispose();
    }
}
