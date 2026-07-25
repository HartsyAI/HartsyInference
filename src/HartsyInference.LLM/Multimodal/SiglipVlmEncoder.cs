using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Multimodal;

/// <summary>Connector that turns the vision patch grid into a small set of image embeddings in the text hidden
/// space. Different VLM families use different connectors.</summary>
public enum VlmProjectorKind
{
    /// <summary>Gemma-3: average-pool the patch grid (pool×pool) → RMSNorm (<c>mm.soft_emb_norm</c>) →
    /// Linear (<c>mm.input_projection</c>, a raw <c>[in,out]</c> parameter).</summary>
    Gemma3,
    /// <summary>Idefics3 / SmolVLM: pixel-shuffle the patch grid (scale_factor → fold s×s spatial into channels)
    /// → Linear (<c>mm.model.fc</c>, an nn.Linear).</summary>
    Idefics3,
    /// <summary>LLaVA: drop the CLS token, then a 2-layer MLP (<c>mm.0</c> → GELU → <c>mm.2</c>) over the patch
    /// tokens. No token reduction (576 patches → 576 image tokens).</summary>
    Llava,
    /// <summary>InternVL (InternViT + dynamic high-res): drop the CLS token, pixel-shuffle the patch grid
    /// (scale_factor → fold s×s into channels, same mapping as Idefics3), then LayerNorm (<c>mm.model.mlp.0</c>) →
    /// Linear (<c>mm.model.mlp.1</c>) → GELU → Linear (<c>mm.model.mlp.3</c>). The InternViT blocks carry LayerScale
    /// (<c>ls1</c>/<c>ls2</c>).</summary>
    InternVL,
    /// <summary>MiniCPM-V: a SigLIP tower (with integer-bucketed position selection into the 70×70 learned table)
    /// feeding a <b>perceiver resampler</b> — <c>query_num</c> learnable queries cross-attend the vision features
    /// (K = features + 2D-sincos pos, separate q/k/v/out projections, LayerNorms) → <c>resampler.proj</c> into the
    /// text hidden size.</summary>
    MiniCpmV,
}

/// <summary>The vision tower + connector of llama.cpp <c>mmproj-*.gguf</c> VLMs. Covers both ViT flavours that
/// share the <c>v.blk.*</c> tensor dialect: SigLIP (Gemma-3, SmolVLM/Idefics3 — no CLS, post-LN, GELU-tanh) and
/// CLIP (LLaVA — CLS token, pre-LN, quick-GELU, penultimate layer). A patch-embed Conv → (+CLS) → +position →
/// (pre-LN) → N pre-norm transformer blocks → (post-LN) produces per-patch features; a family-specific
/// <see cref="VlmProjectorKind">projector</see> then maps them into the text hidden size. The result is spliced
/// into the text decoder's input sequence (see <see cref="MultimodalGenerator"/>).
///
/// <para>All math runs through <see cref="IBackend"/>. ViT attention is bidirectional (non-causal); it reuses the
/// decoder's FlashAttention with <c>causal: false</c>.</para></summary>
public sealed unsafe class SiglipVlmEncoder : IVlmImageEncoder
{
    /// <summary>Prompt-format family tag (see <see cref="IVlmImageEncoder.Family"/>).</summary>
    public string Family => Projector switch
    {
        VlmProjectorKind.Idefics3 => "idefics3",
        VlmProjectorKind.Llava => "llava",
        VlmProjectorKind.InternVL => "internvl",
        VlmProjectorKind.MiniCpmV => "minicpmv",
        _ => "gemma3",
    };

    public int QueryNum { get; }          // MiniCPM-V resampler: number of learnable queries (= tokens/image)

    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    private int _disposed;

    public int Hidden { get; }
    public int NumLayers { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int Intermediate { get; }
    public int PatchSize { get; }
    public int ImageSize { get; }
    public int PatchGrid { get; }       // (image - patch) / patch + 1   (e.g. 896/14 = 64, 384/14 = 27, 336/14 = 24)
    public int NumPatches { get; }      // grid²
    public bool HasClsToken { get; }    // CLIP prepends a class embedding (sequence = NumPatches + 1)
    public int SeqLen { get; }          // NumPatches (+1 if CLS)
    public bool HasPreLn { get; }       // CLIP applies a layernorm before the blocks
    public bool HasPostLn { get; }      // SigLIP applies a final layernorm; CLIP/LLaVA uses the penultimate layer
    public bool UseQuickGelu { get; }   // CLIP MLP activation = x·sigmoid(1.702x); SigLIP = GELU-tanh
    public int ScaleFactor { get; }     // projector reduction factor (Gemma-3 avg-pool size; Idefics3 pixel-shuffle; LLaVA 1)
    public bool FfnUpIsFc1 { get; }     // true (InternViT): ffn_up = fc1 (up). false (clip SigLIP/CLIP): names swapped, ffn_down = fc1.
    public int TokensPerImage { get; }
    public int ProjectionDim { get; }   // text hidden size
    public VlmProjectorKind Projector { get; }
    public float[] ImageMean { get; }
    public float[] ImageStd { get; }
    public float LayerNormEps { get; }

    private SiglipVlmEncoder(GgufModelLoader.LoadedGgufModel handle, IReadOnlyDictionary<string, Tensor> w,
        int hidden, int layers, int heads, int inter, int patch, int image, int projDim,
        float[] mean, float[] std, float eps, VlmProjectorKind projector, int scaleFactor,
        bool hasCls, bool hasPreLn, bool hasPostLn, bool quickGelu, bool ffnUpIsFc1, int queryNum)
    {
        _handle = handle; _w = w;
        Hidden = hidden; NumLayers = layers; NumHeads = heads; HeadDim = hidden / heads; Intermediate = inter;
        PatchSize = patch; ImageSize = image; PatchGrid = (image - patch) / patch + 1; NumPatches = PatchGrid * PatchGrid;
        ProjectionDim = projDim; ImageMean = mean; ImageStd = std; LayerNormEps = eps <= 0f ? 1e-6f : eps;
        Projector = projector; ScaleFactor = scaleFactor; FfnUpIsFc1 = ffnUpIsFc1; QueryNum = queryNum;
        HasClsToken = hasCls; SeqLen = NumPatches + (hasCls ? 1 : 0); HasPreLn = hasPreLn; HasPostLn = hasPostLn; UseQuickGelu = quickGelu;
        TokensPerImage = projector == VlmProjectorKind.MiniCpmV ? queryNum
            : projector == VlmProjectorKind.Llava ? NumPatches
            : (PatchGrid / scaleFactor) * (PatchGrid / scaleFactor);
    }

    /// <summary>Transposes a 2D F32 weight to row-major <c>[d0, d1]</c>. Used for Gemma-3's
    /// <c>mm.input_projection</c>, which is a raw parameter <c>[vision_in, text_out]</c> (applied as <c>x · W</c>),
    /// whereas <see cref="IBackend.Linear"/> wants <c>[out, in]</c> — so an actual data transpose is required.</summary>
    private static Tensor TransposeWeight(Tensor t)
    {
        int d0 = (int)t.Shape[0], d1 = (int)t.Shape[1];   // engine Shape = ggml ne; data is row-major [d1, d0]
        Tensor outp = new(new TensorShape(d0, d1), DType.F32);
        float* src = (float*)t.DataPointer;   // row-major [d1, d0]
        float* dst = (float*)outp.DataPointer; // row-major [d0, d1]
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                dst[(long)i * d1 + j] = src[(long)j * d0 + i];
        return outp;
    }

    /// <summary>Relabels a 2D weight from engine shape <c>[a, b]</c> (= ggml ne) to <c>[b, a]</c> without moving data
    /// — the raw GGUF bytes are already row-major <c>[b, a]</c> = <c>[out, in]</c>, so this just fixes the shape for
    /// <see cref="IBackend.Linear"/>. Copies into an owned tensor (avoids aliasing the loader's memory).</summary>
    private static Tensor Relabel(Tensor t)
    {
        Tensor outp = new(new TensorShape((int)t.Shape[1], (int)t.Shape[0]), DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)outp.DataPointer, outp.ElementCount * 4, t.ElementCount * 4);
        return outp;
    }

    /// <summary>Loads the mmproj GGUF (vision tower + projector) and reads its config.</summary>
    public static SiglipVlmEncoder Load(string mmprojPath)
    {
        (Dictionary<string, Tensor> w, GgufModelLoader.LoadedGgufModel handle) = GgufModelLoader.LoadDequantized(mmprojPath, DType.F32);
        try
        {
            // Detect the connector by which projector tensors are present.
            VlmProjectorKind projector = w.ContainsKey("resampler.query") ? VlmProjectorKind.MiniCpmV
                : w.ContainsKey("mm.model.mlp.0.weight") ? VlmProjectorKind.InternVL
                : w.ContainsKey("mm.model.fc.weight") ? VlmProjectorKind.Idefics3
                : w.ContainsKey("mm.0.weight") ? VlmProjectorKind.Llava
                : VlmProjectorKind.Gemma3;

            // Fix weight layouts for IBackend.Linear (wants [out, in] row-major):
            //  • nn.Linear weights (attn q/k/v/out, ffn up/down, Idefics3 mm.model.fc, LLaVA mm.0/mm.2) are stored by
            //    clip with ggml ne = [in, out] but the raw bytes are already row-major [out, in] — so a *relabel*
            //    (Reshape to [ne1, ne0]) suffices, NOT a data transpose. (Same trick the text GGUF loader uses.)
            //  • Gemma-3's mm.input_projection is a raw parameter [vision_in, text_out] applied as x·W → real transpose.
            // NOTE: clip names the SigLIP/CLIP MLP projections SWAPPED — `ffn_down` is fc1 (up, hidden→intermediate)
            // and `ffn_up` is fc2 (down, intermediate→hidden); the Block wires them by role, not name.
            foreach (string key in w.Keys.ToList())
            {
                if (!key.EndsWith(".weight", StringComparison.Ordinal) || w[key].Shape.Rank != 2) continue;
                if (key.Contains("position_embd")) continue;   // position table, used as [seqLen, hidden] directly
                if (key.Contains("input_projection"))
                    w[key] = TransposeWeight(w[key]);
                else if (key.Contains("attn_q") || key.Contains("attn_k") || key.Contains("attn_v") || key.Contains("attn_out")
                    || key.Contains("ffn_up") || key.Contains("ffn_down") || key.Contains("mm.model.fc")
                    || key == "mm.0.weight" || key == "mm.2.weight"
                    || key == "mm.model.mlp.1.weight" || key == "mm.model.mlp.3.weight"   // InternVL projector Linears
                    || key == "resampler.kv.weight" || key == "resampler.proj.weight"      // MiniCPM-V resampler Linears
                    || key == "resampler.attn.q.weight" || key == "resampler.attn.k.weight"
                    || key == "resampler.attn.v.weight" || key == "resampler.attn.out.weight")
                    w[key] = Relabel(w[key]);
            }
            GgufMetadata m = handle.Metadata;
            int hidden = (int)m.GetUInt32("clip.vision.embedding_length");
            int layers = (int)m.GetUInt32("clip.vision.block_count");
            int heads = (int)m.GetUInt32("clip.vision.attention.head_count");
            int inter = (int)m.GetUInt32("clip.vision.feed_forward_length");
            int patch = (int)m.GetUInt32("clip.vision.patch_size");
            int image = (int)m.GetUInt32("clip.vision.image_size");
            // Text hidden: Gemma-3/SmolVLM expose projection_dim = text hidden; LLaVA's mm.2 output is the text hidden.
            int projDim = projector == VlmProjectorKind.Llava ? (int)w["mm.2.weight"].Shape[0]
                : projector == VlmProjectorKind.InternVL ? (int)w["mm.model.mlp.3.weight"].Shape[0]
                : projector == VlmProjectorKind.MiniCpmV ? (int)w["resampler.proj.weight"].Shape[0]
                : (int)m.GetUInt32("clip.vision.projection_dim");
            // MiniCPM-V resampler: query count = tokens/image (resampler.query ggml ne = [dim, query_num]).
            int queryNum = projector == VlmProjectorKind.MiniCpmV
                ? (int)m.GetUInt32("clip.minicpmv_query_num", (uint)w["resampler.query"].Shape[1]) : 0;
            float eps = m.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
            float[] mean = m.GetFloatArray("clip.vision.image_mean") ?? [0.5f, 0.5f, 0.5f];
            float[] std = m.GetFloatArray("clip.vision.image_std") ?? [0.5f, 0.5f, 0.5f];
            int grid = (image - patch) / patch + 1;
            int scaleFactor = projector switch
            {
                VlmProjectorKind.Idefics3 => (int)m.GetUInt32("clip.vision.projector.scale_factor", 3u),
                VlmProjectorKind.InternVL => (int)m.GetUInt32("clip.vision.projector.scale_factor", 2u),
                VlmProjectorKind.Llava or VlmProjectorKind.MiniCpmV => 1,
                // Gemma-3: pool size = grid / sqrt(mm_tokens_per_image).
                _ => grid / (int)MathF.Round(MathF.Sqrt(m.GetUInt32("clip.vision.mm_tokens_per_image", 256u))),
            };
            bool hasCls = w.ContainsKey("v.class_embd");
            bool hasPreLn = w.ContainsKey("v.pre_ln.weight");
            bool hasPostLn = w.ContainsKey("v.post_ln.weight");
            bool quickGelu = !m.GetBool("clip.use_gelu", true);   // CLIP: use_gelu=false → quick-gelu
            // MLP wiring: clip swaps the SigLIP/CLIP fc1/fc2 names (ffn_down = up), but InternViT uses the natural
            // names (ffn_up = up). Decide by the relabeled out-dim: the fc1 (up) projection produces `intermediate`.
            bool ffnUpIsFc1 = (int)w["v.blk.0.ffn_up.weight"].Shape[0] == inter;
            return new SiglipVlmEncoder(handle, w, hidden, layers, heads, inter, patch, image, projDim, mean, std, eps,
                projector, scaleFactor, hasCls, hasPreLn, hasPostLn, quickGelu, ffnUpIsFc1, queryNum);
        }
        catch { handle.Dispose(); throw; }
    }

    private Tensor W(string key) => _w[key];
    private Tensor? Wopt(string key) => _w.TryGetValue(key, out Tensor? t) ? t : null;

    /// <summary>Raw mmproj weights and GGUF metadata, for <see cref="LlavaNextEncoder"/> to read
    /// <c>clip.vision.image_grid_pinpoints</c> / <c>model.image_newline</c> off the SAME loaded instance instead
    /// of re-loading (and re-dequantizing) the mmproj file a second time.</summary>
    internal IReadOnlyDictionary<string, Tensor> Weights => _w;
    internal GgufMetadata Metadata => _handle.Metadata;

    private static void Dbg(IBackend backend, string tag, Tensor t, int hidden)
    {
        if (Environment.GetEnvironmentVariable("HARTSY_VLM_DEBUG") != "1" && Environment.GetEnvironmentVariable("HARTSY_VLM_DUMP") is null) return;
        backend.Sync();
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount; double sum = 0, max = 0;
        for (long i = 0; i < n; i++) { float v = p[i]; sum += v; max = Math.Max(max, Math.Abs(v)); }
        if (Environment.GetEnvironmentVariable("HARTSY_VLM_DEBUG") == "1")
            Console.Error.WriteLine($"[vis:{tag}] mean={sum / n:F4} maxabs={max:F4} tok0=[{p[0]:F3},{p[1]:F3},{p[2]:F3}] tok1=[{p[hidden]:F3},{p[hidden + 1]:F3}]");
        string? dir = Environment.GetEnvironmentVariable("HARTSY_VLM_DUMP");
        if (dir is not null)
        {
            Directory.CreateDirectory(dir);
            byte[] bytes = new byte[n * 4];
            fixed (byte* b = bytes) Buffer.MemoryCopy(p, b, bytes.Length, bytes.Length);
            File.WriteAllBytes(Path.Combine(dir, $"cs_{tag}.f32"), bytes);
        }
    }

    /// <summary>Encodes a preprocessed image <c>[1, 3, imageSize, imageSize]</c> into <c>[tokensPerImage, projectionDim]</c>
    /// image embeddings, ready to splice into the text decoder's input sequence.</summary>
    public Tensor Encode(IBackend backend, Tensor pixelValues)
    {
        int hidden = Hidden, np = NumPatches, grid = PatchGrid, seqLen = SeqLen, cls = HasClsToken ? 1 : 0;

        Dbg(backend, "pixels", pixelValues, hidden);

        // Patch embedding: Conv2D (kernel=stride=patch) → [1, hidden, grid, grid]. The GGUF [kw,kh,in,out] memory
        // layout equals [out,in,kh,kw] row-major, so a reshape gives the Conv2D weight layout.
        Tensor patchW = W("v.patch_embd.weight").Reshape(new TensorShape(hidden, 3, PatchSize, PatchSize));
        Tensor conv = new(new TensorShape(1, hidden, grid, grid), DType.F32);
        backend.Conv2D(conv, pixelValues, patchW, Wopt("v.patch_embd.bias"), PatchSize, PatchSize, 0, 0);   // CLIP has no conv bias

        // [1, hidden, np] → patches at [1, np, hidden]; prepend the CLS token (CLIP) at row 0 → [1, seqLen, hidden].
        Tensor convFlat = conv.Reshape(new TensorShape(1, hidden, np));
        Tensor seq = new(new TensorShape(1, seqLen, hidden), DType.F32);
        if (HasClsToken)
        {
            using Tensor patches = new(new TensorShape(1, np, hidden), DType.F32);
            backend.Transpose2D(patches, convFlat, hidden, np);
            backend.Sync();
            Buffer.MemoryCopy((void*)W("v.class_embd").DataPointer, (void*)seq.DataPointer, (long)hidden * 4, (long)hidden * 4);
            Buffer.MemoryCopy((void*)patches.DataPointer, (byte*)seq.DataPointer + (long)hidden * 4, (long)np * hidden * 4, (long)np * hidden * 4);
        }
        else
        {
            backend.Transpose2D(seq, convFlat, hidden, np);
        }
        conv.Dispose();
        AddPositions(seq);
        if (HasPreLn)
        {
            Tensor pre = new(new TensorShape(1, seqLen, hidden), DType.F32);
            backend.LayerNorm(pre, seq, W("v.pre_ln.weight"), W("v.pre_ln.bias"), LayerNormEps);
            seq.Dispose();
            seq = pre;
        }
        Dbg(backend, "seq", seq, hidden);

        Tensor h = seq;
        for (int i = 0; i < NumLayers; i++)
        {
            Tensor next = Block(backend, h, i);
            h.Dispose();
            h = next;
            if (i == 0) Dbg(backend, "blk0", h, hidden);
        }
        if (HasPostLn)
        {
            Tensor normed = new(new TensorShape(1, seqLen, hidden), DType.F32);
            backend.LayerNorm(normed, h, W("v.post_ln.weight"), W("v.post_ln.bias"), LayerNormEps);
            h.Dispose();
            h = normed;
        }
        Dbg(backend, "postln", h, hidden);

        Tensor img = Projector switch
        {
            VlmProjectorKind.Idefics3 => ProjectIdefics3(backend, h),
            VlmProjectorKind.Llava => ProjectLlava(backend, h, cls),
            VlmProjectorKind.InternVL => ProjectInternVL(backend, h, cls),
            VlmProjectorKind.MiniCpmV => ProjectMiniCpmV(backend, h),
            _ => ProjectGemma3(backend, h),
        };
        Dbg(backend, "embeds", img, ProjectionDim);
        h.Dispose();
        return img;
    }

    /// <summary>One pre-norm ViT block: LN → MHA(separate q/k/v/out, biased, bidirectional) → +res → LN → MLP(GELU) → +res.</summary>
    private Tensor Block(IBackend backend, Tensor x, int i)
    {
        int hidden = Hidden, seqLen = SeqLen, heads = NumHeads, d = HeadDim;
        string p = $"v.blk.{i}";
        TensorShape flat = new(1, seqLen, hidden);

        Tensor ln1 = new(flat, DType.F32);
        backend.LayerNorm(ln1, x, W($"{p}.ln1.weight"), W($"{p}.ln1.bias"), LayerNormEps);

        Tensor q = new(new TensorShape(1, seqLen, heads, d), DType.F32);
        Tensor k = new(new TensorShape(1, seqLen, heads, d), DType.F32);
        Tensor v = new(new TensorShape(1, seqLen, heads, d), DType.F32);
        backend.Linear(q, ln1, W($"{p}.attn_q.weight"), W($"{p}.attn_q.bias"));
        backend.Linear(k, ln1, W($"{p}.attn_k.weight"), W($"{p}.attn_k.bias"));
        backend.Linear(v, ln1, W($"{p}.attn_v.weight"), W($"{p}.attn_v.bias"));
        ln1.Dispose();

        Tensor qM = new(new TensorShape(1, heads, seqLen, d), DType.F32);
        Tensor kM = new(new TensorShape(1, heads, seqLen, d), DType.F32);
        Tensor vM = new(new TensorShape(1, heads, seqLen, d), DType.F32);
        backend.Permute0213(qM, q, seqLen, heads, d);
        backend.Permute0213(kM, k, seqLen, heads, d);
        backend.Permute0213(vM, v, seqLen, heads, d);
        q.Dispose(); k.Dispose(); v.Dispose();

        // Bidirectional attention over all tokens (non-causal).
        Tensor attn = new(new TensorShape(1, heads, seqLen, d), DType.F32);
        backend.FlashAttention(attn, qM, kM, vM, seqLen, 1, causal: false, qOffset: 0, 1f / MathF.Sqrt(d));
        qM.Dispose(); kM.Dispose(); vM.Dispose();

        Tensor attnFlat = new(flat, DType.F32);
        backend.Permute0213(attnFlat, attn, heads, seqLen, d);
        attn.Dispose();
        Tensor attnOut = new(flat, DType.F32);
        backend.Linear(attnOut, attnFlat, W($"{p}.attn_out.weight"), W($"{p}.attn_out.bias"));
        attnFlat.Dispose();
        ApplyLayerScale(backend, attnOut, $"{p}.ls1.weight");   // InternViT: per-channel LayerScale before the residual

        Tensor afterAttn = new(flat, DType.F32);
        backend.Add(afterAttn, x, attnOut);
        attnOut.Dispose();

        Tensor ln2 = new(flat, DType.F32);
        backend.LayerNorm(ln2, afterAttn, W($"{p}.ln2.weight"), W($"{p}.ln2.bias"), LayerNormEps);
        // MLP. clip swaps the names: `ffn_down` is fc1 (up, hidden→intermediate), `ffn_up` is fc2 (down).
        // fc1 (up, hidden→intermediate) and fc2 (down). clip swaps the names for SigLIP/CLIP (ffn_down = fc1);
        // InternViT uses the natural names (ffn_up = fc1). FfnUpIsFc1 picks the right tensor per role.
        string upKey = FfnUpIsFc1 ? $"{p}.ffn_up" : $"{p}.ffn_down";
        string downKey = FfnUpIsFc1 ? $"{p}.ffn_down" : $"{p}.ffn_up";
        Tensor up = new(new TensorShape(1, seqLen, Intermediate), DType.F32);
        backend.Linear(up, ln2, W($"{upKey}.weight"), W($"{upKey}.bias"));
        ln2.Dispose();
        Tensor act = new(new TensorShape(1, seqLen, Intermediate), DType.F32);
        if (UseQuickGelu) QuickGelu(backend, act, up); else backend.Gelu(act, up);
        up.Dispose();
        Tensor down = new(flat, DType.F32);
        backend.Linear(down, act, W($"{downKey}.weight"), W($"{downKey}.bias"));
        act.Dispose();
        ApplyLayerScale(backend, down, $"{p}.ls2.weight");   // InternViT: per-channel LayerScale before the residual

        Tensor result = new(flat, DType.F32);
        backend.Add(result, afterAttn, down);
        afterAttn.Dispose(); down.Dispose();
        return result;
    }

    /// <summary>InternViT LayerScale: scales the sublayer output by a learned per-channel vector (<c>ls1</c>/<c>ls2</c>,
    /// shape <c>[hidden]</c>) before the residual add. No-op when the tensor is absent (SigLIP/CLIP/LLaVA).</summary>
    private void ApplyLayerScale(IBackend backend, Tensor x, string lsKey)
    {
        Tensor? ls = Wopt(lsKey);
        if (ls is null) return;
        using Tensor scale = ls.Reshape(new TensorShape(1, Hidden));   // [hidden] → [1, hidden] for last-dim broadcast
        backend.AffineBroadcastLastDim(x, x, scale, null);
    }

    /// <summary>Quick-GELU (CLIP): <c>x · sigmoid(1.702 x)</c>, composed from Scale + Sigmoid + Mul.</summary>
    private static void QuickGelu(IBackend backend, Tensor outp, Tensor inp)
    {
        backend.Scale(outp, inp, 1.702f);
        backend.Sigmoid(outp, outp);
        backend.Mul(outp, inp, outp);
    }

    /// <summary>Gemma-3 projector: average-pool the patch grid (pool×pool) → RMSNorm → linear into the text hidden size.</summary>
    private Tensor ProjectGemma3(IBackend backend, Tensor patches)
    {
        int hidden = Hidden, grid = PatchGrid, pool = ScaleFactor, outGrid = grid / pool, nTok = outGrid * outGrid;

        // Average-pool host-side over pool×pool grid cells (one-time, off the decode hot path).
        Tensor pooled = new(new TensorShape(1, nTok, hidden), DType.F32);
        float* src = (float*)patches.DataPointer;   // [1, grid*grid, hidden] (D2H sync on CUDA)
        float* dst = (float*)pooled.DataPointer;
        float inv = 1f / (pool * pool);
        for (int oy = 0; oy < outGrid; oy++)
            for (int ox = 0; ox < outGrid; ox++)
            {
                long outBase = ((long)oy * outGrid + ox) * hidden;
                for (int c = 0; c < hidden; c++) dst[outBase + c] = 0f;
                for (int py = 0; py < pool; py++)
                    for (int px = 0; px < pool; px++)
                    {
                        long inBase = ((long)(oy * pool + py) * grid + (ox * pool + px)) * hidden;
                        for (int c = 0; c < hidden; c++) dst[outBase + c] += src[inBase + c];
                    }
                for (int c = 0; c < hidden; c++) dst[outBase + c] *= inv;
            }

        // Gemma-3's soft-embedding RMSNorm; its (1 + w) offset is already baked into the GGUF weight (mean ≈ 1).
        Tensor normed = new(new TensorShape(1, nTok, hidden), DType.F32);
        backend.RmsNorm(normed, pooled, W("mm.soft_emb_norm.weight"), LayerNormEps);
        pooled.Dispose();
        Tensor img = new(new TensorShape(1, nTok, ProjectionDim), DType.F32);
        backend.Linear(img, normed, W("mm.input_projection.weight"), null);
        normed.Dispose();
        return img;
    }

    /// <summary>Idefics3 / SmolVLM projector: pixel-shuffle the patch grid (fold s×s spatial neighbours into the
    /// channel dim, reducing the token count by s²) → Linear (<c>mm.model.fc</c>) into the text hidden size.</summary>
    private Tensor ProjectIdefics3(IBackend backend, Tensor patches)
    {
        int hidden = Hidden, grid = PatchGrid, s = ScaleFactor, outGrid = grid / s, nTok = outGrid * outGrid, shuf = hidden * s * s;

        // Pixel-shuffle host-side: out[oh*outGrid+ow][(sh*s+sb)*hidden + e] = patches[(s*oh+sh)*grid + (s*ow+sb)][e].
        Tensor folded = new(new TensorShape(1, nTok, shuf), DType.F32);
        float* src = (float*)patches.DataPointer;   // [1, grid*grid, hidden] (D2H sync on CUDA)
        float* dst = (float*)folded.DataPointer;
        for (int oh = 0; oh < outGrid; oh++)
            for (int ow = 0; ow < outGrid; ow++)
            {
                long outBase = ((long)oh * outGrid + ow) * shuf;
                for (int sh = 0; sh < s; sh++)
                    for (int sb = 0; sb < s; sb++)
                    {
                        long inBase = ((long)(s * oh + sh) * grid + (s * ow + sb)) * hidden;
                        long off = outBase + (long)(sh * s + sb) * hidden;
                        for (int e = 0; e < hidden; e++) dst[off + e] = src[inBase + e];
                    }
            }

        Tensor img = new(new TensorShape(1, nTok, ProjectionDim), DType.F32);
        backend.Linear(img, folded, W("mm.model.fc.weight"), null);
        folded.Dispose();
        return img;
    }

    /// <summary>InternVL projector: drop the CLS token, pixel-shuffle the patch grid (fold s×s neighbours into the
    /// channel dim — same mapping as Idefics3, reducing tokens by s²) → LayerNorm (<c>mm.model.mlp.0</c>) →
    /// Linear (<c>mm.model.mlp.1</c>) → GELU → Linear (<c>mm.model.mlp.3</c>) into the text hidden size.</summary>
    private Tensor ProjectInternVL(IBackend backend, Tensor h, int cls)
    {
        int hidden = Hidden, grid = PatchGrid, s = ScaleFactor, outGrid = grid / s, nTok = outGrid * outGrid, shuf = hidden * s * s;

        // Pixel-shuffle host-side over the patch tokens (skipping the leading CLS row). Identical fold to Idefics3:
        // out[oh*outGrid+ow][(sh*s+sb)*hidden + e] = patch[(s*oh+sh)*grid + (s*ow+sb)][e].
        Tensor folded = new(new TensorShape(1, nTok, shuf), DType.F32);
        float* src = (float*)h.DataPointer;   // [1, seqLen, hidden] (D2H sync on CUDA); patches start at row `cls`
        float* dst = (float*)folded.DataPointer;
        for (int oh = 0; oh < outGrid; oh++)
            for (int ow = 0; ow < outGrid; ow++)
            {
                long outBase = ((long)oh * outGrid + ow) * shuf;
                for (int sh = 0; sh < s; sh++)
                    for (int sb = 0; sb < s; sb++)
                    {
                        long inPatch = (long)(s * oh + sh) * grid + (s * ow + sb);
                        long inBase = (cls + inPatch) * hidden;
                        long off = outBase + (long)(sh * s + sb) * hidden;
                        for (int e = 0; e < hidden; e++) dst[off + e] = src[inBase + e];
                    }
            }

        Tensor normed = new(new TensorShape(1, nTok, shuf), DType.F32);
        backend.LayerNorm(normed, folded, W("mm.model.mlp.0.weight"), W("mm.model.mlp.0.bias"), 1e-5f);
        folded.Dispose();
        Tensor h1 = new(new TensorShape(1, nTok, ProjectionDim), DType.F32);
        backend.Linear(h1, normed, W("mm.model.mlp.1.weight"), Wopt("mm.model.mlp.1.bias"));
        normed.Dispose();
        Tensor act = new(new TensorShape(1, nTok, ProjectionDim), DType.F32);
        backend.Gelu(act, h1);
        h1.Dispose();
        Tensor img = new(new TensorShape(1, nTok, ProjectionDim), DType.F32);
        backend.Linear(img, act, W("mm.model.mlp.3.weight"), Wopt("mm.model.mlp.3.bias"));
        act.Dispose();
        return img;
    }

    /// <summary>MiniCPM-V perceiver resampler: <c>query_num</c> learnable queries cross-attend the vision features.
    /// v = ln_kv(kv_proj(features)); q = ln_q(query); K = v + 2D-sincos-pos, V = v; MHA(d_head=128) → out_proj →
    /// ln_post → proj into the text hidden size. Returns <c>[1, query_num, projDim]</c>.</summary>
    private Tensor ProjectMiniCpmV(IBackend backend, Tensor h)
    {
        int np = NumPatches, grid = PatchGrid, dim = ProjectionDim, nq = QueryNum;
        int dHead = 128, nHead = dim / dHead;
        TensorShape flatNp = new(1, np, dim), flatQ = new(1, nq, dim);

        // v = ln_kv(kv_proj(features)) [1, np, dim]; kv/attn projections have no relabel bias for kv.
        Tensor v = new(flatNp, DType.F32);
        backend.Linear(v, h, W("resampler.kv.weight"), null);
        Tensor vln = new(flatNp, DType.F32);
        backend.LayerNorm(vln, v, W("resampler.ln_kv.weight"), W("resampler.ln_kv.bias"), LayerNormEps);
        v.Dispose();

        // q = ln_q(query) [1, nq, dim]; the query weight is [nq, dim] rows (ggml ne = [dim, nq]).
        using Tensor query = W("resampler.query").Reshape(flatQ);
        Tensor qln = new(flatQ, DType.F32);
        backend.LayerNorm(qln, query, W("resampler.ln_q.weight"), W("resampler.ln_q.bias"), LayerNormEps);

        // K = v + 2D-sincos position embed (host-built), V = v.
        Tensor kIn = new(flatNp, DType.F32);
        using (Tensor pos = BuildResamplerPos(np, grid, dim)) backend.Add(kIn, vln, pos);

        // Project to heads and cross-attend (queries over the np keys).
        Tensor Q = new(new TensorShape(1, nq, nHead, dHead), DType.F32);
        Tensor K = new(new TensorShape(1, np, nHead, dHead), DType.F32);
        Tensor V = new(new TensorShape(1, np, nHead, dHead), DType.F32);
        backend.Linear(Q, qln, W("resampler.attn.q.weight"), W("resampler.attn.q.bias"));
        backend.Linear(K, kIn, W("resampler.attn.k.weight"), W("resampler.attn.k.bias"));
        backend.Linear(V, vln, W("resampler.attn.v.weight"), W("resampler.attn.v.bias"));
        qln.Dispose(); kIn.Dispose(); vln.Dispose();

        Tensor qM = new(new TensorShape(1, nHead, nq, dHead), DType.F32);
        Tensor kM = new(new TensorShape(1, nHead, np, dHead), DType.F32);
        Tensor vM = new(new TensorShape(1, nHead, np, dHead), DType.F32);
        backend.Permute0213(qM, Q, nq, nHead, dHead);
        backend.Permute0213(kM, K, np, nHead, dHead);
        backend.Permute0213(vM, V, np, nHead, dHead);
        Q.Dispose(); K.Dispose(); V.Dispose();
        Tensor attn = new(new TensorShape(1, nHead, nq, dHead), DType.F32);
        backend.FlashAttention(attn, qM, kM, vM, np, 1, causal: false, qOffset: 0, 1f / MathF.Sqrt(dHead));
        qM.Dispose(); kM.Dispose(); vM.Dispose();
        Tensor attnFlat = new(flatQ, DType.F32);
        backend.Permute0213(attnFlat, attn, nHead, nq, dHead);
        attn.Dispose();
        Tensor outp = new(flatQ, DType.F32);
        backend.Linear(outp, attnFlat, W("resampler.attn.out.weight"), W("resampler.attn.out.bias"));
        attnFlat.Dispose();

        Tensor lnp = new(flatQ, DType.F32);
        backend.LayerNorm(lnp, outp, W("resampler.ln_post.weight"), W("resampler.ln_post.bias"), LayerNormEps);
        outp.Dispose();
        Tensor img = new(new TensorShape(1, nq, ProjectionDim), DType.F32);
        backend.Linear(img, lnp, W("resampler.proj.weight"), null);
        lnp.Dispose();
        return img;
    }

    /// <summary>2D sinusoidal position embedding for the resampler keys: for patch p at grid (row = p/grid,
    /// col = p%grid), <c>[sin(col·ω), cos(col·ω), sin(row·ω), cos(row·ω)]</c> with <c>ω_k = base^(−k/(dim/4))</c>.</summary>
    private static Tensor BuildResamplerPos(int np, int grid, int dim)
    {
        int quarter = dim / 4;
        float[] omega = new float[quarter];
        for (int k = 0; k < quarter; k++) omega[k] = 1f / MathF.Pow(10000f, (float)k / quarter);
        Tensor pos = new(new TensorShape(1, np, dim), DType.F32);
        float* p = (float*)pos.DataPointer;
        for (int patch = 0; patch < np; patch++)
        {
            float row = patch / grid, col = patch % grid;
            long b = (long)patch * dim;
            for (int k = 0; k < quarter; k++)
            {
                p[b + k] = MathF.Sin(col * omega[k]);
                p[b + quarter + k] = MathF.Cos(col * omega[k]);
                p[b + 2 * quarter + k] = MathF.Sin(row * omega[k]);
                p[b + 3 * quarter + k] = MathF.Cos(row * omega[k]);
            }
        }
        return pos;
    }

    /// <summary>LLaVA projector: drop the leading CLS token, then a 2-layer MLP (<c>mm.0</c> → GELU → <c>mm.2</c>)
    /// over the patch tokens into the text hidden size.</summary>
    private Tensor ProjectLlava(IBackend backend, Tensor h, int cls)
    {
        int hidden = Hidden, np = NumPatches;
        // Drop the CLS token: take patch rows [cls .. cls+np).
        Tensor patches = new(new TensorShape(1, np, hidden), DType.F32);
        backend.Sync();
        Buffer.MemoryCopy((byte*)h.DataPointer + (long)cls * hidden * 4, (void*)patches.DataPointer, (long)np * hidden * 4, (long)np * hidden * 4);

        Tensor mid = new(new TensorShape(1, np, ProjectionDim), DType.F32);
        backend.Linear(mid, patches, W("mm.0.weight"), W("mm.0.bias"));
        patches.Dispose();
        Tensor act = new(new TensorShape(1, np, ProjectionDim), DType.F32);
        backend.Gelu(act, mid);
        mid.Dispose();
        Tensor img = new(new TensorShape(1, np, ProjectionDim), DType.F32);
        backend.Linear(img, act, W("mm.2.weight"), W("mm.2.bias"));
        act.Dispose();
        return img;
    }

    private void AddPositions(Tensor seq)
    {
        // v.position_embd.weight is [hidden, nPos] (GGUF) → position p's vector is the p-th row of [nPos, hidden].
        float* sp = (float*)seq.DataPointer;
        float* pp = (float*)W("v.position_embd.weight").DataPointer;
        if (Projector == VlmProjectorKind.MiniCpmV)
        {
            // MiniCPM-V buckets the gridxgrid patches into the trained nPerSide×nPerSide (70×70) position table:
            // patch (i,j) → table row floor(nPerSide·i/grid)·nPerSide + floor(nPerSide·j/grid). (No CLS.)
            int grid = PatchGrid, nPos = (int)(W("v.position_embd.weight").ElementCount / Hidden);
            int nPerSide = (int)MathF.Round(MathF.Sqrt(nPos));
            for (int i = 0; i < grid; i++)
                for (int j = 0; j < grid; j++)
                {
                    int bh = (int)MathF.Floor((float)nPerSide * i / grid);
                    int bw = (int)MathF.Floor((float)nPerSide * j / grid);
                    long dst = ((long)i * grid + j) * Hidden, srcRow = ((long)bh * nPerSide + bw) * Hidden;
                    for (int c = 0; c < Hidden; c++) sp[dst + c] += pp[srcRow + c];
                }
            return;
        }
        long total = (long)SeqLen * Hidden;
        for (long i = 0; i < total; i++) sp[i] += pp[i];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _handle.Dispose();
    }
}
