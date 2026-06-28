using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Gguf;

namespace HartsyInference.LLM.Multimodal;

/// <summary>The Llama-3.2-Vision (mllama) vision tower + projector, loaded from a <c>mmproj-*.gguf</c>. Unlike the
/// SigLIP/CLIP VLMs, mllama does not splice image tokens into the text sequence; this encoder produces
/// <c>cross_attention_states</c> <c>[numTiles·seqLen, textHidden]</c> that the gated cross-attention decoder
/// layers (<see cref="MllamaCrossAttentionLayer"/>) read.
///
/// <para>Architecture (560px tiles, patch 14 → 1600+1 CLS = 1601 tokens, hidden 1280, 16 heads, 32 local + 8 gated
/// global blocks): patch Conv → +pre-tile aspect embed → +CLS → +gated (base position + tile position) → pre-LN →
/// 32 local ViT blocks (collecting the intermediate layers <c>[3,7,15,23,30]</c>) → post-LN → +post-tile aspect
/// embed → 8 gated global blocks → concat(global, 5 intermediates) = 6·1280 = 7680 → <c>mm.0</c> Linear → 4096.</para>
///
/// <para><b>Gates are pre-tanh'd in the GGUF</b> (Ollama's converter applies <c>tanh()</c> at conversion, and
/// <c>1-tanh</c> for <c>position_embd.gate</c>), so the forward multiplies by the stored gate value directly — no
/// tanh at inference. The MLP names are clip-swapped (<c>ffn_down</c> is fc1/up, <c>ffn_up</c> is fc2/down). The
/// converter's q/k weight permute is a no-op here (no RoPE in the vision tower), so q/k load as-is. Single-tile
/// (aspect-ratio (1,1), id 1) covers square inputs and the synthetic harness; HF's multiple-of-8 padding is a
/// masked-out no-op for the real tokens and is skipped.</para></summary>
public sealed unsafe class MllamaVisionEncoder : IDisposable
{
    public int Hidden { get; }
    public int NumLocalLayers { get; }
    public int NumGlobalLayers { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int Intermediate { get; }
    public int PatchSize { get; }
    public int ImageSize { get; }
    public int PatchGrid { get; }
    public int SeqLen { get; }          // patches + CLS
    public int ProjectionDim { get; }   // text hidden
    public float LayerNormEps { get; }
    public int[] IntermediateLayers { get; }

    private const int AspectId = 1;     // (1,1) single-tile aspect-ratio id

    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    private int _disposed;

    private Tensor W(string k) => _w[k];

    private MllamaVisionEncoder(GgufModelLoader.LoadedGgufModel handle, IReadOnlyDictionary<string, Tensor> w,
        int hidden, int local, int global, int heads, int inter, int patch, int image, int projDim, float eps, int[] interIdx)
    {
        _handle = handle; _w = w;
        Hidden = hidden; NumLocalLayers = local; NumGlobalLayers = global; NumHeads = heads; HeadDim = hidden / heads;
        Intermediate = inter; PatchSize = patch; ImageSize = image; PatchGrid = image / patch;
        SeqLen = PatchGrid * PatchGrid + 1; ProjectionDim = projDim; LayerNormEps = eps <= 0 ? 1e-5f : eps;
        IntermediateLayers = interIdx;
    }

    /// <summary>Relabels a 2D GGUF weight from engine shape <c>[in, out]</c> to <c>[out, in]</c> for
    /// <see cref="IBackend.Linear"/> (data is already row-major <c>[out, in]</c> — shape relabel only).</summary>
    private static Tensor Relabel(Tensor t)
    {
        Tensor o = new(new TensorShape((int)t.Shape[1], (int)t.Shape[0]), DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)o.DataPointer, o.ElementCount * 4, t.ElementCount * 4);
        return o;
    }

    public static MllamaVisionEncoder Load(string mmprojPath)
    {
        (Dictionary<string, Tensor> w, GgufModelLoader.LoadedGgufModel handle) = GgufModelLoader.LoadDequantized(mmprojPath, DType.F32);
        try
        {
            // nn.Linear weights → [out, in]. Position/tile tables, conv, class embed, gates and biases stay as-is.
            foreach (string key in w.Keys.ToList())
            {
                if (!key.EndsWith(".weight", StringComparison.Ordinal) || w[key].Shape.Rank != 2) continue;
                if (key.Contains("position_embd") || key.Contains("patch_embd")) continue;
                if (key.Contains("attn_q") || key.Contains("attn_k") || key.Contains("attn_v") || key.Contains("attn_out")
                    || key.Contains("ffn_up") || key.Contains("ffn_down") || key == "mm.0.weight")
                    w[key] = Relabel(w[key]);
            }
            GgufMetadata m = handle.Metadata;
            int hidden = (int)m.GetUInt32("mllama.vision.embedding_length");
            int local = (int)m.GetUInt32("mllama.vision.block_count");
            int global = (int)m.GetUInt32("mllama.vision.global.block_count");
            int heads = (int)m.GetUInt32("mllama.vision.attention.head_count");
            int inter = (int)m.GetUInt32("mllama.vision.feed_forward_length");
            int patch = (int)m.GetUInt32("mllama.vision.patch_size");
            int image = (int)m.GetUInt32("mllama.vision.image_size");
            int projDim = (int)m.GetUInt32("mllama.vision.projection_dim");
            float eps = m.GetFloat32("mllama.vision.attention.layer_norm_epsilon", 1e-5f);
            // intermediate_layers_indices = [3,7,15,23,30] for 11B (fixed by the architecture).
            int[] interIdx = [3, 7, 15, 23, 30];
            return new MllamaVisionEncoder(handle, w, hidden, local, global, heads, inter, patch, image, projDim, eps, interIdx);
        }
        catch { foreach (Tensor t in w.Values) t.Dispose(); handle.Dispose(); throw; }
    }

    /// <summary>Encodes a preprocessed image <c>[1, 3, 560, 560]</c> into cross-attention states
    /// <c>[1, SeqLen, ProjectionDim]</c> (single tile).</summary>
    public Tensor Encode(IBackend backend, Tensor pixelValues)
    {
        int hidden = Hidden, grid = PatchGrid, np = grid * grid, seq = SeqLen, heads = NumHeads, d = HeadDim;

        // Patch embed: Conv2D (kernel=stride=patch, no bias). GGUF [kw,kh,in,out] memory == [out,in,kh,kw].
        Tensor patchW = W("v.patch_embd.weight").Reshape(new TensorShape(hidden, 3, PatchSize, PatchSize));
        Tensor conv = new(new TensorShape(1, hidden, grid, grid), DType.F32);
        backend.Conv2D(conv, pixelValues, patchW, null, PatchSize, PatchSize, 0, 0);
        Tensor convFlat = conv.Reshape(new TensorShape(1, hidden, np));

        // [1,hidden,np] → patches [1,np,hidden]; + pre-tile aspect embed; prepend CLS → [1,seq,hidden].
        Tensor seqT = new(new TensorShape(1, seq, hidden), DType.F32);
        using (Tensor patches = new(new TensorShape(1, np, hidden), DType.F32))
        {
            backend.Transpose2D(patches, convFlat, hidden, np);
            backend.Sync();
            float preGate = Scalar("v.pre_tile_position_embd.gate");
            float* pre = TileVec("v.pre_tile_position_embd.weight", AspectId, 0, hidden);  // [hidden]
            float* pp = (float*)patches.DataPointer;
            for (int i = 0; i < np; i++) for (int j = 0; j < hidden; j++) pp[(long)i * hidden + j] += preGate * pre[j];
            // CLS at row 0, patches after.
            float* sp = (float*)seqT.DataPointer;
            Buffer.MemoryCopy((void*)W("v.class_embd").DataPointer, sp, (long)hidden * 4, (long)hidden * 4);
            Buffer.MemoryCopy(pp, sp + hidden, (long)np * hidden * 4, (long)np * hidden * 4);
        }
        conv.Dispose();

        // Gated positional embedding: + position_gate*base + tile_gate*tile[aspect][tile0]. (gates pre-tanh'd)
        AddGatedPositions(seqT, seq, hidden);

        // pre-LN
        Tensor h = new(new TensorShape(1, seq, hidden), DType.F32);
        backend.LayerNorm(h, seqT, W("v.pre_ln.weight"), W("v.pre_ln.bias"), LayerNormEps);
        seqT.Dispose();
        Dump("seq", h);

        // Local transformer, collecting intermediate hidden states. hs[0]=input, hs[k]=output of layer k-1.
        List<Tensor> intermediates = [];
        HashSet<int> wanted = [.. IntermediateLayers];
        if (wanted.Contains(0)) intermediates.Add(Clone(h));
        for (int i = 0; i < NumLocalLayers; i++)
        {
            Tensor next = Block(backend, h, $"v.blk.{i}", seq, gated: false);
            h.Dispose(); h = next;
            if (i == 0) Dump("blk0", h);
            if (wanted.Contains(i + 1)) intermediates.Add(Clone(h));
        }
        // post-LN
        Tensor postln = new(new TensorShape(1, seq, hidden), DType.F32);
        backend.LayerNorm(postln, h, W("v.post_ln.weight"), W("v.post_ln.bias"), LayerNormEps);
        h.Dispose(); h = postln;
        Dump("postln", h);

        // + post-tile aspect embed (gated)
        AddTileVec(h, "v.post_tile_position_embd.weight", "v.post_tile_position_embd.gate", seq, hidden);

        // Global transformer (gated)
        for (int i = 0; i < NumGlobalLayers; i++)
        {
            Tensor next = Block(backend, h, $"v.global.blk.{i}", seq, gated: true);
            h.Dispose(); h = next;
        }

        // Concat [global, intermediate_0..4] → [1, seq, 6*hidden], then mm.0 projector.
        int cdim = hidden * (1 + intermediates.Count);
        Tensor cat = new(new TensorShape(1, seq, cdim), DType.F32);
        ConcatFeatures(cat, h, intermediates, seq, hidden);
        h.Dispose(); foreach (Tensor t in intermediates) t.Dispose();
        Dump("cat", cat);

        Tensor embeds = new(new TensorShape(1, seq, ProjectionDim), DType.F32);
        backend.Linear(embeds, cat, W("mm.0.weight"), W("mm.0.bias"));
        cat.Dispose();
        Dump("embeds", embeds);
        return embeds;
    }

    /// <summary>One ViT block: LN → MHA(no bias, bidirectional) → [+gate] +res → LN → MLP(gelu, clip-swapped) → [+gate] +res.</summary>
    private Tensor Block(IBackend backend, Tensor x, string p, int seq, bool gated)
    {
        int hidden = Hidden, heads = NumHeads, d = HeadDim;
        TensorShape flat = new(1, seq, hidden);

        Tensor ln1 = new(flat, DType.F32);
        backend.LayerNorm(ln1, x, W($"{p}.ln1.weight"), W($"{p}.ln1.bias"), LayerNormEps);
        Tensor q = new(new TensorShape(1, seq, heads, d), DType.F32);
        Tensor k = new(new TensorShape(1, seq, heads, d), DType.F32);
        Tensor v = new(new TensorShape(1, seq, heads, d), DType.F32);
        backend.Linear(q, ln1, W($"{p}.attn_q.weight"), null);
        backend.Linear(k, ln1, W($"{p}.attn_k.weight"), null);
        backend.Linear(v, ln1, W($"{p}.attn_v.weight"), null);
        ln1.Dispose();
        Tensor qM = new(new TensorShape(1, heads, seq, d), DType.F32); backend.Permute0213(qM, q, seq, heads, d);
        Tensor kM = new(new TensorShape(1, heads, seq, d), DType.F32); backend.Permute0213(kM, k, seq, heads, d);
        Tensor vM = new(new TensorShape(1, heads, seq, d), DType.F32); backend.Permute0213(vM, v, seq, heads, d);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor attn = new(new TensorShape(1, heads, seq, d), DType.F32);
        backend.FlashAttention(attn, qM, kM, vM, seq, 1, causal: false, qOffset: 0, 1f / MathF.Sqrt(d));
        qM.Dispose(); kM.Dispose(); vM.Dispose();
        Tensor attnFlat = new(flat, DType.F32); backend.Permute0213(attnFlat, attn, heads, seq, d); attn.Dispose();
        Tensor attnOut = new(flat, DType.F32); backend.Linear(attnOut, attnFlat, W($"{p}.attn_out.weight"), null); attnFlat.Dispose();
        if (gated) backend.Scale(attnOut, attnOut, Scalar($"{p}.attn_gate"));
        Tensor afterAttn = new(flat, DType.F32); backend.Add(afterAttn, x, attnOut); attnOut.Dispose();

        Tensor ln2 = new(flat, DType.F32);
        backend.LayerNorm(ln2, afterAttn, W($"{p}.ln2.weight"), W($"{p}.ln2.bias"), LayerNormEps);
        // clip swaps the MLP names: ffn_down is fc1 (up), ffn_up is fc2 (down).
        Tensor up = new(new TensorShape(1, seq, Intermediate), DType.F32);
        backend.Linear(up, ln2, W($"{p}.ffn_down.weight"), W($"{p}.ffn_down.bias")); ln2.Dispose();
        Tensor act = new(new TensorShape(1, seq, Intermediate), DType.F32);
        backend.Gelu(act, up); up.Dispose();
        Tensor down = new(flat, DType.F32); backend.Linear(down, act, W($"{p}.ffn_up.weight"), W($"{p}.ffn_up.bias")); act.Dispose();
        if (gated) backend.Scale(down, down, Scalar($"{p}.ffn_gate"));
        Tensor result = new(flat, DType.F32); backend.Add(result, afterAttn, down);
        afterAttn.Dispose(); down.Dispose();
        return result;
    }

    private float Scalar(string k) => *(float*)W(k).DataPointer;

    /// <summary>Pointer to the [hidden] slice for (aspect, tile) of a flat [9, 4, hidden] aspect-tile table.</summary>
    private float* TileVec(string key, int aspect, int tile, int hidden)
        => (float*)W(key).DataPointer + ((long)aspect * 4 + tile) * hidden;

    private void AddGatedPositions(Tensor seqT, int seq, int hidden)
    {
        float posGate = Scalar("v.position_embd.gate"), tileGate = Scalar("v.tile_position_embd.gate");
        float* basePos = (float*)W("v.position_embd.weight").DataPointer;                 // [seq, hidden]
        float* tilePos = (float*)W("v.tile_position_embd.weight").DataPointer + (long)aspectTileOffset(seq, hidden);  // [seq,hidden]
        float* sp = (float*)seqT.DataPointer;
        for (long i = 0; i < (long)seq * hidden; i++) sp[i] += posGate * basePos[i] + tileGate * tilePos[i];
    }
    private long aspectTileOffset(int seq, int hidden) => ((long)AspectId * 4 + 0) * seq * hidden;

    private void AddTileVec(Tensor h, string weightKey, string gateKey, int seq, int hidden)
    {
        float gate = Scalar(gateKey);
        float* vec = TileVec(weightKey, AspectId, 0, hidden);   // [hidden], broadcast over seq
        float* hp = (float*)h.DataPointer;
        for (int i = 0; i < seq; i++) for (int j = 0; j < hidden; j++) hp[(long)i * hidden + j] += gate * vec[j];
    }

    private static Tensor Clone(Tensor t)
    {
        Tensor c = new(t.Shape, DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)c.DataPointer, c.ElementCount * 4, t.ElementCount * 4);
        return c;
    }

    private static void ConcatFeatures(Tensor cat, Tensor global, List<Tensor> inter, int seq, int hidden)
    {
        int parts = 1 + inter.Count, cdim = hidden * parts;
        float* cp = (float*)cat.DataPointer;
        float* gp = (float*)global.DataPointer;
        for (int s = 0; s < seq; s++)
        {
            Buffer.MemoryCopy(gp + (long)s * hidden, cp + (long)s * cdim, (long)hidden * 4, (long)hidden * 4);
            for (int m = 0; m < inter.Count; m++)
            {
                float* ip = (float*)inter[m].DataPointer;
                Buffer.MemoryCopy(ip + (long)s * hidden, cp + (long)s * cdim + (long)(m + 1) * hidden, (long)hidden * 4, (long)hidden * 4);
            }
        }
    }

    private void Dump(string stage, Tensor t)
    {
        string? dir = Environment.GetEnvironmentVariable("HARTSY_MLLAMA_DUMP");
        if (dir is null) return;
        Directory.CreateDirectory(dir);
        using FileStream fs = File.Create(Path.Combine(dir, $"cs_{stage}.f32"));
        using BinaryWriter bw = new(fs);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) bw.Write(p[i]);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _w.Values) t.Dispose();
        _handle.Dispose();
    }
}
