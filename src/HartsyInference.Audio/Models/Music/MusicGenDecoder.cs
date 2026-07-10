using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Music;

/// <summary>MusicGen causal decoder: K codebook embeddings (summed) + sinusoidal positions → a stack of
/// pre-norm blocks (causal self-attention + cross-attention to the T5 text states + GELU MLP) → K parallel
/// output heads. Reuses the WhisperOps attention helpers (`ProjectLinear`, multi-head reshape, SDPA).
/// AR decoding is incremental: <see cref="CreateCache"/> pre-projects the cross-attn K/V once, then one
/// cache-capturing prefill <see cref="Forward"/> plus O(T) per-frame <see cref="ForwardStep"/> calls against
/// the <see cref="MusicGenKvCache"/>. Cross-attn states are the caller-projected T5 features
/// <c>[1, T_text, hidden]</c>.</summary>
public sealed unsafe class MusicGenDecoder : IDisposable
{
    private readonly MusicGenConfig _cfg;
    private readonly MusicGenBlock[] _blocks;
    private int _disposed;

    private Tensor?[] _codebookEmbed;   // K × [codebookSize+1, hidden]
    private Tensor? _encToDecW;         // textDim → hidden (project T5 states once)
    private Tensor? _encToDecB;         // enc_to_dec_proj bias [hidden] (HF nn.Linear default bias=True)
    private Tensor? _lnOutG, _lnOutB;
    private Tensor?[] _heads;           // K × [codebookSize, hidden]

    public MusicGenConfig Config => _cfg;

    public MusicGenDecoder(MusicGenConfig cfg)
    {
        _cfg = cfg;
        _blocks = new MusicGenBlock[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++) _blocks[i] = new MusicGenBlock(cfg);
        _codebookEmbed = new Tensor?[cfg.NumCodebooks];
        _heads = new Tensor?[cfg.NumCodebooks];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model.decoder")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        // Codebook embeddings are read by host pointer-math (EmbedFrame/EmbedFrames index into DataPointer), so they
        // MUST be F32. enc_to_dec_proj + the K lm_heads are consumed only via backend.Linear (ProjectText/HeadLogits),
        // which casts a non-F32 weight on the GPU — keep them in native (fp16) dtype to save host RAM. Output norm +
        // enc_to_dec bias stay F32 (tiny / precision-sensitive).
        for (int i = 0; i < _cfg.NumCodebooks; i++)
            _codebookEmbed[i] = WhisperOps.EnsureF32(w[$"{p}embed_tokens.{i}.weight"]);
        _encToDecW = w["enc_to_dec_proj.weight"];
        _encToDecB = w.TryGetValue("enc_to_dec_proj.bias", out Tensor? edb) ? WhisperOps.EnsureF32(edb) : null;
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"{p}layers.{i}");
        _lnOutG = WhisperOps.EnsureF32(w[$"{p}layer_norm.weight"]);
        _lnOutB = WhisperOps.EnsureF32(w[$"{p}layer_norm.bias"]);
        for (int i = 0; i < _cfg.NumCodebooks; i++)
            _heads[i] = w[$"lm_heads.{i}.weight"];
    }

    /// <summary>Projects raw T5 states <c>[1, T_text, textDim]</c> to the decoder's cross-attn space
    /// <c>[1, T_text, hidden]</c> (the once-per-generation <c>enc_to_dec_proj</c>).</summary>
    public Tensor ProjectText(IBackend backend, Tensor t5States)
    {
        int tt = (int)t5States.Shape[1];
        return WhisperOps.ProjectLinear(backend, t5States, _encToDecW!, _encToDecB, 1, tt, _cfg.TextDim, _cfg.Hidden);
    }

    /// <summary>Embeds a sequence of K-codebook frames (sum of the per-codebook embeddings) + sinusoidal
    /// positions → <c>[1, T, hidden]</c>. <paramref name="frames"/> is <c>[T, K]</c>.</summary>
    public Tensor EmbedFrames(int[,] frames)
    {
        int t = frames.GetLength(0);
        int h = _cfg.Hidden;
        Tensor outT = new(new TensorShape(1, t, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int s = 0; s < t; s++)
        {
            long row = (long)s * h;
            for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
            {
                int id = Math.Clamp(frames[s, cb], 0, _cfg.CodebookSize);
                float* tab = (float*)_codebookEmbed[cb]!.DataPointer + (long)id * h;
                for (int c = 0; c < h; c++) op[row + c] += tab[c];
            }
            AddSinusoid(op + row, s, h);
        }
        return outT;
    }

    /// <summary>Runs the decoder stack and returns the K next-step logits <c>[K][codebookSize]</c> from
    /// the last position. <paramref name="cross"/> is the projected T5 states. Pass <paramref name="cache"/>
    /// to capture every layer's self-attn K/V for the prompt (prefill), enabling incremental continuation
    /// via <see cref="ForwardStep"/>.</summary>
    public float[][] Forward(IBackend backend, Tensor inputEmbeds, Tensor cross, MusicGenKvCache? cache = null)
    {
        int t = (int)inputEmbeds.Shape[1];
        int h = _cfg.Hidden;
        if (cache is not null && t > cache.Capacity)
        {
            throw new ArgumentException($"Prompt length {t} exceeds KV cache capacity {cache.Capacity}.");
        }
        Tensor? mask = t > 1 ? BuildCausalMask(t) : null;
        Tensor hidden = inputEmbeds;
        bool owns = false;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, hidden, cross, mask, cache?.Layers[i]);
            if (owns) hidden.Dispose();
            hidden = next; owns = true;
        }
        mask?.Dispose();
        if (cache is not null)
        {
            cache.Length = t;
        }

        Tensor normed = new(hidden.Shape, DType.F32);
        backend.LayerNorm(normed, hidden, _lnOutG!, _lnOutB!, 1e-5f);
        if (owns) hidden.Dispose();

        Tensor last = SliceLast(normed, h);
        normed.Dispose();
        float[][] logits = HeadLogits(backend, last);
        last.Dispose();
        return logits;
    }

    /// <summary>Creates a K/V cache sized to <paramref name="capacity"/> positions, pre-projecting each
    /// layer's cross-attn K/V from <paramref name="cross"/> once (they depend only on the text states,
    /// never per step). For use with <see cref="Forward"/> (prefill) + <see cref="ForwardStep"/>.</summary>
    public MusicGenKvCache CreateCache(IBackend backend, Tensor cross, int capacity)
    {
        int b = (int)cross.Shape[0];   // 1 (no CFG) or 2 (CFG: [0]=cond text, [1]=null/zeros)
        MusicGenKvCache cache = new(b, _cfg.NumLayers, _cfg.NumHeads, capacity, _cfg.HeadDim);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].PrimeCross(backend, cross, cache.Layers[i]);
        cache.CrossLength = (int)cross.Shape[1];
        if (backend.GraphDecodeSupported)
        {
            cache.DevicePos = backend.AllocDevicePos();   // {kvLen, qOffset}, refreshed each step
            cache.PosBackend = backend;
        }
        return cache;
    }

    /// <summary>Incremental decode: embeds one K-codebook <paramref name="frame"/> at position
    /// <see cref="MusicGenKvCache.Length"/>, steps the stack against the cache (appending its K/V there),
    /// and returns the K next-step logits. Equivalent to the last position of a full causal
    /// <see cref="Forward"/> over the cached prefix plus this frame, at O(T) instead of O(T²).
    /// <paramref name="advance"/>=false leaves <c>Length</c> unchanged so the same row can be rewritten —
    /// probe a provisional frame for logits, then commit the sampled one over it.</summary>
    public float[][] ForwardStep(IBackend backend, int[] frame, MusicGenKvCache cache, bool advance = true)
    {
        int pos = cache.Length;
        if (pos >= cache.Capacity)
        {
            throw new InvalidOperationException($"KV cache is full ({cache.Capacity}); size it to the generation length.");
        }

        // Refresh the device position {kvLen, qOffset} = {pos+1, pos} so the self-attn KV-append/flash kernels read
        // it from device memory (graph-replayable). Written outside any capture region (this call precedes the block
        // ops). No-op when the backend doesn't support device-position decode (DevicePos stays 0 → host-int path).
        if (cache.DevicePos != 0) backend.WriteDevicePos(cache.DevicePos, pos + 1, pos);
        Tensor hidden = EmbedFrame(frame, pos);
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].ForwardStep(backend, hidden, cache.Layers[i], pos, cache.CrossLength, cache.DevicePos);
            hidden.Dispose();
            hidden = next;
        }
        if (advance) cache.Length = pos + 1;

        Tensor normed = new(hidden.Shape, DType.F32);
        backend.LayerNorm(normed, hidden, _lnOutG!, _lnOutB!, 1e-5f);
        hidden.Dispose();
        float[][] logits = HeadLogits(backend, normed);
        normed.Dispose();
        return logits;
    }

    // ── Graph-decode fixed buffers (capture boundary), batched over CFG ──────────────────────────────────────
    // The cond + uncond streams are decoded as ONE batch=B pass (B=2 with CFG, 1 without), halving the kernel count
    // vs two sequential passes and widening every GEMM. A captured graph reads/writes ONLY fixed-address device
    // buffers: the token embedding is refreshed into `_embedIn` [B,1,hidden] via CopyInto OUTSIDE capture; the K
    // head logits land in `_logitsFixed[cb]` [B,1,codebookSize] via CopyInto INSIDE the captured body; both batch
    // rows are read back (non-freeing DtoH) after the launch — row 0 = conditional, row 1 = unconditional.
    private int _graphBatch;
    private Tensor? _embedIn;
    private Tensor?[]? _logitsFixed;

    /// <summary>Allocates the fixed embed-in + K logits-out buffers for a batch of <paramref name="batch"/> streams.</summary>
    public void EnsureGraphBuffers(int batch)
    {
        if (_graphBatch == batch && _embedIn is not null) return;
        ReleaseGraphBuffers();
        _graphBatch = batch;
        _embedIn = new Tensor(new TensorShape(batch, 1, _cfg.Hidden), DType.F32);
        _logitsFixed = new Tensor?[_cfg.NumCodebooks];
        for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
            _logitsFixed[cb] = new Tensor(new TensorShape(batch, 1, _cfg.CodebookSize), DType.F32);
    }

    /// <summary>Refreshes the fixed batched embedding buffer for one step (host build + CopyInto). Call OUTSIDE
    /// capture. Both batch rows get the SAME embedding (cond and uncond feed the same sampled token).</summary>
    public void PrepareEmbed(IBackend backend, int[] frame, int pos, int batch)
    {
        Tensor e = EmbedFrameBatched(frame, pos, batch);
        backend.CopyInto(_embedIn!, e);
        e.Dispose();
    }

    /// <summary>The CAPTURABLE batched body: reads the fixed embedding, runs the block stack against the batched
    /// <paramref name="cache"/> (self-attn position from <c>cache.DevicePos</c>), and lands the K head logits in
    /// the fixed buffers via CopyInto — no host reads, no fresh fixed-address allocations.</summary>
    public void RunBatchedIntoFixed(IBackend backend, MusicGenKvCache cache)
    {
        int b = cache.Batch;
        int pos = cache.Length;   // ignored by the device-position kernels (they read cache.DevicePos)
        Tensor hidden = _embedIn!;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].ForwardStep(backend, hidden, cache.Layers[i], pos, cache.CrossLength, cache.DevicePos);
            if (!ReferenceEquals(hidden, _embedIn)) hidden.Dispose();
            hidden = next;
        }
        Tensor normed = new(hidden.Shape, DType.F32);
        backend.LayerNorm(normed, hidden, _lnOutG!, _lnOutB!, 1e-5f);
        if (!ReferenceEquals(hidden, _embedIn)) hidden.Dispose();
        for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
        {
            Tensor l = WhisperOps.ProjectLinear(backend, normed, _heads[cb]!, bias: null, b, 1, _cfg.Hidden, _cfg.CodebookSize);
            backend.CopyInto(_logitsFixed![cb]!, l);
            l.Dispose();
        }
        normed.Dispose();
    }

    /// <summary>Reads back the batched K logits after a launch (non-freeing DtoH, so the fixed buffers keep their
    /// baked addresses). Returns <c>[batch][K][codebookSize]</c>; index 0 = conditional stream, 1 = unconditional.</summary>
    public float[][][] ReadBatchedLogits(IBackend backend, int batch)
    {
        int cs = _cfg.CodebookSize;
        float[][][] outp = new float[batch][][];
        for (int s = 0; s < batch; s++) outp[s] = new float[_cfg.NumCodebooks][];
        float[] scratch = new float[batch * cs];
        for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
        {
            backend.ReadResidentInto(_logitsFixed![cb]!, scratch);   // [batch, codebookSize] row-major
            for (int s = 0; s < batch; s++)
            {
                float[] row = new float[cs];
                Array.Copy(scratch, s * cs, row, 0, cs);
                outp[s][cb] = row;
            }
        }
        return outp;
    }

    /// <summary>Disposes the graph-decode fixed buffers (call at end of generation, after StepGraphReset).</summary>
    public void ReleaseGraphBuffers()
    {
        _embedIn?.Dispose(); _embedIn = null;
        if (_logitsFixed is not null) { foreach (Tensor? t in _logitsFixed) t?.Dispose(); _logitsFixed = null; }
        _graphBatch = 0;
    }

    /// <summary>Batched frame embedding: <c>[batch, 1, hidden]</c> with every row the same (cond+uncond share the
    /// input token). Reuses the single-frame host build then replicates across the batch.</summary>
    private Tensor EmbedFrameBatched(int[] frame, int pos, int batch)
    {
        int h = _cfg.Hidden;
        Tensor outT = new(new TensorShape(batch, 1, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
        {
            int id = Math.Clamp(frame[cb], 0, _cfg.CodebookSize);
            float* tab = (float*)_codebookEmbed[cb]!.DataPointer + (long)id * h;
            for (int c = 0; c < h; c++) op[c] += tab[c];
        }
        AddSinusoid(op, pos, h);
        for (int s = 1; s < batch; s++)
            Buffer.MemoryCopy(op, op + (long)s * h, h * 4, h * 4);   // replicate row 0 to the other streams
        return outT;
    }

    /// <summary>Projects a <c>[1, 1, hidden]</c> state through the K output heads → <c>[K][codebookSize]</c>.</summary>
    private float[][] HeadLogits(IBackend backend, Tensor last)
    {
        float[][] logits = new float[_cfg.NumCodebooks][];
        for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
        {
            Tensor l = WhisperOps.ProjectLinear(backend, last, _heads[cb]!, bias: null, 1, 1, _cfg.Hidden, _cfg.CodebookSize);
            logits[cb] = new float[_cfg.CodebookSize];
            new Span<float>((void*)l.DataPointer, _cfg.CodebookSize).CopyTo(logits[cb]);
            l.Dispose();
        }
        return logits;
    }

    /// <summary>Embeds one K-codebook frame (summed embeddings + sinusoid at absolute <paramref name="pos"/>) → <c>[1, 1, hidden]</c>.</summary>
    private Tensor EmbedFrame(int[] frame, int pos)
    {
        int h = _cfg.Hidden;
        Tensor outT = new(new TensorShape(1, 1, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
        {
            int id = Math.Clamp(frame[cb], 0, _cfg.CodebookSize);
            float* tab = (float*)_codebookEmbed[cb]!.DataPointer + (long)id * h;
            for (int c = 0; c < h; c++) op[c] += tab[c];
        }
        AddSinusoid(op, pos, h);
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? e in _codebookEmbed) if (e is not null) yield return e;
        if (_encToDecW is not null) yield return _encToDecW;
        if (_encToDecB is not null) yield return _encToDecB;
        foreach (MusicGenBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        if (_lnOutG is not null) yield return _lnOutG;
        if (_lnOutB is not null) yield return _lnOutB;
        foreach (Tensor? hd in _heads) if (hd is not null) yield return hd;
    }

    private static void AddSinusoid(float* row, int pos, int h)
    {
        // Matches HF MusicgenSinusoidalPositionalEmbedding.get_embedding:
        //   emb = exp(arange(half) * -(log(10000)/(half-1)))
        //   pos_emb = cat([cos(pos*emb), sin(pos*emb)])   ← COS first, then SIN.
        int half = h / 2;
        for (int i = 0; i < half; i++)
        {
            double freq = Math.Exp(-Math.Log(10000.0) * i / Math.Max(1, half - 1));
            row[i] += (float)Math.Cos(pos * freq);
            row[half + i] += (float)Math.Sin(pos * freq);
        }
    }

    private static Tensor BuildCausalMask(int t)
    {
        Tensor mask = new(new TensorShape(1, 1, t, t), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int q = 0; q < t; q++)
            for (int k = 0; k < t; k++)
                mp[(long)q * t + k] = k <= q ? 0f : float.NegativeInfinity;
        return mask;
    }

    private static Tensor SliceLast(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        return last;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (MusicGenBlock b in _blocks) b.Dispose();
        GC.SuppressFinalize(this);
    }
}
