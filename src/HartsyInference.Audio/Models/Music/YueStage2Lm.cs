using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.Music;

/// <summary>YuE Stage-2 residual upsampler — a ~1.5B LLaMA decoder (reuses <see cref="Qwen2Model"/>, same body as Stage-1) that predicts residual codebooks 1..7 from one track's codebook-0 stream, yielding the full 8-codebook grid X-Codec decodes to waveform.</summary>
/// <remarks>Faithful port of
/// upstream <c>infer.py:stage2_generate</c> (m-a-p/YuE, <c>YuE-s2-1B-general</c>).
///
/// <para><b>xcodec token layout</b> (mm_tokenizer_v0.2): codebook k's index i maps to absolute id
/// <c>i + 45334 + k·1024</c>. Stage-2 sees cb0 tokens (k=0, range [45334,46358)) and emits residuals in
/// [46358,53526) (cb1..cb7). Upstream masks everything outside [46358,53526) and greedily decodes 7
/// tokens/frame; the token's slot (1..7) determines which codebook it belongs to (so a token whose value
/// falls in the wrong codebook's sub-range decodes out of [0,1023] and is patched afterward).</para>
///
/// <para><b>Windowing</b>: upstream processes independent 6-second (300-frame) windows, each with a fresh
/// prompt <c>[SOA][&lt;stage_1&gt;] + cb0(window) + [&lt;stage_2&gt;]</c> then per-frame teacher forcing.
/// The trailing partial window (&lt;300 frames) is handled the same way. We reproduce that per-window.</para>
///
/// <para><b>Sampling</b>: greedy argmax (upstream <c>generate</c> passes no temperature/top_p and forces
/// exactly 7 new tokens → HF greedy). No repetition penalty / CFG in Stage-2.</para></remarks>
public sealed unsafe class YueStage2Lm : IDisposable
{
    // xcodec offset math (CodecManipulator("xcodec"): global_offset=45334, codebook_size=1024).
    private const int GlobalOffset = 45_334;
    private const int CodebookSize = 1_024;
    private const int Cb0Lo = GlobalOffset;                          // 45334
    private const int ResidualLo = GlobalOffset + 1 * CodebookSize;  // 46358 (cb1 start)
    private const int ResidualHi = GlobalOffset + 8 * CodebookSize;  // 53526 (cb7 end, exclusive)
    private const int WindowFrames = 300;                           // upstream 6s @ 50 fps

    private readonly YueConfig _cfg;
    private readonly Qwen2Model _lm;
    private readonly int _soa, _stage1Tok, _stage2Tok;
    private int _disposed;

    /// <param name="soaId">mm tokenizer &lt;SOA&gt; (32001).</param>
    /// <param name="stage1Id">mm tokenizer &lt;stage_1&gt; (32013).</param>
    /// <param name="stage2Id">mm tokenizer &lt;stage_2&gt; (32017).</param>
    public YueStage2Lm(YueConfig cfg, int soaId, int stage1Id, int stage2Id)
    {
        _cfg = cfg;
        _lm = new Qwen2Model(cfg.Stage2);
        _soa = soaId;
        _stage1Tok = stage1Id;
        _stage2Tok = stage2Id;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model")
        => _lm.LoadWeights(w, prefix);

    /// <summary>Upsamples one track's codebook-0 stream (<paramref name="cb0Indices"/>, raw indices in [0,1023]) to a full <c>[8][T]</c> codebook grid (row 0 = the input cb0). Runs the LM per 300-frame window; invalid residuals are patched to the per-row mode (upstream's "fix invalid codes").</summary>
    public int[][] Upsample(IBackend backend, ReadOnlySpan<int> cb0Indices)
    {
        ThrowIfDisposed();
        int t = cb0Indices.Length;
        int[][] codes = new int[_cfg.NumCodebooks][];
        for (int k = 0; k < _cfg.NumCodebooks; k++) codes[k] = new int[t];
        for (int i = 0; i < t; i++) codes[0][i] = cb0Indices[i];   // cb0 row is the given stream
        if (t == 0) return codes;

        for (int w0 = 0; w0 < t; w0 += WindowFrames)
        {
            int wlen = Math.Min(WindowFrames, t - w0);
            UpsampleWindow(backend, cb0Indices.Slice(w0, wlen), codes, w0);
        }
        FixInvalidCodes(codes);
        return codes;
    }

    // ── PERF TODO (2026-07-04): Stage-2 is the YuE bottleneck. Stage-1 was fixed to ~40 ms/tok (Q4_K dp4a GEMV,
    // resident) and BEATS the Python reference. Stage-2 here is still ~2.3x off Python's batched rate and dominates a
    // full song (a 30 s clip ≈ 24k forwards ≈ ~40 min — same ballpark as official YuE on a 3060, so this is the last
    // real gap to close, not a bug).
    //
    // Root cause of the gap: the per-frame residual loop below (UpsampleWindowBoth) does, per frame, 8 transformer
    // forwards + 7 lm_head projections, and each residual reads `logitsT.DataPointer` on the HOST to run
    // ArgmaxResidual → that triggers a D2H sync (EnsureCpuData drains the compute stream), then the chosen token is
    // fed back via EmbedLookup(host int[]) → next forward. So there are ~7 GPU→host→GPU round-trips PER FRAME whose
    // latency the batched GEMM can't hide. Measured actual ≈137 ms/op vs ~4 s total pure compute — the rest is this
    // serialization. Batching the 2 tracks (this method) only removed the halving-able projection work (~18%); the
    // sync latency is untouched.
    //
    // To close it (pick up later — needs GPU test cycles + nsys to confirm):
    //   1. On-GPU argmax over [ResidualLo,ResidualHi) → a device-resident token id (add an IBackend ArgmaxRange op /
    //      reuse any existing GPU argmax). Avoids the full-logits D2H.
    //   2. Device-index embedding feed: a GenericTransformer/Qwen2Model EmbedLookup variant that gathers rows using a
    //      DEVICE index tensor (not host int[]), so the chosen token never leaves the GPU. Then the whole 7-residual
    //      chain stays on-device and only ONE D2H per frame (or per window) is needed to read the emitted codes.
    //   3. With (1)+(2), the per-frame decode becomes a fixed launch sequence → CUDA-graph capture/replay (the user's
    //      suggestion) can then collapse launch overhead too.
    // Expected: Stage-2 ~565 s → ~350 s for a 5 s clip (match Python). Won't make full songs "fast" (3060 is the
    // ceiling), but removes the last software gap. Keep the numerics identical (greedy argmax + range mask + -1
    // out-of-range + FixInvalidCodes must be byte-for-byte).
    // ────────────────────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Upsamples BOTH tracks (vocal + accompaniment) together as a decode batch of 2, halving Stage-2 wall-clock versus two sequential <see cref="Upsample"/> passes. Numerically identical: each track keeps its own KV cache and per-sequence attention; only the projections/MLP run as one batched GEMM. Requires equal track lengths (their windows advance in lockstep); falls back to two sequential passes otherwise.</summary>
    public (int[][] vocalCodes, int[][] accompCodes) UpsampleBoth(IBackend backend, ReadOnlySpan<int> vocalCb0, ReadOnlySpan<int> accompCb0)
    {
        ThrowIfDisposed();
        int t = vocalCb0.Length;
        // Unequal lengths would desync the two caches' windows — fall back to the exact per-track path.
        if (accompCb0.Length != t)
            return (Upsample(backend, vocalCb0), Upsample(backend, accompCb0));
        // Batched B=2 (two KV caches + batched activations) OOMs past one window: VRAM creeps to ~11.4 GB even with
        // per-window/per-64-frame TrimMemoryPool (the growth isn't in the stream-ordered pool the trim reclaims —
        // likely activation-cache accumulation in the B=2 path). Until that's root-caused, long songs use the
        // lighter sequential per-track path (one cache at a time) so they complete instead of OOMing.
        if (t > WindowFrames)
            return (Upsample(backend, vocalCb0), Upsample(backend, accompCb0));

        int[][] vCodes = new int[_cfg.NumCodebooks][];
        int[][] aCodes = new int[_cfg.NumCodebooks][];
        for (int k = 0; k < _cfg.NumCodebooks; k++) { vCodes[k] = new int[t]; aCodes[k] = new int[t]; }
        for (int i = 0; i < t; i++) { vCodes[0][i] = vocalCb0[i]; aCodes[0][i] = accompCb0[i]; }
        if (t == 0) return (vCodes, aCodes);

        for (int w0 = 0; w0 < t; w0 += WindowFrames)
        {
            int wlen = Math.Min(WindowFrames, t - w0);
            UpsampleWindowBoth(backend, vocalCb0.Slice(w0, wlen), accompCb0.Slice(w0, wlen), vCodes, aCodes, w0);
        }
        FixInvalidCodes(vCodes);
        FixInvalidCodes(aCodes);
        return (vCodes, aCodes);
    }

    /// <summary>One window for both tracks as a B=2 batch. Prefills each track's own prompt into its own <see cref="FixedKvCache"/> (prompts differ since cb0 differs, but lengths match), then runs the per-frame + 7-residual loop batched via <see cref="Qwen2Model.ForwardBatchDecode"/>. Mirrors <see cref="UpsampleWindow"/> exactly per track (same feed order, argmax, out-of-range marking).</summary>
    private void UpsampleWindowBoth(IBackend backend, ReadOnlySpan<int> vCb0, ReadOnlySpan<int> aCb0,
        int[][] vCodes, int[][] aCodes, int frameOffset)
    {
        int wlen = vCb0.Length;                                         // == aCb0.Length (guarded by caller)
        int promptLen = 2 + wlen + 1;
        int seqMax = promptLen + wlen * (_cfg.NumCodebooks - 1) + wlen + 8;
        int cacheCap = Math.Min(_cfg.Stage2.MaxPositionEmbeddings, seqMax);
        FixedKvCache cacheV = _lm.CreateFixedCache(cacheCap);
        FixedKvCache cacheA = _lm.CreateFixedCache(cacheCap);
        try
        {
            int h = _cfg.Stage2.HiddenSize;
            int vocab = _cfg.Stage2.VocabSize;
            int nResidual = _cfg.NumCodebooks - 1;                      // 7
            FixedKvCache[] caches = [cacheV, cacheA];

            // Prefill each track's prompt into its own cache (per-sequence — prompts differ by cb0).
            _lm.Forward(backend, BuildWindowPrompt(vCb0), 1, 0, cacheV).Dispose();
            _lm.Forward(backend, BuildWindowPrompt(aCb0), 1, 0, cacheA).Dispose();

            bool prof = Environment.GetEnvironmentVariable("HARTSY_YUE_PROFILE") == "1";
            long tProj = 0, tArg = 0, tFeed = 0;   // proj+D2H-drain / host-argmax / feed-queue (ticks)
            for (int fr = 0; fr < wlen; fr++)
            {
                // Feed both tracks' cb0 token for this frame (batched); resulting hidden predicts residual cb1.
                Tensor embeds = new(new TensorShape(1, 2, h), DType.F32);
                _lm.EmbedLookup(embeds, [vCb0[fr] + GlobalOffset, aCb0[fr] + GlobalOffset], 1, 2);
                Tensor hidden = _lm.ForwardBatchDecode(backend, embeds, [cacheV.CurrentLength, cacheA.CurrentLength], caches);
                embeds.Dispose();
                if (cacheV.CurrentLength >= cacheCap - (nResidual + 2)) { hidden.Dispose(); break; }

                for (int j = 1; j <= nResidual; j++)                    // codebooks 1..7
                {
                    long ts = prof ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                    Tensor logitsT = _lm.ProjectLogits(backend, hidden, 1, 2);  // [1,2,vocab] (row per track)
                    hidden.Dispose();
                    float* lp = (float*)logitsT.DataPointer;                    // D2H sync drains the queued GPU work
                    if (prof) { long n = System.Diagnostics.Stopwatch.GetTimestamp(); tProj += n - ts; ts = n; }

                    // Greedy argmax over the residual range [46358,53526) per track (== upstream block_list mask).
                    int bestV = ArgmaxResidual(lp, 0, vocab);
                    int bestA = ArgmaxResidual(lp, 1, vocab);
                    logitsT.Dispose();
                    if (prof) { long n = System.Diagnostics.Stopwatch.GetTimestamp(); tArg += n - ts; ts = n; }

                    // Decode assuming this slot is codebook j; out-of-range -> -1 (patched later).
                    int decV = bestV - (GlobalOffset + j * CodebookSize);
                    vCodes[j][frameOffset + fr] = (uint)decV < (uint)CodebookSize ? decV : -1;
                    int decA = bestA - (GlobalOffset + j * CodebookSize);
                    aCodes[j][frameOffset + fr] = (uint)decA < (uint)CodebookSize ? decA : -1;

                    // Autoregress: both tracks' emitted tokens enter their caches for subsequent positions.
                    Tensor rEmbeds = new(new TensorShape(1, 2, h), DType.F32);
                    _lm.EmbedLookup(rEmbeds, [bestV, bestA], 1, 2);
                    hidden = _lm.ForwardBatchDecode(backend, rEmbeds, [cacheV.CurrentLength, cacheA.CurrentLength], caches);
                    rEmbeds.Dispose();
                    if (prof) tFeed += System.Diagnostics.Stopwatch.GetTimestamp() - ts;
                }
                hidden.Dispose();
            }
            if (prof)
            {
                double f = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                Core.Logging.Logs.Info($"[YuE-S2-prof] {wlen} fr: proj+sync={tProj * f:0}ms argmax={tArg * f:0}ms feed={tFeed * f:0}ms");
            }
        }
        finally
        {
            cacheV.Dispose();
            cacheA.Dispose();
        }
    }

    /// <summary>Greedy argmax over the residual range [<see cref="ResidualLo"/>,<see cref="ResidualHi"/>) for row <paramref name="row"/> of a <c>[1, B, vocab]</c> logits buffer (same tie-break as the sequential path).</summary>
    private static int ArgmaxResidual(float* logits, int row, int vocab)
    {
        float* r = logits + (long)row * vocab;
        int best = ResidualLo;
        float bestv = float.NegativeInfinity;
        for (int v = ResidualLo; v < ResidualHi; v++)
            if (r[v] > bestv) { bestv = r[v]; best = v; }
        return best;
    }

    /// <summary>Builds a window's prompt <c>[SOA][stage_1] + cb0(window)+GlobalOffset + [stage_2]</c>.</summary>
    private int[] BuildWindowPrompt(ReadOnlySpan<int> cb0)
    {
        int wlen = cb0.Length;
        int[] prompt = new int[2 + wlen + 1];
        prompt[0] = _soa;
        prompt[1] = _stage1Tok;
        for (int i = 0; i < wlen; i++) prompt[2 + i] = cb0[i] + GlobalOffset;  // cb0 outline (k=0 offset)
        prompt[^1] = _stage2Tok;
        return prompt;
    }

    /// <summary>One independent window: prime <c>[SOA][stage_1] + cb0(window) + [stage_2]</c>, then for each frame feed its cb0 token and greedily emit the 7 residual tokens (cb1..cb7). Uses an incremental KV cache — numerically identical to upstream's re-prefill-per-frame loop, O(T) instead of O(T²).</summary>
    private void UpsampleWindow(IBackend backend, ReadOnlySpan<int> cb0, int[][] codes, int frameOffset)
    {
        int wlen = cb0.Length;
        int promptLen = 2 + wlen + 1;                                   // [soa,stage_1] + cb0 outline + [stage_2]
        int seqMax = promptLen + wlen * (_cfg.NumCodebooks - 1) + wlen + 8; // + per-frame (cb0 + 7 residuals)
        int cacheCap = Math.Min(_cfg.Stage2.MaxPositionEmbeddings, seqMax);
        using IKvCache cache = _lm.CreateDecodeCache(cacheCap);

        int[] prompt = BuildWindowPrompt(cb0);

        int h = _cfg.Stage2.HiddenSize;
        int nResidual = _cfg.NumCodebooks - 1;                          // 7
        Tensor hidden = _lm.Forward(backend, prompt, 1, 0, cache);
        try
        {
            for (int t = 0; t < wlen; t++)
            {
                // Feed this frame's cb0 token; the resulting hidden predicts residual codebook 1.
                int[] cb0Step = [cb0[t] + GlobalOffset];
                hidden.Dispose();
                hidden = _lm.Forward(backend, cb0Step, 1, cache.CurrentLength, cache);
                if (cache.CurrentLength >= cacheCap - (nResidual + 2)) break;

                for (int j = 1; j <= nResidual; j++)                    // codebooks 1..7
                {
                    Tensor last = SliceLast(hidden, h);
                    hidden.Dispose();
                    Tensor logitsT = _lm.ProjectLogits(backend, last, 1, 1);
                    last.Dispose();
                    float* lp = (float*)logitsT.DataPointer;

                    // Greedy argmax over the residual range [46358,53526) (== upstream block_list mask).
                    int best = ResidualLo;
                    float bestv = float.NegativeInfinity;
                    for (int v = ResidualLo; v < ResidualHi; v++)
                        if (lp[v] > bestv) { bestv = lp[v]; best = v; }
                    logitsT.Dispose();

                    // Decode assuming this slot is codebook j; out-of-range -> -1 (patched later).
                    int decoded = best - (GlobalOffset + j * CodebookSize);
                    codes[j][frameOffset + t] = (uint)decoded < (uint)CodebookSize ? decoded : -1;

                    // Autoregress: the emitted token must enter the cache for subsequent positions.
                    int[] resStep = [best];
                    hidden = _lm.Forward(backend, resStep, 1, cache.CurrentLength, cache);
                }
            }
        }
        finally
        {
            hidden.Dispose();
        }
    }

    /// <summary>Upstream "fix invalid codes": any residual marked invalid (-1) is replaced with the most frequent valid value in its codebook row (falling back to 0 for an all-invalid row).</summary>
    private void FixInvalidCodes(int[][] codes)
    {
        Span<int> counts = stackalloc int[CodebookSize];
        for (int k = 1; k < codes.Length; k++)
        {
            int[] row = codes[k];
            bool anyInvalid = false;
            counts.Clear();
            foreach (int v in row)
            {
                if ((uint)v < (uint)CodebookSize) counts[v]++;
                else anyInvalid = true;
            }
            if (!anyInvalid) continue;
            int mode = 0, modeCount = -1;
            for (int v = 0; v < CodebookSize; v++)
                if (counts[v] > modeCount) { modeCount = counts[v]; mode = v; }
            for (int i = 0; i < row.Length; i++)
                if ((uint)row[i] >= (uint)CodebookSize) row[i] = mode;
        }
    }

    private static Tensor SliceLast(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        return last;
    }

    public IEnumerable<Tensor> EnumerateWeights() => _lm.EnumerateWeights();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(YueStage2Lm));
    }
}
